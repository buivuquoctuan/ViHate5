"""Meaningful checks for sequence shapes, padding, CTC feasibility and gradients."""

import pytest
import torch
from torch import nn

from crnn.codec import Alphabet
from crnn.data import BATCH_PADDING_POLICY, collate_words
from crnn.evaluate import load_checkpoint
from crnn.model import CRNN
from crnn.train import encode_targets, validate_ctc_alignment


def test_variable_width_forward_and_ctc_backward():
    torch.manual_seed(7)
    model = CRNN(num_classes=37, hidden_size=8)
    images = torch.randn(2, 1, 32, 75)
    log_probs, lengths = model(images, torch.tensor([75, 64]))
    assert log_probs.shape == (18, 2, 37)
    assert lengths.tolist() == [18, 16]
    assert torch.allclose(log_probs.exp().sum(2), torch.ones(18, 2), atol=1e-6)
    alphabet = Alphabet()
    labels = ["book", "a1"]
    validate_ctc_alignment(labels, lengths)
    targets, target_lengths = encode_targets(labels, alphabet, torch.device("cpu"))
    loss = nn.CTCLoss(zero_infinity=False)(log_probs, targets, lengths, target_lengths)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.cnn[0].weight.grad is not None
    assert torch.isfinite(model.cnn[0].weight.grad).all()
    assert model.cnn[0].weight.grad.abs().sum() > 0
    assert model.rnn.weight_ih_l0.grad.abs().sum() > 0


def test_backward_lstm_ignores_padded_tail():
    torch.manual_seed(11)
    model = CRNN(num_classes=4, hidden_size=8).eval()

    class FixedCNN(nn.Module):
        """Isolate recurrent behavior from CNN boundary effects."""
        def __init__(self, features):
            super().__init__()
            self.features = features

        def forward(self, images):
            return self.features

    valid = torch.randn(1, 512, 1, 5)
    padded = torch.cat((valid, torch.randn(1, 512, 1, 4) * 100), dim=3)
    with torch.no_grad():
        model.cnn = FixedCNN(valid)
        short, _ = model(torch.zeros(1, 1, 32, 20), torch.tensor([20]))
        model.cnn = FixedCNN(padded)
        long, _ = model(torch.zeros(1, 1, 32, 36), torch.tensor([20]))
    assert torch.equal(short, long[:5])


def test_full_model_prediction_is_independent_of_other_word_widths():
    """White margin must also isolate the CNN boundary, before packed BiLSTM."""
    torch.manual_seed(17)
    alphabet = Alphabet()
    model = CRNN(num_classes=alphabet.num_classes, hidden_size=8).eval()

    def sample(width, identifier):
        return {"image": torch.rand(1, 32, width) * 2 - 1,
                "width": width, "id": identifier, "label": "book",
                "image_path": f"/{identifier}.png"}

    short = sample(80, "short")
    single_batch = collate_words([short])
    mixed_batch = collate_words([short, sample(120, "long")])
    with torch.inference_mode():
        alone, single_lengths = model(single_batch["images"], single_batch["widths"])
        together, mixed_lengths = model(mixed_batch["images"], mixed_batch["widths"])
    valid_steps = int(single_lengths[0])
    assert single_lengths[0] == mixed_lengths[0]
    torch.testing.assert_close(alone[:valid_steps, 0], together[:valid_steps, 0],
                               atol=1e-5, rtol=0)
    assert (alphabet.decode_greedy(alone, single_lengths)[0] ==
            alphabet.decode_greedy(together, mixed_lengths)[0])


@pytest.mark.parametrize("format_version,padding", [(1, BATCH_PADDING_POLICY), (2, "old_padding")])
def test_checkpoint_rejects_legacy_or_different_padding(tmp_path, format_version, padding):
    path = tmp_path / "incompatible.pt"
    torch.save({"format_version": format_version, "batch_padding": padding}, path)
    with pytest.raises(ValueError, match="format|padding"):
        load_checkpoint(path, torch.device("cpu"))


def test_ctc_repeated_characters_require_blank_time_steps():
    validate_ctc_alignment(["book"], torch.tensor([5]))
    with pytest.raises(ValueError, match="requires 5 time steps but has 4"):
        validate_ctc_alignment(["book"], torch.tensor([4]), ["word-1"])
    with pytest.raises(ValueError, match="requires 5 time steps but has 3"):
        validate_ctc_alignment(["aaa"], torch.tensor([3]))


def test_model_rejects_wrong_height_and_impossible_widths():
    model = CRNN(num_classes=4, hidden_size=8)
    with pytest.raises(ValueError, match="shape"):
        model(torch.zeros(1, 1, 48, 64), torch.tensor([64]))
    with pytest.raises(ValueError, match="valid image widths"):
        model(torch.zeros(1, 1, 32, 64), torch.tensor([65]))


def test_default_dropout_preserves_legacy_weights_and_predictions():
    torch.manual_seed(29)
    legacy_config_model = CRNN(**{"num_classes": 37, "hidden_size": 8}).eval()
    torch.manual_seed(29)
    explicit_zero_model = CRNN(num_classes=37, hidden_size=8, dropout=0.0).eval()
    # Dropout adds no parameters/buffers, so format-2 CNN/RNN keys stay intact.
    assert list(legacy_config_model.state_dict()) == list(explicit_zero_model.state_dict())
    assert all(key.startswith(("cnn.", "rnn.", "classifier."))
               for key in legacy_config_model.state_dict())
    explicit_zero_model.load_state_dict(legacy_config_model.state_dict(), strict=True)
    images = torch.rand(1, 1, 32, 104) * 2 - 1
    widths = torch.tensor([80])
    with torch.inference_mode():
        original, _ = legacy_config_model(images, widths)
        current, _ = explicit_zero_model(images, widths)
    assert torch.equal(original, current)


def test_dropout_changes_training_samples_but_is_disabled_for_evaluation():
    torch.manual_seed(31)
    model = CRNN(num_classes=37, hidden_size=8, dropout=0.2)
    images = torch.rand(2, 1, 32, 104) * 2 - 1
    widths = torch.tensor([80, 80])
    with torch.no_grad():
        first, _ = model(images, widths)
        second, _ = model(images, widths)
        assert not torch.equal(first, second)
        model.eval()
        eval_first, _ = model(images, widths)
        eval_second, _ = model(images, widths)
    assert torch.equal(eval_first, eval_second)

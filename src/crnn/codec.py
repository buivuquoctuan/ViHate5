"""Alphabet and CTC decoding. The blank is never a target character."""

import torch

DEFAULT_CHARACTERS = "0123456789abcdefghijklmnopqrstuvwxyz"


class Alphabet:
    def __init__(self, characters: str = DEFAULT_CHARACTERS):
        if not characters or len(set(characters)) != len(characters):
            raise ValueError("Alphabet must contain unique, nonempty characters.")
        self.characters = characters
        self.blank_id = 0
        self._ids = {char: index + 1 for index, char in enumerate(characters)}

    @property
    def num_classes(self) -> int:
        return len(self.characters) + 1

    def encode(self, text: str) -> list[int]:
        if not text:
            raise ValueError("Empty ground-truth word is not supported.")
        try:
            return [self._ids[char] for char in text]
        except KeyError as exc:
            raise ValueError(f"Unsupported character {exc.args[0]!r} in {text!r}") from exc

    def decode_greedy(self, log_probs: torch.Tensor, input_lengths) -> list[str]:
        if log_probs.ndim != 3 or log_probs.shape[2] != self.num_classes:
            raise ValueError("Expected [T,B,C] probabilities with the matching alphabet.")
        paths = log_probs.detach().argmax(dim=2).cpu()
        lengths = torch.as_tensor(input_lengths).cpu().tolist()
        if len(lengths) != paths.shape[1]:
            raise ValueError("One input length is required per sample.")
        words = []
        for sample, length in enumerate(lengths):
            if not 0 < length <= paths.shape[0]:
                raise ValueError("Invalid CTC input length.")
            previous = self.blank_id
            chars = []
            for value in paths[:length, sample].tolist():
                # Collapse consecutive duplicates FIRST, then remove blanks.
                if value != previous and value != self.blank_id:
                    chars.append(self.characters[value - 1])
                previous = value
            words.append("".join(chars))
        return words


def minimum_ctc_steps(text: str) -> int:
    """Adjacent duplicate target letters each need an intervening blank."""
    return len(text) + sum(a == b for a, b in zip(text, text[1:]))

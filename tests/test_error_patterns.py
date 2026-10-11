import pytest

from crnn.error_patterns import alignment
from crnn.metrics import edit_distance


@pytest.mark.parametrize("reference,prediction", [
    ("book", "bok"), ("book", "boook"), ("private", "pritate"),
    ("", "ab"), ("ab", ""), ("abab", "baba"), ("123abc", "a21bc")])
def test_alignment_reconstructs_strings_and_minimum_cost(reference, prediction):
    operations = alignment(reference, prediction)
    assert "".join(op["reference"] for op in operations) == reference
    assert "".join(op["prediction"] for op in operations) == prediction
    assert sum(op["operation"] != "match" for op in operations) == edit_distance(reference, prediction)
    assert operations == alignment(reference, prediction)


def test_edit_types_are_distinct():
    assert [op["operation"] for op in alignment("cat", "cut")] == ["match", "substitution", "match"]
    assert sum(op["operation"] == "deletion" for op in alignment("book", "bok")) == 1
    assert sum(op["operation"] == "insertion" for op in alignment("book", "boook")) == 1

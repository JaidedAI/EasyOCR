import numpy as np
import pytest

from easyocr.utils import CTCLabelConverter, ctcBeamSearch


def _peaked(path, num_classes):
    """Probability matrix that puts (almost) all mass on the given path."""
    mat = np.full((len(path), num_classes), 1e-4)
    mat[np.arange(len(path)), path] = 1.0
    return mat / mat.sum(axis=1, keepdims=True)


@pytest.mark.parametrize(
    "text, path",
    [
        ("hello", [1, 2, 3, 0, 3, 4]),
        ("hello", [1, 2, 3, 3, 0, 3, 4]),
        ("ee", [2, 0, 2]),
        ("lol", [3, 4, 3]),
        ("lllo", [3, 0, 3, 0, 3, 4]),
    ],
)
def test_beam_search_keeps_repeated_characters(text, path):
    # alphabet: blank, h, e, l, o
    converter = CTCLabelConverter("helo")
    mat = _peaked(path, len(converter.character))
    greedy = converter.decode_greedy(np.array(path), [len(path)])[0]
    assert greedy == text
    assert ctcBeamSearch(mat, converter.character, converter.ignore_idx, None, beamWidth=5) == text


def test_beam_search_does_not_double_count_blank_paths():
    # alphabet: blank, a, b, c. The most probable labeling is "ab" (also the greedy result).
    # Extending a beam with the blank class as well double-counts paths ending in a blank
    # and used to give "aab" here.
    classes = ["[blank]", "a", "b", "c"]
    mat = np.array(
        [
            [0.0, 0.936, 0.0, 0.064],
            [0.409, 0.591, 0.001, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.998, 0.002],
        ]
    )
    assert ctcBeamSearch(mat, classes, [0], None, beamWidth=5) == "ab"

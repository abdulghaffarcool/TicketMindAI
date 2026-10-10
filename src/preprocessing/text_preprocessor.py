
import re


def tokenize(text: str) -> list[str]:
    """Convert text into lowercase word tokens."""
    return re.findall(r"\b\w+\b", text.lower())


def text_to_sequence(
    text: str,
    word_to_id: dict[str, int]
) -> list[int]:
    """Convert tokens into IDs, using UNK for unseen words."""
    unk_id = word_to_id["<UNK>"]

    return [
        word_to_id.get(token, unk_id)
        for token in tokenize(text)
    ]


def pad_sequence(
    sequence: list[int],
    max_length: int,
    pad_id: int
) -> list[int]:
    """Truncate or pad a sequence to a fixed length."""
    sequence = sequence[:max_length]

    return sequence + [pad_id] * (max_length - len(sequence))


def preprocess_text(
    text: str,
    word_to_id: dict[str, int],
    max_length: int,
    pad_id: int
) -> list[int]:
    """Convert raw text into a fixed-length sequence of token IDs."""
    sequence = text_to_sequence(text, word_to_id)

    return pad_sequence(sequence, max_length, pad_id)
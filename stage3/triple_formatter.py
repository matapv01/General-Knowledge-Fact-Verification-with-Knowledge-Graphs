"""Triple-to-string formatting for BERT input.

Two strategies are provided:

- ``format_triple`` (default): plain whitespace-separated concatenation.
- ``format_triple_with_separators``: adds [H] [R] [T] markers — these are
  ordinary tokens to a pretrained BERT but their embeddings will be fine-tuned.

The choice is controlled via ``configs/default.yaml`` (``triple_format``).
"""


def format_triple(head: str, relation: str, tail: str) -> str:
    return f"{head} {relation} {tail}"


def format_triple_with_separators(head: str, relation: str, tail: str) -> str:
    return f"[H] {head} [R] {relation} [T] {tail}"


_FORMATTERS = {
    "plain": format_triple,
    "separators": format_triple_with_separators,
}


def get_formatter(name: str = "plain"):
    if name not in _FORMATTERS:
        raise ValueError(f"Unknown triple format: {name}. Choices: {list(_FORMATTERS)}")
    return _FORMATTERS[name]


DEFAULT_FORMATTER = format_triple

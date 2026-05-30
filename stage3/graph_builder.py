"""Build an evidence graph from a triple pool.

Each node = (claim, triple). The pool is padded with NULL nodes up to
``max_nodes`` so a batch tensor of fixed shape can be produced.
"""

from typing import Dict, List, Tuple

from .triple_formatter import get_formatter

Triple = Tuple[str, str, str]

NULL_TRIPLE_TEXT = "no_triple no_relation no_triple"


def build_graph(
    claim: str,
    triple_pool: List[Triple],
    max_nodes: int = 10,
    formatter_name: str = "plain",
) -> Dict:
    """Return a fixed-size graph dict.

    {
      'claim':       str,
      'nodes':       [{triple, triple_text, is_null}, ...]   # len == max_nodes
    }
    """
    formatter = get_formatter(formatter_name)
    nodes: List[Dict] = []

    for triple in triple_pool[:max_nodes]:
        head, rel, tail = triple
        nodes.append(
            {
                "triple": tuple(triple),
                "triple_text": formatter(head, rel, tail),
                "is_null": False,
            }
        )

    while len(nodes) < max_nodes:
        nodes.append(
            {
                "triple": None,
                "triple_text": NULL_TRIPLE_TEXT,
                "is_null": True,
            }
        )

    return {"claim": claim, "nodes": nodes}

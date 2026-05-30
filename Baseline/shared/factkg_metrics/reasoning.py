TYPE_NAMES = {
    "num1":        "One-hop",
    "multi claim": "Conjunction",
    "existence":   "Existence",
    "multi hop":   "Multi-hop",
    "negation":    "Negation",
}

TYPE_IDS = {
    "num1":        0,
    "multi claim": 2,
    "existence":   3,
    "multi hop":   1,
    "negation":    4,
}

# Display order matching Table 3 in the FactKG paper
TABLE_ORDER = [
    (0, "One-hop"),
    (2, "Conjunction"),
    (3, "Existence"),
    (1, "Multi-hop"),
    (4, "Negation"),
]

# When a sample has multiple tags, the first match in this list wins
TYPE_PRIORITY = ["negation", "num1", "multi hop", "multi claim", "existence"]


def get_type_id(type_tags):
    """Return the integer ID for a sample's reasoning type.

    Returns -1 if none of the known tags appear in type_tags.
    """
    for tag in TYPE_PRIORITY:
        if tag in type_tags:
            return TYPE_IDS[tag]
    return -1

"""Build a triple pool from KG-GPT Stage 2 output.

This faithfully replicates KG-GPT's evidence construction
(``kg-gpt/factkg_test.py::get_answer`` lines ~294-465), so the triple pool
keeps everything KG-GPT produces:

- per-sub-claim candidate triples (forward + reverse relation forms),
- **type-entity expansion** (when one endpoint is a DBpedia *type*, expand the
  other endpoint's relation to all its actual tails),
- **multi-hop / cross-sub-claim bridging** (the 3-hop ``additional`` logic and
  the ``before_final`` cross-product that connects entities living in
  different sub-claims through an intermediate entity),
- de-duplication + ``graph_extractor`` pruning.

The only change versus KG-GPT is the return type: a deterministic, ordered
list of ``(head, rel, tail)`` tuples (capped at ``max_triples``) instead of
the input to an LLM verification prompt. The ordering produced by
``graph_extractor`` is deterministic, which fixes the earlier
``list(set(...))`` non-determinism.
"""

import os
import sys
from typing import Dict, List, Tuple

# Reuse KG-GPT's graph_extractor verbatim.
_KGGPT_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "kg-gpt")
)
if _KGGPT_DIR not in sys.path:
    sys.path.insert(0, _KGGPT_DIR)

from factkg_test import graph_extractor  # noqa: E402

Triple = Tuple[str, str, str]


def build_subclaim_triples(relations: List[str], entity_set: List[str]) -> List[list]:
    """Materialize candidate triples for ONE sub-claim.

    Mirrors KG-GPT's ``total_triples`` construction. For a 1-entity sub-claim
    we emit 2-element ``[entity, rel]`` stubs (forward + reverse) that the
    grounding step later expands; for >=2 entities we emit ordered
    ``[e1, rel, e2]`` triples (forward + reverse) for every distinct pair.
    """
    total_triples: List[list] = []
    for rel in relations:
        if len(entity_set) == 1:
            total_triples.append([entity_set[0], rel])
            total_triples.append([entity_set[0], "~" + rel])
        for fir in range(len(entity_set)):
            for sec in range(len(entity_set)):
                if fir != sec:
                    total_triples.append([entity_set[fir], rel, entity_set[sec]])
                    total_triples.append([entity_set[fir], "~" + rel, entity_set[sec]])
    return total_triples


def build_triple_pool(
    total_evidence: List[List[list]],
    KG: dict,
    type_dict: dict,
    gt_entities: List[str],
    max_triples: int = 30,
) -> List[Triple]:
    """Replicate KG-GPT evidence grounding + bridging, return ordered triples.

    Args:
        total_evidence: list (per sub-claim) of candidate triple lists, each
            produced by ``build_subclaim_triples``.
        KG: DBpedia dict {entity: {relation: [tails]}}.
        type_dict: type-relation lookup (also used to detect type entities).
        gt_entities: the claim's full entity set (KG-GPT's ``gt_entities``).
        max_triples: cap on returned triples.
    """
    # ---- 3-hop relation hints (KG-GPT "additional") ------------------------
    additional: List[str] = []
    for evi_set in total_evidence:
        try:
            _ = type_dict[evi_set[0][0]]
            _ = type_dict[evi_set[0][2]]
            for trip in evi_set:
                additional.append(trip[1])
        except Exception:
            continue

    # ---- Ground each sub-claim's triples in the KG -------------------------
    before_final: List[List[list]] = []
    for evi_set in total_evidence:
        before_final_evi: List[list] = []
        for trip in evi_set:
            try:
                _ = type_dict[trip[0]]      # head is a TYPE → skip (handled via tail)
                continue
            except Exception:
                try:
                    _ = type_dict[trip[2]]  # tail is a TYPE → expand head's relation
                    try:
                        tails = KG[trip[0]][trip[1]]
                        for tail in tails:
                            before_final_evi.append([trip[0], trip[1], tail])
                    except Exception:
                        continue
                except Exception:           # neither endpoint is a type
                    try:
                        if len(trip) == 2:
                            before_final_evi.append(trip)
                        elif trip[2] in KG[trip[0]][trip[1]]:
                            before_final_evi.append(trip)
                    except Exception:
                        pass
        if len(before_final_evi) > 0:
            before_final.append(before_final_evi)

    # ---- Connect evidence across sub-claims (multi-hop bridging) -----------
    final_evidence: List[list] = []
    for chunk in before_final:
        find_gt = 0
        for trip in chunk:
            if len(trip) != 2 and trip[0] in gt_entities and trip[2] in gt_entities:
                final_evidence.append(trip)
                find_gt = 1

        if find_gt == 1:
            continue

        if len(before_final) == 1:
            for trip in chunk:
                if len(trip) == 2:
                    try:
                        tails = KG[trip[0]][trip[1]]
                        for tail in tails:
                            final_evidence.append([trip[0], trip[1], tail])
                    except Exception:
                        continue
            break

        additional = list(set(additional))

        if len(additional) != 0:
            for sec_chunk in before_final:
                if chunk == sec_chunk:
                    continue
                for trip in chunk:
                    if len(trip) == 2:
                        continue
                    for sec_trip in sec_chunk:
                        if len(sec_trip) == 2:
                            continue
                        for rel_ in additional:
                            for trip_id in [0, 2]:
                                for sec_trip_id in [0, 2]:
                                    for rel_add in ["", "~"]:
                                        try:
                                            if rel_add == "" and "~" in rel_:
                                                if trip[trip_id] in KG[sec_trip[sec_trip_id]][rel_.split("~")[1]]:
                                                    final_evidence.append(trip)
                                                    final_evidence.append(sec_trip)
                                                    final_evidence.append([sec_trip[sec_trip_id], rel_.split("~")[1], trip[trip_id]])
                                        except Exception:
                                            pass
                                        try:
                                            if trip[trip_id] in KG[sec_trip[sec_trip_id]][rel_add + rel_]:
                                                final_evidence.append(trip)
                                                final_evidence.append(sec_trip)
                                                final_evidence.append([sec_trip[sec_trip_id], rel_add + rel_, trip[trip_id]])
                                        except Exception:
                                            pass
        else:
            for sec_chunk in before_final:
                if chunk == sec_chunk:
                    continue
                for trip in chunk:
                    for sec_trip in sec_chunk:
                        if len(trip) == 2 or len(sec_trip) == 2:
                            continue
                        if (trip[0] in sec_trip and trip[0] not in gt_entities) or (
                            trip[2] in sec_trip and trip[2] not in gt_entities
                        ):
                            final_evidence.append(trip)
                            final_evidence.append(sec_trip)

    # ---- De-duplicate (normalize reverse relations) ------------------------
    new_final_evidence: List[list] = []
    for trip in final_evidence:
        if "~" in trip[1]:
            flipped = [trip[2], trip[1].split("~")[1], trip[0]]
            if flipped not in new_final_evidence and flipped not in final_evidence:
                new_final_evidence.append(flipped)
                continue
        else:
            if trip not in new_final_evidence:
                new_final_evidence.append(trip)

    # ---- KG-GPT graph pruning (deterministic order) ------------------------
    pruned = graph_extractor(new_final_evidence)

    # Keep only well-formed 3-element triples, as tuples, capped.
    pool: List[Triple] = []
    seen = set()
    for trip in pruned:
        if len(trip) >= 3:
            t = (trip[0], trip[1], trip[2])
            if t not in seen:
                seen.add(t)
                pool.append(t)
        if len(pool) >= max_triples:
            break
    return pool

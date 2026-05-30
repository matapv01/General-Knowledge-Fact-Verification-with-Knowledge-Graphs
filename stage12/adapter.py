"""
Stage 1+2 adapter — wraps KG-GPT pipeline but stops before final LLM verification.

KG-GPT (kg-gpt/factkg_test.py) inlines Stage 1 + Stage 2 inside `get_answer()`.
Following Option B in section 7.1 of the pipeline doc, we copy the relevant
logic here and reuse the helper functions (claim_divider_parse_answer,
relation_candidates, retrieval_relation_parse_answer) directly.

Output: triple_pool (list of (head, rel, tail)) instead of verification result.
"""

import os
import sys
import time
from typing import Dict, List, Tuple

import openai

# Import helpers from the existing KG-GPT repo (sibling folder).
_KGGPT_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "kg-gpt")
)
if _KGGPT_DIR not in sys.path:
    sys.path.insert(0, _KGGPT_DIR)

from factkg_test import (  # noqa: E402
    claim_divider_parse_answer,
    relation_candidates,
    retrieval_relation_parse_answer,
)

PROMPTS_DIR = os.path.join(_KGGPT_DIR, "prompts")


def _load_prompt(name: str) -> str:
    path = os.path.join(PROMPTS_DIR, name)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


class Stage12Adapter:
    """Run KG-GPT Stage 1 (sentence divide) + Stage 2 (relation retrieval),
    then build a triple pool from candidate relations.

    Args:
        dbpedia: KG dict {entity: {relation: [tails]}}
        type_dict: type-relation lookup built by kg-gpt/data/make_type_dict.py
        model_name: OpenAI model (e.g. "gpt-3.5-turbo-0613")
        top_k: number of relations to retrieve per sub-sentence
        max_tokens: GPT max tokens
        max_triples: cap on triple pool size returned per claim
    """

    def __init__(
        self,
        dbpedia: dict,
        type_dict: dict,
        model_name: str = "gpt-3.5-turbo-0613",
        top_k: int = 5,
        max_tokens: int = 1024,
        max_triples: int = 30,
    ):
        self.KG = dbpedia
        self.type_dict = type_dict
        self.model_name = model_name
        self.top_k = top_k
        self.max_tokens = max_tokens
        self.max_triples = max_triples

        self._sentence_divide_template = _load_prompt("sentence_divide_prompt.txt")
        self._relation_retrieval_template = _load_prompt("relation_retrieval_prompt.txt")

    # ------------------------------------------------------------------ #
    # Stage 1: sentence division                                          #
    # ------------------------------------------------------------------ #
    def _sentence_divide(self, claim: str, gt_entities: list) -> Dict[str, list]:
        """Call LLM to split a complex claim into atomic sub-sentences.

        Returns a dict {sub_sentence_text: entity_list}.
        """
        query = (
            self._sentence_divide_template
            .replace("<<<<CLAIM>>>>", claim)
            .replace("<<<<ENTITY_SET>>>>", str(gt_entities))
        )

        divided = None
        for _ in range(3):
            try:
                resp = openai.ChatCompletion.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": query},
                    ],
                    max_tokens=self.max_tokens,
                    temperature=0.2,
                    top_p=0.1,
                )
                content = resp["choices"][0]["message"]["content"]
                divided = claim_divider_parse_answer(content, gt_entities)
                if divided:
                    break
            except Exception as e:
                print("[stage12.adapter] sentence_divide error:", e)
                time.sleep(5)

        # Fallback: treat the whole claim as one atomic sentence
        if not divided:
            divided = {claim: list(gt_entities)}
        return divided

    # ------------------------------------------------------------------ #
    # Stage 2.2: top-K relation retrieval via LLM                         #
    # ------------------------------------------------------------------ #
    def _top_k_relations(self, sub_sentence: str, candidates: list) -> list:
        """Ask GPT to pick the top-K most relevant relations from candidates."""
        if len(candidates) <= self.top_k:
            return list(candidates)

        query = (
            self._relation_retrieval_template
            .replace("<<<<TOP_K>>>>", str(self.top_k))
            .replace("<<<<SENTENCE>>>>", sub_sentence)
            .replace("<<<<RELATION_SET>>>>", str(candidates))
        )

        for _ in range(3):
            try:
                resp = openai.ChatCompletion.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": query},
                    ],
                    max_tokens=self.max_tokens,
                    temperature=0.2,
                    top_p=0.1,
                )
                content = resp["choices"][0]["message"]["content"]
                picked = retrieval_relation_parse_answer(content)
                if picked:
                    # Keep only relations that exist in the candidate pool
                    return [r for r in picked if r in candidates] or list(candidates[: self.top_k])
            except Exception as e:
                print("[stage12.adapter] relation_retrieval error:", e)
                time.sleep(5)

        # Fallback: first K candidates
        return list(candidates[: self.top_k])

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #
    def process(self, claim: str, entity_set: list) -> dict:
        """Run Stage 1+2 on a single claim, return structured output with triple_pool.

        Returns:
            {
              'claim': str,
              'entity_set': list,
              'sub_sentences': [{'text', 'entities', 'top_k_relations'}, ...],
              'triple_pool': [(head, rel, tail), ...],
            }
        """
        from .triple_pool_builder import build_subclaim_triples, build_triple_pool

        # Stage 1
        divided = self._sentence_divide(claim, entity_set)

        # Stage 2 — for each sub-claim: candidate relations → top-K → triples.
        # ``total_evidence`` mirrors KG-GPT's per-sub-claim triple lists so the
        # builder can run KG-GPT's exact grounding + multi-hop bridging.
        sub_data = []
        total_evidence = []
        for sub_text, sub_entities in divided.items():
            try:
                cand_rels, normalized_entities = relation_candidates(
                    self.KG, self.type_dict, sub_entities
                )
            except Exception as e:
                print("[stage12.adapter] relation_candidates error:", e)
                cand_rels, normalized_entities = [], list(sub_entities)

            if len(cand_rels) == 0:
                chosen_rels = []
            elif len(cand_rels) < self.top_k:
                # KG-GPT skips the LLM call when there are fewer candidates than K.
                chosen_rels = list(cand_rels)
            else:
                chosen_rels = self._top_k_relations(sub_text, cand_rels)

            sub_data.append(
                {
                    "text": sub_text,
                    "entities": list(normalized_entities),
                    "top_k_relations": list(chosen_rels),
                }
            )

            sub_triples = build_subclaim_triples(chosen_rels, list(normalized_entities))
            if sub_triples:
                total_evidence.append(sub_triples)

        # Stage 2.3: ground + bridge into the final triple pool (KG-GPT logic)
        triple_pool = build_triple_pool(
            total_evidence,
            self.KG,
            self.type_dict,
            gt_entities=list(entity_set),
            max_triples=self.max_triples,
        )

        return {
            "claim": claim,
            "entity_set": list(entity_set),
            "sub_sentences": sub_data,
            "triple_pool": triple_pool,
        }

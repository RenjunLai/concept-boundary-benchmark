from __future__ import annotations

from dataclasses import dataclass

from .resources import Inventory


@dataclass(frozen=True)
class TruthTrace:
    candidate_id: str
    inside_a1: bool
    hits_positive_branch: bool
    hits_any_b: bool
    satisfies_constraint: bool
    a1_id: str
    a_hits: tuple[str, ...]
    b_hits: tuple[str, ...]


def satisfies(
    inventory: Inventory,
    candidate_id: str,
    a1_id: str,
    a_branch_ids: tuple[str, ...] = (),
    b_ids: tuple[str, ...] = (),
) -> TruthTrace:
    inside_a1 = candidate_id in inventory.usable_closure(a1_id)
    branch_hits = tuple(branch_id for branch_id in a_branch_ids if candidate_id in inventory.usable_closure(branch_id))
    hits_positive_branch = bool(branch_hits) if a_branch_ids else inside_a1
    b_hits = tuple(b_id for b_id in b_ids if candidate_id in inventory.usable_closure(b_id))
    satisfies_constraint = inside_a1 and hits_positive_branch and not b_hits
    a_hits = (a1_id,) + branch_hits if inside_a1 else branch_hits
    return TruthTrace(
        candidate_id=candidate_id,
        inside_a1=inside_a1,
        hits_positive_branch=hits_positive_branch,
        hits_any_b=bool(b_hits),
        satisfies_constraint=satisfies_constraint,
        a1_id=a1_id,
        a_hits=a_hits,
        b_hits=b_hits,
    )


def gold_bool(
    inventory: Inventory,
    candidate_id: str,
    a1_id: str,
    a_branch_ids: tuple[str, ...] = (),
    b_ids: tuple[str, ...] = (),
) -> bool:
    return satisfies(inventory, candidate_id, a1_id, a_branch_ids, b_ids).satisfies_constraint


def gold_pool(
    inventory: Inventory,
    candidate_ids: tuple[str, ...],
    a1_id: str,
    a_branch_ids: tuple[str, ...] = (),
    b_ids: tuple[str, ...] = (),
) -> tuple[int, ...]:
    return tuple(
        index
        for index, candidate_id in enumerate(candidate_ids, 1)
        if gold_bool(inventory, candidate_id, a1_id, a_branch_ids, b_ids)
    )

from __future__ import annotations

import pickle
import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from .config import ResourceConfig
from .schemas import ResourceObject

HAN_RE = re.compile(r"^[\u4e00-\u9fff]+$")


def clean_zh(value: str) -> bool:
    return isinstance(value, str) and bool(HAN_RE.match(value))


def unique(values: Iterable[str]) -> tuple[str, ...]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            out.append(value)
            seen.add(value)
    return tuple(out)


def is_proper_babelnet(item: dict) -> bool:
    joined = " ".join(str(value) for value in (item.get("sememes", []) or []))
    return "ProperName" in joined


def clean_zh_synonyms(item: dict) -> tuple[str, ...]:
    return unique(value for value in (item.get("zh_synonyms", []) or []) if clean_zh(value) and value != "弥")


def preferred_chinese_surface(item: dict) -> str:
    synonyms = clean_zh_synonyms(item)
    if item.get("bn") == "bn:00054288n":
        for preferred in ("迷因", "模因"):
            if preferred in synonyms:
                return preferred
    for value in synonyms:
        if value != "弥":
            return value
    return ""


class Inventory:
    def __init__(self, dataset_family: str, objects: dict[str, ResourceObject], audit_stats: dict | None = None):
        self.dataset_family = dataset_family
        self.objects = objects
        self.audit_stats = audit_stats or {}
        self.object_ids = frozenset(objects)
        self.children = {key: tuple(obj.children) for key, obj in objects.items()}
        self.parents = {key: tuple(obj.parents) for key, obj in objects.items()}

    def __len__(self) -> int:
        return len(self.objects)

    def get(self, object_id: str) -> ResourceObject:
        return self.objects[object_id]

    @lru_cache(maxsize=None)
    def closure(self, object_id: str) -> frozenset[str]:
        seen: set[str] = set()
        stack = list(self.children.get(object_id, ()))
        while stack:
            current = stack.pop()
            if current in seen or current not in self.objects:
                continue
            seen.add(current)
            stack.extend(self.children.get(current, ()))
        return frozenset(seen)

    @lru_cache(maxsize=None)
    def usable_closure(self, object_id: str) -> frozenset[str]:
        return self.closure(object_id) & self.object_ids

    def direct_usable_children(self, object_id: str, min_closure: int = 1) -> tuple[str, ...]:
        out = []
        for child in self.children.get(object_id, ()):
            if child in self.objects and len(self.usable_closure(child)) >= min_closure:
                out.append(child)
        return tuple(out)

    @lru_cache(maxsize=None)
    def sibling_pool(self, object_id: str) -> frozenset[str]:
        object_space = set(self.closure(object_id))
        object_space.add(object_id)
        siblings: set[str] = set()
        for parent in self.parents.get(object_id, ()):
            for child in self.children.get(parent, ()):
                if child != object_id and child in self.objects:
                    siblings.add(child)
        pool: set[str] = set()
        for sibling in siblings:
            pool.add(sibling)
            pool.update(self.usable_closure(sibling))
        return frozenset(pool - object_space)


def load_wordnet_inventory(config: ResourceConfig) -> Inventory:
    from nltk.corpus import wordnet as wn

    raw_synsets = list(wn.all_synsets("n"))
    raw_ids = {syn.name() for syn in raw_synsets}
    child_map: dict[str, set[str]] = defaultdict(set)
    parent_map: dict[str, set[str]] = defaultdict(set)
    for syn in raw_synsets:
        object_id = syn.name()
        for child in syn.hyponyms():
            child_id = child.name()
            if child_id in raw_ids:
                child_map[object_id].add(child_id)
                parent_map[child_id].add(object_id)

    objects: dict[str, ResourceObject] = {}
    clean_surface_count = 0
    gloss_count = 0
    for syn in raw_synsets:
        object_id = syn.name()
        lemmas = unique(
            lemma.replace("_", " ")
            for lemma in syn.lemma_names()
            if lemma and "_" not in lemma and lemma.isascii() and lemma.isalpha()
        )
        if lemmas:
            clean_surface_count += 1
        if syn.definition():
            gloss_count += 1
        if not lemmas or not syn.definition():
            continue
        objects[object_id] = ResourceObject(
            dataset_family="wordnet",
            object_id=object_id,
            surface=lemmas[0],
            gloss=syn.definition(),
            synonyms=lemmas,
            children=tuple(sorted(child_map.get(object_id, ()))),
            parents=tuple(sorted(parent_map.get(object_id, ()))),
            usable_flags={"noun": True, "clean_surface": True, "has_gloss": True},
            metadata={"pos": "n", "wordnet_name": object_id},
        )
    audit_stats = {
        "raw_noun_synset_count": len(raw_synsets),
        "clean_surface_count": clean_surface_count,
        "gloss_count": gloss_count,
        "usable_object_count": len(objects),
        "surface_coverage": clean_surface_count / len(raw_synsets) if raw_synsets else 0.0,
        "gloss_coverage": gloss_count / len(raw_synsets) if raw_synsets else 0.0,
        "usable_coverage": len(objects) / len(raw_synsets) if raw_synsets else 0.0,
    }
    return Inventory("wordnet", objects, audit_stats=audit_stats)


def load_babelnet_inventory(config: ResourceConfig) -> Inventory:
    path = Path(config.babelnet_path)
    with path.open("rb") as handle:
        data = pickle.load(handle)
    raw_items = {item["bn"]: item for item in data if item.get("pos") == "n"}
    raw_chinese_synonym_count = sum(1 for item in raw_items.values() if item.get("zh_synonyms"))
    raw_chinese_gloss_count = sum(
        1
        for item in raw_items.values()
        if any(isinstance(value, str) and value.strip() for value in (item.get("zh_glosses", []) or []))
    )
    raw_clean_chinese_surface_count = sum(1 for item in raw_items.values() if clean_zh_synonyms(item))
    child_map: dict[str, set[str]] = defaultdict(set)
    parent_map: dict[str, set[str]] = defaultdict(set)
    for bn, item in raw_items.items():
        for child in (item.get("rel", {}) or {}).get("hyponym", []) or []:
            if child in raw_items:
                child_map[bn].add(child)
                parent_map[child].add(bn)

    objects: dict[str, ResourceObject] = {}
    non_proper_count = 0
    clean_surface_count = 0
    gloss_count = 0
    clean_surface_and_gloss_count = 0
    for bn, item in raw_items.items():
        if is_proper_babelnet(item):
            continue
        non_proper_count += 1
        surface = preferred_chinese_surface(item)
        glosses = [value.strip() for value in (item.get("zh_glosses", []) or []) if isinstance(value, str) and value.strip()]
        if surface:
            clean_surface_count += 1
        if glosses:
            gloss_count += 1
        if surface and glosses:
            clean_surface_and_gloss_count += 1
        if not surface or not glosses:
            continue
        objects[bn] = ResourceObject(
            dataset_family="babelnet",
            object_id=bn,
            surface=surface,
            gloss=glosses[0],
            synonyms=clean_zh_synonyms(item),
            children=tuple(sorted(child_map.get(bn, ()))),
            parents=tuple(sorted(parent_map.get(bn, ()))),
            usable_flags={"noun": True, "clean_surface": True, "has_gloss": True, "non_proper": True},
            metadata={
                "pos": "n",
                "en_synonyms": tuple((item.get("en_synonyms") or [])[:8]),
                "surface_rule": "preferred_zh_surface_exact_filter_mi",
            },
        )
    noun_count = len(raw_items)
    audit_stats = {
        "raw_noun_synset_count": noun_count,
        "raw_chinese_synonym_count": raw_chinese_synonym_count,
        "raw_chinese_gloss_count": raw_chinese_gloss_count,
        "raw_clean_chinese_surface_count": raw_clean_chinese_surface_count,
        "raw_chinese_synonym_coverage": raw_chinese_synonym_count / noun_count if noun_count else 0.0,
        "raw_chinese_gloss_coverage": raw_chinese_gloss_count / noun_count if noun_count else 0.0,
        "raw_clean_chinese_surface_coverage": raw_clean_chinese_surface_count / noun_count if noun_count else 0.0,
        "non_proper_count": non_proper_count,
        "non_proper_clean_surface_count": clean_surface_count,
        "non_proper_gloss_count": gloss_count,
        "non_proper_clean_surface_and_gloss_count": clean_surface_and_gloss_count,
        "clean_surface_count": clean_surface_count,
        "gloss_count": gloss_count,
        "clean_surface_and_gloss_count": clean_surface_and_gloss_count,
        "usable_object_count": len(objects),
        "surface_coverage": clean_surface_count / noun_count if noun_count else 0.0,
        "gloss_coverage": gloss_count / noun_count if noun_count else 0.0,
        "usable_coverage": len(objects) / noun_count if noun_count else 0.0,
        "exact_filtered_surface": "弥",
    }
    return Inventory("babelnet", objects, audit_stats=audit_stats)


def load_inventories(config: ResourceConfig) -> dict[str, Inventory]:
    inventories: dict[str, Inventory] = {}
    families = set(config.dataset_families)
    if config.wordnet_enabled and "wordnet" in families:
        inventories["wordnet"] = load_wordnet_inventory(config)
    if config.babelnet_enabled and "babelnet" in families:
        inventories["babelnet"] = load_babelnet_inventory(config)
    return inventories


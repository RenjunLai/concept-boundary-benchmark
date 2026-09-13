from __future__ import annotations

from collections import deque

from .resources import Inventory


class BitsetHelper:
    def __init__(self, inventory: Inventory):
        self.inventory = inventory
        self.object_ids = tuple(sorted(inventory.objects))
        self.index = {object_id: index for index, object_id in enumerate(self.object_ids)}
        self.all_mask = (1 << len(self.object_ids)) - 1
        self._bit_cache: dict[str, int] = {}
        self._closure_cache: dict[str, int] = {}
        self._sibling_cache: dict[str, int] = {}
        self._children_cache: dict[str, tuple[str, ...]] = {}
        self._parents_cache: dict[str, tuple[str, ...]] = {}
        self._ancestor_cache: dict[str, tuple[tuple[str, int], ...]] = {}

    def bit(self, object_id: str) -> int:
        cached = self._bit_cache.get(object_id)
        if cached is None:
            cached = 1 << self.index[object_id]
            self._bit_cache[object_id] = cached
        return cached

    def closure_bits(self, object_id: str) -> int:
        cached = self._closure_cache.get(object_id)
        if cached is not None:
            return cached
        bits = 0
        for child_id in self.inventory.usable_closure(object_id):
            bits |= self.bit(child_id)
        self._closure_cache[object_id] = bits
        return bits

    def closure_size(self, object_id: str) -> int:
        return self.closure_bits(object_id).bit_count()

    def usable_children(self, object_id: str) -> tuple[str, ...]:
        cached = self._children_cache.get(object_id)
        if cached is not None:
            return cached
        children = tuple(
            sorted(
                child_id
                for child_id in self.inventory.children.get(object_id, ())
                if child_id in self.inventory.objects and self.closure_size(child_id) >= 1
            )
        )
        self._children_cache[object_id] = children
        return children

    def usable_parents(self, object_id: str) -> tuple[str, ...]:
        cached = self._parents_cache.get(object_id)
        if cached is not None:
            return cached
        parents = tuple(sorted(parent_id for parent_id in self.inventory.parents.get(object_id, ()) if parent_id in self.inventory.objects))
        self._parents_cache[object_id] = parents
        return parents

    def sibling_bits(self, focus_id: str) -> int:
        cached = self._sibling_cache.get(focus_id)
        if cached is not None:
            return cached
        focus_space = self.closure_bits(focus_id) | self.bit(focus_id)
        bits = 0
        for parent_id in self.usable_parents(focus_id):
            for sibling_id in self.usable_children(parent_id):
                if sibling_id == focus_id:
                    continue
                bits |= self.bit(sibling_id)
                bits |= self.closure_bits(sibling_id)
        bits &= ~focus_space
        self._sibling_cache[focus_id] = bits
        return bits

    def ancestor_walk(self, seed_id: str) -> tuple[tuple[str, int], ...]:
        cached = self._ancestor_cache.get(seed_id)
        if cached is not None:
            return cached
        seen = {seed_id}
        queue: deque[tuple[str, int]] = deque()
        for parent_id in sorted(self.usable_parents(seed_id), key=lambda item: (self.closure_size(item), item)):
            queue.append((parent_id, 1))
        out: list[tuple[str, int]] = []
        while queue:
            current_id, distance = queue.popleft()
            if current_id in seen:
                continue
            seen.add(current_id)
            out.append((current_id, distance))
            parents = sorted(self.usable_parents(current_id), key=lambda item: (distance + 1, self.closure_size(item), item))
            for parent_id in parents:
                if parent_id not in seen:
                    queue.append((parent_id, distance + 1))
        cached = tuple(out)
        self._ancestor_cache[seed_id] = cached
        return cached

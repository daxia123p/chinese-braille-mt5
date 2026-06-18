from __future__ import annotations

from typing import List, Sequence

from .structured_braille_decoder import BrailleCellCandidate


class CandidateExpander:
    def __init__(
        self,
        confidence_threshold: float = 0.6,
        added_dot_confidence: float = 0.45,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.added_dot_confidence = added_dot_confidence

    def expand(self, cells: Sequence[BrailleCellCandidate]) -> List[List[BrailleCellCandidate]]:
        expanded: List[List[BrailleCellCandidate]] = []
        for cell in cells:
            expanded.append(self._expand_cell(cell))
        return expanded

    def _expand_cell(self, cell: BrailleCellCandidate) -> List[BrailleCellCandidate]:
        candidates = [cell]
        if cell.confidence >= self.confidence_threshold:
            return candidates

        removal_candidate = self._remove_low_confidence_dot(cell)
        addition_candidate = self._add_probable_missing_dot(cell)

        for option in (removal_candidate, addition_candidate):
            if option is not None and not self._is_duplicate(option, candidates):
                candidates.append(option)
            if len(candidates) >= 3:
                break
        return candidates

    @staticmethod
    def _is_duplicate(candidate: BrailleCellCandidate, existing: List[BrailleCellCandidate]) -> bool:
        return any(other.bits == candidate.bits for other in existing)

    def _remove_low_confidence_dot(self, cell: BrailleCellCandidate) -> BrailleCellCandidate | None:
        if not cell.dot_confidences:
            return None
        weakest_bit, weakest_conf = min(cell.dot_confidences.items(), key=lambda item: item[1])
        new_bits = cell.bits & ~(1 << weakest_bit)
        if new_bits == cell.bits:
            return None

        candidate = cell.clone(bits=new_bits, confidence=max(cell.confidence, weakest_conf))
        candidate.dot_confidences.pop(weakest_bit, None)
        return candidate

    def _add_probable_missing_dot(self, cell: BrailleCellCandidate) -> BrailleCellCandidate | None:
        missing_bits = [bit for bit in range(6) if not (cell.bits & (1 << bit))]
        if not missing_bits:
            return None

        best_bit = max(missing_bits, key=lambda bit: self._neighbor_support(cell, bit))
        new_bits = cell.bits | (1 << best_bit)
        candidate = cell.clone(bits=new_bits, confidence=min(0.99, cell.confidence + 0.1))
        candidate.dot_confidences[best_bit] = max(self.added_dot_confidence, cell.confidence * 0.5)
        return candidate

    @staticmethod
    def _neighbor_support(cell: BrailleCellCandidate, bit: int) -> float:
        neighbors = _NEIGHBOR_MAP[bit]
        support = sum(cell.dot_confidences.get(idx, 0.0) for idx in neighbors)
        return support / max(1, len(neighbors))


_NEIGHBOR_MAP = {
    0: (1, 2, 3),
    1: (0, 2, 4),
    2: (0, 1, 5),
    3: (0, 4, 5),
    4: (1, 3, 5),
    5: (2, 3, 4),
} 

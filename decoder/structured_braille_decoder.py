from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Tuple


@dataclass(frozen=True)
class DotCandidate:
    x: float
    y: float
    confidence: float


@dataclass
class BrailleCellCandidate:
    bits: int
    confidence: float
    row: int
    column: int
    dot_confidences: Dict[int, float] = field(default_factory=dict)

    def clone(self, *, bits: int | None = None, confidence: float | None = None) -> "BrailleCellCandidate":
        return BrailleCellCandidate(
            bits=self.bits if bits is None else bits,
            confidence=self.confidence if confidence is None else confidence,
            row=self.row,
            column=self.column,
            dot_confidences=dict(self.dot_confidences),
        )

    def to_unicode(self) -> str:
        return chr(0x2800 + (self.bits & 0b111111))


class StructuredBrailleDecoder:
    DOT_TO_BIT = {
        (0, 0): 0,
        (1, 0): 1,
        (2, 0): 2,
        (0, 1): 3,
        (1, 1): 4,
        (2, 1): 5,
    }

    def __init__(
        self,
        min_dot_confidence: float = 0.05,
        spacing_safety: float = 1e-3,
    ) -> None:
        self.min_dot_confidence = min_dot_confidence
        self.spacing_safety = spacing_safety

    def decode(self, candidates: Sequence[DotCandidate]) -> List[BrailleCellCandidate]:
        usable = [dot for dot in candidates if dot.confidence >= self.min_dot_confidence]
        if not usable:
            return []

        row_spacing = self._estimate_spacing([dot.y for dot in usable])
        col_spacing = self._estimate_spacing([dot.x for dot in usable])
        if row_spacing <= self.spacing_safety or col_spacing <= self.spacing_safety:
            return []

        min_y = min(dot.y for dot in usable)
        min_x = min(dot.x for dot in usable)

        cell_map: Dict[Tuple[int, int], _CellAccumulator] = {}
        for dot in usable:
            row_idx = self._quantize(dot.y - min_y, row_spacing)
            col_idx = self._quantize(dot.x - min_x, col_spacing)
            if row_idx < 0 or col_idx < 0:
                continue

            dot_row = row_idx % 3
            dot_col = col_idx % 2
            if (dot_row, dot_col) not in self.DOT_TO_BIT:
                continue

            line_idx = row_idx // 3
            cell_idx = col_idx // 2
            bit_index = self.DOT_TO_BIT[(dot_row, dot_col)]
            key = (line_idx, cell_idx)
            accumulator = cell_map.setdefault(key, _CellAccumulator(line_idx, cell_idx))
            accumulator.add_dot(bit_index, dot.confidence)

        cells = [acc.finalize() for acc in cell_map.values()]
        cells.sort(key=lambda cell: (cell.row, cell.column))
        return [cell for cell in cells if self._is_valid_cell(cell)]

    def _estimate_spacing(self, values: Iterable[float]) -> float:
        sorted_values = sorted(values)
        if len(sorted_values) < 2:
            return 1.0

        diffs = [b - a for a, b in zip(sorted_values[:-1], sorted_values[1:]) if b > a]
        if not diffs:
            return 1.0

        diffs.sort()
        window = max(1, len(diffs) // 3)
        trimmed = diffs[:window]
        median = trimmed[len(trimmed) // 2]
        return max(median, self.spacing_safety)

    @staticmethod
    def _quantize(value: float, spacing: float) -> int:
        if spacing <= 0:
            return -1
        return int(round(value / spacing))

    @staticmethod
    def _is_valid_cell(cell: BrailleCellCandidate) -> bool:
        return 0 <= cell.bits < 64


class _CellAccumulator:
    def __init__(self, row: int, column: int) -> None:
        self.row = row
        self.column = column
        self.dot_confidences: Dict[int, float] = {}

    def add_dot(self, bit_index: int, confidence: float) -> None:
        best_confidence = max(confidence, self.dot_confidences.get(bit_index, 0.0))
        self.dot_confidences[bit_index] = best_confidence

    def finalize(self) -> BrailleCellCandidate:
        bits = 0
        for bit_index in self.dot_confidences:
            bits |= 1 << bit_index
        if self.dot_confidences:
            avg_conf = sum(self.dot_confidences.values()) / len(self.dot_confidences)
        else:
            avg_conf = 0.0
        return BrailleCellCandidate(
            bits=bits,
            confidence=avg_conf,
            row=self.row,
            column=self.column,
            dot_confidences=dict(self.dot_confidences),
        ) 

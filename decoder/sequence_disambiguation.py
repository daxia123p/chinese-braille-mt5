from __future__ import annotations

from collections import Counter
from typing import Iterable, List, Sequence, Tuple

from .structured_braille_decoder import BrailleCellCandidate


class SequenceDisambiguator:
    def __init__(
        self,
        invalid_bits: Iterable[int] | None = None,
        repetition_penalty: float = 0.05,
        frequency_weight: float = 0.1,
    ) -> None:
        self.invalid_bits = set(invalid_bits or [])
        self.repetition_penalty = repetition_penalty
        self.frequency_weight = frequency_weight
        self.bit_counts: Counter[int] = Counter()

    def select_sequence(
        self, candidate_sequences: Sequence[Sequence[BrailleCellCandidate]]
    ) -> List[BrailleCellCandidate]:
        selections: List[BrailleCellCandidate] = []
        prev_char = None

        for options in candidate_sequences:
            if not options:
                continue
            best_candidate = max(options, key=lambda cand: self._score_candidate(cand, prev_char))
            selections.append(best_candidate)
            prev_char = best_candidate.to_unicode()
            self.bit_counts[best_candidate.bits] += 1
        return selections

    def resolve_documents(
        self, documents: Sequence[Sequence[Sequence[BrailleCellCandidate]]]
    ) -> List[List[BrailleCellCandidate]]:
        return [self.select_sequence(seq) for seq in documents]

    def _score_candidate(self, candidate: BrailleCellCandidate, prev_char: str | None) -> float:
        score = candidate.confidence
        if candidate.bits in self.invalid_bits:
            score -= 0.4
        if prev_char is not None and candidate.to_unicode() == prev_char:
            score -= self.repetition_penalty

        total = sum(self.bit_counts.values()) + 1
        freq_bonus = self.bit_counts.get(candidate.bits, 0) / total
        score += self.frequency_weight * freq_bonus
        return score


def braille_cells_to_string(cells: Sequence[BrailleCellCandidate]) -> str:
    return "".join(cell.to_unicode() for cell in cells)


def compute_char_accuracy(predictions: Sequence[str], references: Sequence[str]) -> float:
    assert len(predictions) == len(references), "Prediction and reference lengths must match."
    total_chars = 0
    correct_chars = 0
    for pred, ref in zip(predictions, references):
        matches = sum(1 for p, r in zip(pred, ref) if p == r)
        total_chars += max(len(ref), 1)
        correct_chars += matches
    return correct_chars / total_chars if total_chars else 0.0


def compute_cer(prediction: str, reference: str) -> float:
    distance = _levenshtein(prediction, reference)
    return distance / max(len(reference), 1)


def compute_wer(prediction: str, reference: str) -> float:
    pred_tokens = prediction.split()
    ref_tokens = reference.split()
    distance = _levenshtein(pred_tokens, ref_tokens)
    return distance / max(len(ref_tokens), 1)


def compute_document_metrics(
    predictions: Sequence[str], references: Sequence[str]
) -> Tuple[float, float, float]:
    char_acc = compute_char_accuracy(predictions, references)
    cer_values = [compute_cer(pred, ref) for pred, ref in zip(predictions, references)]
    wer_values = [compute_wer(pred, ref) for pred, ref in zip(predictions, references)]
    avg_cer = sum(cer_values) / len(cer_values) if cer_values else 0.0
    avg_wer = sum(wer_values) / len(wer_values) if wer_values else 0.0
    return char_acc, avg_cer, avg_wer


def five_fold_split(items: Sequence) -> List[Tuple[List, List]]:
    folds: List[Tuple[List, List]] = []
    total = len(items)
    fold_size = max(1, total // 5)
    for i in range(5):
        start = i * fold_size
        end = (i + 1) * fold_size if i < 4 else total
        test = list(items[start:end])
        train = list(items[:start]) + list(items[end:])
        folds.append((train, test))
    return folds


def _levenshtein(seq_a: Sequence, seq_b: Sequence) -> int:
    if not seq_a:
        return len(seq_b)
    if not seq_b:
        return len(seq_a)

    dp = list(range(len(seq_b) + 1))
    for i, token_a in enumerate(seq_a, start=1):
        prev_diag = dp[0]
        dp[0] = i
        for j, token_b in enumerate(seq_b, start=1):
            insert_cost = dp[j] + 1
            delete_cost = dp[j - 1] + 1
            replace_cost = prev_diag + (0 if token_a == token_b else 1)
            prev_diag = dp[j]
            dp[j] = min(insert_cost, delete_cost, replace_cost)
    return dp[-1] 

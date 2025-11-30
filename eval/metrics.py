"""
Evaluation metrics for structured data extraction quality assessment.

This module provides functions to compute precision, recall, and F1 scores
for comparing extracted structured data against ground truth, with support
for field-level analysis and fuzzy string matching for historical texts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union


@dataclass
class FieldMetrics:
    """Metrics for a single field across all entries."""
    
    field_name: str
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    
    @property
    def precision(self) -> float:
        """Precision = TP / (TP + FP)"""
        if self.true_positives + self.false_positives == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_positives)
    
    @property
    def recall(self) -> float:
        """Recall = TP / (TP + FN)"""
        if self.true_positives + self.false_negatives == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_negatives)
    
    @property
    def f1(self) -> float:
        """F1 = 2 * (precision * recall) / (precision + recall)"""
        p, r = self.precision, self.recall
        if p + r == 0:
            return 0.0
        return 2 * (p * r) / (p + r)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "field_name": self.field_name,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "precision_percent": round(self.precision * 100, 2),
            "recall_percent": round(self.recall * 100, 2),
            "f1_percent": round(self.f1 * 100, 2),
        }


@dataclass
class EntryMetrics:
    """Metrics for a single entry (record) comparison."""
    
    entry_index: int
    matched: bool
    field_scores: Dict[str, float] = field(default_factory=dict)
    overall_score: float = 0.0
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "entry_index": self.entry_index,
            "matched": self.matched,
            "field_scores": self.field_scores,
            "overall_score": round(self.overall_score, 4),
        }


@dataclass
class ExtractionMetrics:
    """Container for complete extraction evaluation metrics."""
    
    # Entry-level metrics
    total_gt_entries: int = 0
    total_hyp_entries: int = 0
    matched_entries: int = 0
    
    # Field-level metrics
    field_metrics: Dict[str, FieldMetrics] = field(default_factory=dict)
    
    # Per-entry details
    entry_details: List[EntryMetrics] = field(default_factory=list)
    
    @property
    def entry_precision(self) -> float:
        """Entry-level precision."""
        if self.total_hyp_entries == 0:
            return 0.0
        return self.matched_entries / self.total_hyp_entries
    
    @property
    def entry_recall(self) -> float:
        """Entry-level recall."""
        if self.total_gt_entries == 0:
            return 0.0
        return self.matched_entries / self.total_gt_entries
    
    @property
    def entry_f1(self) -> float:
        """Entry-level F1."""
        p, r = self.entry_precision, self.entry_recall
        if p + r == 0:
            return 0.0
        return 2 * (p * r) / (p + r)
    
    @property
    def macro_precision(self) -> float:
        """Macro-averaged precision across fields."""
        if not self.field_metrics:
            return 0.0
        return sum(m.precision for m in self.field_metrics.values()) / len(self.field_metrics)
    
    @property
    def macro_recall(self) -> float:
        """Macro-averaged recall across fields."""
        if not self.field_metrics:
            return 0.0
        return sum(m.recall for m in self.field_metrics.values()) / len(self.field_metrics)
    
    @property
    def macro_f1(self) -> float:
        """Macro-averaged F1 across fields."""
        if not self.field_metrics:
            return 0.0
        return sum(m.f1 for m in self.field_metrics.values()) / len(self.field_metrics)
    
    @property
    def micro_precision(self) -> float:
        """Micro-averaged precision (sum all TP/FP across fields)."""
        total_tp = sum(m.true_positives for m in self.field_metrics.values())
        total_fp = sum(m.false_positives for m in self.field_metrics.values())
        if total_tp + total_fp == 0:
            return 0.0
        return total_tp / (total_tp + total_fp)
    
    @property
    def micro_recall(self) -> float:
        """Micro-averaged recall (sum all TP/FN across fields)."""
        total_tp = sum(m.true_positives for m in self.field_metrics.values())
        total_fn = sum(m.false_negatives for m in self.field_metrics.values())
        if total_tp + total_fn == 0:
            return 0.0
        return total_tp / (total_tp + total_fn)
    
    @property
    def micro_f1(self) -> float:
        """Micro-averaged F1."""
        p, r = self.micro_precision, self.micro_recall
        if p + r == 0:
            return 0.0
        return 2 * (p * r) / (p + r)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "entry_level": {
                "total_ground_truth": self.total_gt_entries,
                "total_hypothesis": self.total_hyp_entries,
                "matched": self.matched_entries,
                "precision": round(self.entry_precision, 4),
                "recall": round(self.entry_recall, 4),
                "f1": round(self.entry_f1, 4),
                "precision_percent": round(self.entry_precision * 100, 2),
                "recall_percent": round(self.entry_recall * 100, 2),
                "f1_percent": round(self.entry_f1 * 100, 2),
            },
            "field_level": {
                "macro_precision": round(self.macro_precision, 4),
                "macro_recall": round(self.macro_recall, 4),
                "macro_f1": round(self.macro_f1, 4),
                "micro_precision": round(self.micro_precision, 4),
                "micro_recall": round(self.micro_recall, 4),
                "micro_f1": round(self.micro_f1, 4),
                "per_field": {
                    name: metrics.to_dict() 
                    for name, metrics in self.field_metrics.items()
                },
            },
            "entry_details": [e.to_dict() for e in self.entry_details],
        }


def normalize_string(text: str, lowercase: bool = True, normalize_ws: bool = True) -> str:
    """
    Normalize a string for comparison.
    
    Args:
        text: Input text
        lowercase: Whether to convert to lowercase
        normalize_ws: Whether to normalize whitespace
        
    Returns:
        Normalized string
    """
    if text is None:
        return ""
    
    text = str(text)
    
    if normalize_ws:
        text = re.sub(r'\s+', ' ', text.strip())
    
    if lowercase:
        text = text.lower()
    
    return text


def levenshtein_ratio(s1: str, s2: str) -> float:
    """
    Compute the Levenshtein similarity ratio between two strings.
    
    Returns a value between 0.0 (completely different) and 1.0 (identical).
    
    Args:
        s1: First string
        s2: Second string
        
    Returns:
        Similarity ratio
    """
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    
    # Use dynamic programming for Levenshtein distance
    len1, len2 = len(s1), len(s2)
    
    if len1 < len2:
        s1, s2 = s2, s1
        len1, len2 = len2, len1
    
    previous_row = list(range(len2 + 1))
    
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    distance = previous_row[-1]
    max_len = max(len1, len2)
    
    return 1.0 - (distance / max_len)


def compare_values(
    gt_value: Any,
    hyp_value: Any,
    threshold: float = 0.85,
    case_sensitive: bool = False,
    normalize_ws: bool = True,
) -> Tuple[bool, float]:
    """
    Compare two values with fuzzy matching support.
    
    Args:
        gt_value: Ground truth value
        hyp_value: Hypothesis value
        threshold: Similarity threshold for fuzzy matching
        case_sensitive: Whether comparison is case-sensitive
        normalize_ws: Whether to normalize whitespace
        
    Returns:
        Tuple of (match_bool, similarity_score)
    """
    # Handle None/null values
    if gt_value is None and hyp_value is None:
        return True, 1.0
    if gt_value is None or hyp_value is None:
        return False, 0.0
    
    # Handle numeric values
    if isinstance(gt_value, (int, float)) and isinstance(hyp_value, (int, float)):
        if gt_value == hyp_value:
            return True, 1.0
        return False, 0.0
    
    # Handle boolean values
    if isinstance(gt_value, bool) and isinstance(hyp_value, bool):
        if gt_value == hyp_value:
            return True, 1.0
        return False, 0.0
    
    # Handle lists (check if all elements match)
    if isinstance(gt_value, list) and isinstance(hyp_value, list):
        return compare_lists(gt_value, hyp_value, threshold, case_sensitive, normalize_ws)
    
    # Handle dicts (nested objects)
    if isinstance(gt_value, dict) and isinstance(hyp_value, dict):
        return compare_dicts(gt_value, hyp_value, threshold, case_sensitive, normalize_ws)
    
    # String comparison with fuzzy matching
    gt_str = normalize_string(str(gt_value), not case_sensitive, normalize_ws)
    hyp_str = normalize_string(str(hyp_value), not case_sensitive, normalize_ws)
    
    if gt_str == hyp_str:
        return True, 1.0
    
    ratio = levenshtein_ratio(gt_str, hyp_str)
    return ratio >= threshold, ratio


def compare_lists(
    gt_list: List[Any],
    hyp_list: List[Any],
    threshold: float = 0.85,
    case_sensitive: bool = False,
    normalize_ws: bool = True,
) -> Tuple[bool, float]:
    """
    Compare two lists, finding best matches for each element.
    
    Returns:
        Tuple of (all_matched, average_score)
    """
    if not gt_list and not hyp_list:
        return True, 1.0
    if not gt_list or not hyp_list:
        return False, 0.0
    
    # Try to match each ground truth element with a hypothesis element
    used_hyp = set()
    total_score = 0.0
    matches = 0
    
    for gt_item in gt_list:
        best_score = 0.0
        best_idx = -1
        
        for i, hyp_item in enumerate(hyp_list):
            if i in used_hyp:
                continue
            _, score = compare_values(gt_item, hyp_item, threshold, case_sensitive, normalize_ws)
            if score > best_score:
                best_score = score
                best_idx = i
        
        if best_idx >= 0 and best_score >= threshold:
            used_hyp.add(best_idx)
            matches += 1
        
        total_score += best_score
    
    avg_score = total_score / len(gt_list) if gt_list else 0.0
    all_matched = matches == len(gt_list) and len(gt_list) == len(hyp_list)
    
    return all_matched, avg_score


def compare_dicts(
    gt_dict: Dict[str, Any],
    hyp_dict: Dict[str, Any],
    threshold: float = 0.85,
    case_sensitive: bool = False,
    normalize_ws: bool = True,
) -> Tuple[bool, float]:
    """
    Compare two dictionaries field by field.
    
    Returns:
        Tuple of (all_matched, average_score)
    """
    if not gt_dict and not hyp_dict:
        return True, 1.0
    
    all_keys = set(gt_dict.keys()) | set(hyp_dict.keys())
    if not all_keys:
        return True, 1.0
    
    total_score = 0.0
    all_matched = True
    
    for key in all_keys:
        gt_val = gt_dict.get(key)
        hyp_val = hyp_dict.get(key)
        
        matched, score = compare_values(gt_val, hyp_val, threshold, case_sensitive, normalize_ws)
        total_score += score
        
        if not matched:
            all_matched = False
    
    avg_score = total_score / len(all_keys)
    return all_matched, avg_score


def get_nested_value(obj: Dict[str, Any], path: str) -> Any:
    """
    Get a value from a nested dictionary using dot notation.
    
    Args:
        obj: Dictionary to search
        path: Dot-separated path (e.g., "edition_info.year")
        
    Returns:
        Value at path, or None if not found
    """
    keys = path.split(".")
    current = obj
    
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        elif isinstance(current, list) and key.isdigit():
            idx = int(key)
            if 0 <= idx < len(current):
                current = current[idx]
            else:
                return None
        else:
            return None
    
    return current


def match_entries(
    gt_entries: List[Dict[str, Any]],
    hyp_entries: List[Dict[str, Any]],
    key_fields: List[str],
    threshold: float = 0.85,
    case_sensitive: bool = False,
    normalize_ws: bool = True,
) -> List[Tuple[int, int, float]]:
    """
    Match hypothesis entries to ground truth entries using key fields.
    
    Returns list of (gt_index, hyp_index, match_score) tuples.
    """
    matches = []
    used_hyp = set()
    
    for gt_idx, gt_entry in enumerate(gt_entries):
        best_score = 0.0
        best_hyp_idx = -1
        
        for hyp_idx, hyp_entry in enumerate(hyp_entries):
            if hyp_idx in used_hyp:
                continue
            
            # Compute similarity based on key fields
            field_scores = []
            for field_path in key_fields:
                gt_val = get_nested_value(gt_entry, field_path)
                hyp_val = get_nested_value(hyp_entry, field_path)
                _, score = compare_values(gt_val, hyp_val, threshold, case_sensitive, normalize_ws)
                field_scores.append(score)
            
            avg_score = sum(field_scores) / len(field_scores) if field_scores else 0.0
            
            if avg_score > best_score:
                best_score = avg_score
                best_hyp_idx = hyp_idx
        
        if best_hyp_idx >= 0 and best_score >= threshold:
            matches.append((gt_idx, best_hyp_idx, best_score))
            used_hyp.add(best_hyp_idx)
    
    return matches


def compute_extraction_metrics(
    ground_truth: Dict[str, Any],
    hypothesis: Dict[str, Any],
    entries_key: str = "entries",
    fields_to_evaluate: Optional[List[str]] = None,
    key_fields: Optional[List[str]] = None,
    threshold: float = 0.85,
    case_sensitive: bool = False,
    normalize_ws: bool = True,
) -> ExtractionMetrics:
    """
    Compute comprehensive extraction metrics.
    
    Args:
        ground_truth: Ground truth JSON data
        hypothesis: Hypothesis (model output) JSON data
        entries_key: Key for the array of entries (e.g., "entries")
        fields_to_evaluate: List of field paths to evaluate (None = all fields)
        key_fields: Fields used for matching entries (defaults to first 2 of fields_to_evaluate)
        threshold: Similarity threshold for fuzzy matching
        case_sensitive: Whether comparison is case-sensitive
        normalize_ws: Whether to normalize whitespace
        
    Returns:
        ExtractionMetrics with all computed values
    """
    metrics = ExtractionMetrics()
    
    # Get entries
    gt_entries = ground_truth.get(entries_key, [])
    hyp_entries = hypothesis.get(entries_key, [])
    
    if not isinstance(gt_entries, list):
        gt_entries = [gt_entries] if gt_entries else []
    if not isinstance(hyp_entries, list):
        hyp_entries = [hyp_entries] if hyp_entries else []
    
    metrics.total_gt_entries = len(gt_entries)
    metrics.total_hyp_entries = len(hyp_entries)
    
    # Determine fields to evaluate
    if not fields_to_evaluate and gt_entries:
        # Auto-detect fields from first ground truth entry
        fields_to_evaluate = list(gt_entries[0].keys())
    
    if not fields_to_evaluate:
        return metrics
    
    # Initialize field metrics
    for field_path in fields_to_evaluate:
        metrics.field_metrics[field_path] = FieldMetrics(field_name=field_path)
    
    # Determine key fields for matching
    if key_fields is None:
        key_fields = fields_to_evaluate[:2] if len(fields_to_evaluate) >= 2 else fields_to_evaluate
    
    # Match entries
    entry_matches = match_entries(
        gt_entries, hyp_entries, key_fields, threshold, case_sensitive, normalize_ws
    )
    
    matched_gt_indices = {gt_idx for gt_idx, _, _ in entry_matches}
    matched_hyp_indices = {hyp_idx for _, hyp_idx, _ in entry_matches}
    
    metrics.matched_entries = len(entry_matches)
    
    # Evaluate matched entries field by field
    for gt_idx, hyp_idx, match_score in entry_matches:
        gt_entry = gt_entries[gt_idx]
        hyp_entry = hyp_entries[hyp_idx]
        
        entry_metric = EntryMetrics(entry_index=gt_idx, matched=True, overall_score=match_score)
        
        for field_path in fields_to_evaluate:
            gt_val = get_nested_value(gt_entry, field_path)
            hyp_val = get_nested_value(hyp_entry, field_path)
            
            matched, score = compare_values(gt_val, hyp_val, threshold, case_sensitive, normalize_ws)
            entry_metric.field_scores[field_path] = score
            
            if matched:
                metrics.field_metrics[field_path].true_positives += 1
            else:
                if hyp_val is not None:
                    metrics.field_metrics[field_path].false_positives += 1
                if gt_val is not None:
                    metrics.field_metrics[field_path].false_negatives += 1
        
        metrics.entry_details.append(entry_metric)
    
    # Count false negatives for unmatched ground truth entries
    for gt_idx, gt_entry in enumerate(gt_entries):
        if gt_idx not in matched_gt_indices:
            entry_metric = EntryMetrics(entry_index=gt_idx, matched=False, overall_score=0.0)
            
            for field_path in fields_to_evaluate:
                gt_val = get_nested_value(gt_entry, field_path)
                entry_metric.field_scores[field_path] = 0.0
                
                if gt_val is not None:
                    metrics.field_metrics[field_path].false_negatives += 1
            
            metrics.entry_details.append(entry_metric)
    
    # Count false positives for unmatched hypothesis entries
    for hyp_idx, hyp_entry in enumerate(hyp_entries):
        if hyp_idx not in matched_hyp_indices:
            for field_path in fields_to_evaluate:
                hyp_val = get_nested_value(hyp_entry, field_path)
                
                if hyp_val is not None:
                    metrics.field_metrics[field_path].false_positives += 1
    
    return metrics


def aggregate_metrics(metrics_list: List[ExtractionMetrics]) -> ExtractionMetrics:
    """
    Aggregate metrics across multiple files/documents.
    
    Args:
        metrics_list: List of per-file metrics
        
    Returns:
        Aggregated ExtractionMetrics
    """
    if not metrics_list:
        return ExtractionMetrics()
    
    aggregated = ExtractionMetrics()
    
    # Aggregate entry counts
    aggregated.total_gt_entries = sum(m.total_gt_entries for m in metrics_list)
    aggregated.total_hyp_entries = sum(m.total_hyp_entries for m in metrics_list)
    aggregated.matched_entries = sum(m.matched_entries for m in metrics_list)
    
    # Aggregate field metrics
    all_field_names = set()
    for m in metrics_list:
        all_field_names.update(m.field_metrics.keys())
    
    for field_name in all_field_names:
        aggregated.field_metrics[field_name] = FieldMetrics(field_name=field_name)
        
        for m in metrics_list:
            if field_name in m.field_metrics:
                fm = m.field_metrics[field_name]
                aggregated.field_metrics[field_name].true_positives += fm.true_positives
                aggregated.field_metrics[field_name].false_positives += fm.false_positives
                aggregated.field_metrics[field_name].false_negatives += fm.false_negatives
    
    # Aggregate entry details
    offset = 0
    for m in metrics_list:
        for detail in m.entry_details:
            new_detail = EntryMetrics(
                entry_index=detail.entry_index + offset,
                matched=detail.matched,
                field_scores=detail.field_scores.copy(),
                overall_score=detail.overall_score,
            )
            aggregated.entry_details.append(new_detail)
        offset += m.total_gt_entries
    
    return aggregated


def format_metrics_table(
    model_metrics: Dict[str, Dict[str, ExtractionMetrics]],
    categories: Optional[List[str]] = None,
) -> str:
    """
    Format metrics as a Markdown table.
    
    Args:
        model_metrics: Dict mapping model_name -> category -> metrics
        categories: List of category names (if None, derived from data)
        
    Returns:
        Markdown-formatted table string
    """
    if not model_metrics:
        return "No metrics to display."
    
    # Collect all categories
    if categories is None:
        categories = set()
        for cat_metrics in model_metrics.values():
            categories.update(cat_metrics.keys())
        categories = sorted(categories)
    
    # Build header
    lines = [
        "| Model | Category | Entry P (%) | Entry R (%) | Entry F1 (%) | Micro P (%) | Micro R (%) | Micro F1 (%) |"
    ]
    lines.append("|-------|----------|-------------|-------------|--------------|-------------|-------------|--------------|")
    
    # Add rows
    for model_name in sorted(model_metrics.keys()):
        for category in categories:
            if category in model_metrics[model_name]:
                m = model_metrics[model_name][category]
                lines.append(
                    f"| {model_name} | {category} | "
                    f"{m.entry_precision*100:.2f} | {m.entry_recall*100:.2f} | {m.entry_f1*100:.2f} | "
                    f"{m.micro_precision*100:.2f} | {m.micro_recall*100:.2f} | {m.micro_f1*100:.2f} |"
                )
    
    return "\n".join(lines)


def format_field_metrics_table(metrics: ExtractionMetrics) -> str:
    """
    Format field-level metrics as a Markdown table.
    
    Args:
        metrics: ExtractionMetrics object
        
    Returns:
        Markdown-formatted table string
    """
    lines = ["| Field | Precision (%) | Recall (%) | F1 (%) | TP | FP | FN |"]
    lines.append("|-------|---------------|------------|--------|----|----|-----|")
    
    for field_name, fm in sorted(metrics.field_metrics.items()):
        lines.append(
            f"| {field_name} | {fm.precision*100:.2f} | {fm.recall*100:.2f} | {fm.f1*100:.2f} | "
            f"{fm.true_positives} | {fm.false_positives} | {fm.false_negatives} |"
        )
    
    return "\n".join(lines)


if __name__ == "__main__":
    # Quick test
    ground_truth = {
        "entries": [
            {"title": "The Art of French Cooking", "author": "Julia Child", "year": 1961},
            {"title": "Mastering the Art of French Cooking", "author": "Julia Child", "year": 1961},
        ]
    }
    
    hypothesis = {
        "entries": [
            {"title": "The Art of French Cooking", "author": "Julia Childs", "year": 1961},
            {"title": "Mastering Art of French Cooking", "author": "Julia Child", "year": 1962},
        ]
    }
    
    metrics = compute_extraction_metrics(
        ground_truth, hypothesis,
        fields_to_evaluate=["title", "author", "year"],
        key_fields=["title"],
    )
    
    print("Entry-level metrics:")
    print(f"  Precision: {metrics.entry_precision:.2%}")
    print(f"  Recall: {metrics.entry_recall:.2%}")
    print(f"  F1: {metrics.entry_f1:.2%}")
    print()
    print("Field-level metrics (micro-averaged):")
    print(f"  Precision: {metrics.micro_precision:.2%}")
    print(f"  Recall: {metrics.micro_recall:.2%}")
    print(f"  F1: {metrics.micro_f1:.2%}")
    print()
    print("Per-field breakdown:")
    print(format_field_metrics_table(metrics))

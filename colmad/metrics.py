"""Error-detection metrics with ``error`` as the positive class."""

from __future__ import annotations

from sklearn.metrics import matthews_corrcoef, precision_recall_fscore_support

METRICS = ["acc", "precision", "recall", "f1", "f2", "mcc"]


def score(y_true, y_pred):
    """Percentages rounded to two decimals, as reported in the paper."""
    t = [1 if y == "error" else 0 for y in y_true]
    p = [1 if y == "error" else 0 for y in y_pred]
    acc = sum(a == b for a, b in zip(y_true, y_pred)) / len(y_true)
    precision, recall, f1, _ = precision_recall_fscore_support(t, p, average="binary", zero_division=0)
    f2 = (5 * precision * recall) / (4 * precision + recall) if (4 * precision + recall) != 0 else 0
    mcc = matthews_corrcoef(t, p)
    vals = dict(acc=acc, precision=precision, recall=recall, f1=f1, f2=f2, mcc=mcc)
    out = {k: float(f"{v * 100:.2f}") for k, v in vals.items()}
    out["n"] = len(y_true)
    return out


def average(per_task, keys=("f1", "f2")):
    """Mean over tasks of the rounded per-task values (never pooled over samples)."""
    return {k: round(sum(m[k] for m in per_task) / len(per_task), 2) for k in keys}

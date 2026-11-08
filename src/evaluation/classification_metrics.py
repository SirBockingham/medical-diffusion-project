from collections.abc import Sequence

import numpy as np


def confusion_matrix(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    num_classes: int
) -> np.ndarray:
    """[K, K] matrix, rows are the true classes, columns the predicted ones."""
    true_labels = np.asarray(true_labels, dtype=np.int64)
    predicted_labels = np.asarray(predicted_labels, dtype=np.int64)
    if true_labels.shape != predicted_labels.shape:
        raise ValueError(
            f"Label arrays differ in length: true {true_labels.shape}, "
            f"predicted {predicted_labels.shape}."
        )

    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_label, predicted_label in zip(true_labels, predicted_labels):
        matrix[true_label, predicted_label] += 1
    return matrix



def per_class_recall(matrix: np.ndarray) -> np.ndarray:
    """NaN for classes that have no samples at all."""
    true_counts = matrix.sum(axis=1)
    recalls = np.full(matrix.shape[0], np.nan)
    for class_index in range(matrix.shape[0]):
        if true_counts[class_index] > 0:
            recalls[class_index] = matrix[class_index, class_index] / true_counts[class_index]
    return recalls



def per_class_precision(matrix: np.ndarray) -> np.ndarray:
    """0 for classes that were never predicted (same as zero_division=0 in scikit-learn)."""
    predicted_counts = matrix.sum(axis=0)
    precisions = np.zeros(matrix.shape[0])
    for class_index in range(matrix.shape[0]):
        if predicted_counts[class_index] > 0:
            precisions[class_index] = matrix[class_index, class_index] / predicted_counts[class_index]
    return precisions



def compute_classification_metrics(
    true_labels: np.ndarray, 
    predicted_labels: np.ndarray,
    num_classes: int
) -> dict:
    """Balanced accuracy and macro F1 only average over classes present in true_labels,
    a class missing from the test set has no meaningful recall."""
    matrix = confusion_matrix(true_labels, predicted_labels, num_classes)
    recalls = per_class_recall(matrix)
    precisions = per_class_precision(matrix)

    present_classes = []
    for class_index in range(num_classes):
        if matrix[class_index].sum() > 0:
            present_classes.append(class_index)
    if not present_classes:
        raise ValueError("No samples to evaluate.")

    f1_scores = []
    for class_index in present_classes:
        precision = precisions[class_index]
        recall = recalls[class_index]
        if precision + recall > 0:
            f1_scores.append(2 * precision * recall / (precision + recall))
        else:
            f1_scores.append(0.0)

    present_recalls = [recalls[class_index] for class_index in present_classes]

    recall_list = []
    for value in recalls:
        recall_list.append(None if np.isnan(value) else float(value))

    return {
        "accuracy": float(np.trace(matrix) / matrix.sum()),
        "balanced_accuracy": float(np.mean(present_recalls)),
        "macro_f1": float(np.mean(f1_scores)),
        "per_class_recall": recall_list,
        "confusion_matrix": matrix.tolist(),
        "num_samples": int(matrix.sum()),
    }



def mean_and_std(values: Sequence[float]) -> dict[str, float]:
    """Sample std (ddof=1) over the repeats, 0 for a single run."""
    array = np.asarray(values, dtype=np.float64)
    if array.size == 1:
        return {"mean": float(array[0]), "std": 0.0}
    return {"mean": float(array.mean()), "std": float(array.std(ddof=1))}



def mean_per_class(recall_lists: list[list[float | None]]) -> list[float | None]:
    """Per-class mean over the repeats, None where no repeat had a value."""
    averaged = []
    for class_index in range(len(recall_lists[0])):
        values = [recalls[class_index] for recalls in recall_lists if recalls[class_index] is not None]
        averaged.append(float(np.mean(values)) if values else None)
    return averaged
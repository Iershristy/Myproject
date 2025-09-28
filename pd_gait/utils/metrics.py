from typing import Tuple, Dict, Any
import numpy as np
from sklearn.metrics import (
	accuracy_score,
	precision_recall_fscore_support,
	classification_report,
)


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
	acc = accuracy_score(y_true, y_pred)
	# macro F1
	prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
	return acc, f1


def full_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels=None) -> Dict[str, Any]:
	acc = accuracy_score(y_true, y_pred)
	prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
		y_true, y_pred, average="macro", zero_division=0
	)
	prec_micro, rec_micro, f1_micro, _ = precision_recall_fscore_support(
		y_true, y_pred, average="micro", zero_division=0
	)
	prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
		y_true, y_pred, average="weighted", zero_division=0
	)
	per_class = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
	# per_class is (prec, rec, f1, support)
	return {
		"accuracy": float(acc),
		"precision_macro": float(prec_macro),
		"recall_macro": float(rec_macro),
		"f1_macro": float(f1_macro),
		"precision_micro": float(prec_micro),
		"recall_micro": float(rec_micro),
		"f1_micro": float(f1_micro),
		"precision_weighted": float(prec_weighted),
		"recall_weighted": float(rec_weighted),
		"f1_weighted": float(f1_weighted),
		"per_class": {
			"precision": per_class[0].tolist(),
			"recall": per_class[1].tolist(),
			"f1": per_class[2].tolist(),
			"support": per_class[3].tolist(),
		},
	}


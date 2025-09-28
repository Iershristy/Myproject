from typing import Tuple
import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
	acc = accuracy_score(y_true, y_pred)
	f1 = f1_score(y_true, y_pred, average="macro")
	return acc, f1


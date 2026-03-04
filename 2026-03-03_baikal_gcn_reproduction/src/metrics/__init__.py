import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    accuracy_score,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_percentage_error,
    mean_absolute_error,
    precision_recall_curve,
)

THRESHOLD = 0.5

def extract_angles(vector):
    x, y, z = vector
    theta = np.arcsin(z)
    phi = np.arctan2(y, x)
    
    # Convert from radians to degrees
    theta_deg = np.rad2deg(theta)
    phi_deg = np.rad2deg(phi)
    
    return theta_deg, phi_deg

def precision_at_recall(y_true, y_pred_prob, target_recall=0.9):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_prob)
    # Find the precision where recall is closest to target_recall (and >= target_recall)
    idx = np.where(recalls >= target_recall)[0]
    if len(idx) == 0:
        return 0.0
    return precisions[idx[-1]]

def binary_clf_metrics(y_pred_prob, y_true, threshold=THRESHOLD):
    y_pred = np.array(y_pred_prob > threshold, dtype=np.int32)
    y_true = np.array(y_true, dtype=np.int32)
    try:
        metrics = {
            "auc": roc_auc_score(y_true, y_pred_prob),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "precision@recall0.9": precision_at_recall(y_true, y_pred_prob, 0.9),
        }
        return metrics
    except ValueError:
        return {}


def regression_metrics(y_pred, y_true):
    y_pred = np.array(y_pred, dtype=np.float32)
    y_true = np.array(y_true, dtype=np.float32)
    try:
        return {
            "mae": mean_absolute_error(y_true, y_pred),
            "mse": mean_squared_error(y_true, y_pred),
            "mape": mean_absolute_percentage_error(y_true, y_pred),
        }
    except ValueError:
        return {}


def angle_reconstruction_metrics(y_pred, y_true):
    y_pred = np.array(y_pred, dtype=np.float32)
    y_true = np.array(y_true, dtype=np.float32)
    metrics = {}
    metrics.update(regression_metrics(y_pred, y_true))

    angles_true = np.array([extract_angles(vec) for vec in y_true], dtype=np.float32)
    angles_pred = np.array([extract_angles(vec) for vec in y_pred], dtype=np.float32)
    y_true_theta_angle, y_true_phi_angle = angles_true[:, 0], angles_true[:, 1]
    y_pred_theta_angle, y_pred_phi_angle = angles_pred[:, 0], angles_pred[:, 1]
    if not np.isnan(y_pred_theta_angle).any() and not np.isnan(y_pred_phi_angle).any():
        metrics["theta_mae"] = mean_absolute_error(
            y_true_theta_angle, y_pred_theta_angle
        )
        metrics["phi_mae"] = mean_absolute_error(y_true_phi_angle, y_pred_phi_angle)

    return metrics


def regression_and_clf_metrics(y_pred, y_true):
    metrics = {}
    metrics.update(
        binary_clf_metrics(y_pred[:, :, 2].reshape(-1), y_true[:, :, 2].reshape(-1))
    )
    metrics.update(
        regression_metrics(y_pred[:, 0, :2].reshape(-1), y_true[:, 0, :2].reshape(-1))
    )
    return metrics


def angle_and_track_cascade_metrics(y_pred, y_true, angles_pred, angles_true):
    metrics = {}
    metrics.update(
        binary_clf_metrics(y_pred.reshape(-1), y_true.reshape(-1))
    )
    metrics.update(
        angle_reconstruction_metrics(angles_pred, angles_true)
    )
    return metrics

def track_cascade_clf_metrics(y_pred, y_true, threshold=THRESHOLD):
    metrics = {}
    y_true = np.array(y_true, dtype=bool)
    metrics = {k + "[cascade=1]": v for k, v in binary_clf_metrics(y_pred, y_true, threshold).items()}
    metrics.update({k + "[track=1]": v for k, v in binary_clf_metrics(1 - y_pred, ~y_true, 1 - threshold).items()})
    metrics.update({"n_cascade/n_track": y_pred.sum() / (~y_true).sum()})
    return metrics


def tres_and_track_cascade_metrics(y_pred, y_true):
    metrics = {}
    metrics.update(
        track_cascade_clf_metrics(y_pred[:, 0].reshape(-1), y_true[:, 0].reshape(-1))
    )
    metrics.update(
        regression_metrics(y_pred[:, 1].reshape(-1), y_true[:, 1].reshape(-1))
    )
    # metrics.update(
    #     angle_reconstruction_metrics(y_pred[:, :2].reshape(-1), y_true[:, :2].reshape(-1))
    # )
    return metrics
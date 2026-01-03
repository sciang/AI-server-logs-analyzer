import numpy as np
import pandas as pd
import re
import joblib
import logging

from datetime import datetime
from typing import Dict, List
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LogAnomalyDetector:
    """
    Multi-model anomaly detection system for server logs
    Uses Isolation Forest + DBSCAN
    """

    def __init__(self):
        self.isolation_forest = None
        self.dbscan = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.trained = False

    # --------------------------------------------------
    # Feature Extraction
    # --------------------------------------------------
    def extract_features(self, logs: List[Dict]) -> pd.DataFrame:
        features = []

        for log in logs:
            f = {}

            # Timestamp
            timestamp = log.get("timestamp", datetime.utcnow())
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))

            f["hour"] = timestamp.hour
            f["day_of_week"] = timestamp.weekday()
            f["minute"] = timestamp.minute

            # Log level
            level_map = {
                "DEBUG": 0,
                "INFO": 1,
                "WARNING": 2,
                "ERROR": 3,
                "CRITICAL": 4,
            }
            f["level_code"] = level_map.get(log.get("level", "INFO"), 1)

            message = log.get("message", "")
            f["message_length"] = len(message)
            f["word_count"] = len(message.split())

            # Pattern flags
            f["has_error_keyword"] = int(bool(re.search(r"error|exception|fail", message.lower())))
            f["has_number"] = int(bool(re.search(r"\d", message)))
            f["has_ip"] = int(bool(re.search(r"\d{1,3}(\.\d{1,3}){3}", message)))

            # Optional metrics
            f["request_rate"] = log.get("request_rate", 0)
            meta = log.get("metadata", {})
            f["response_time"] = meta.get("response_time", 0)
            f["status_code"] = meta.get("status_code", 200)

            features.append(f)

        df = pd.DataFrame(features)
        self.feature_names = df.columns.tolist()
        return df

    # --------------------------------------------------
    # Training
    # --------------------------------------------------
    def train(self, logs: List[Dict], contamination: float = 0.1):
        logger.info(f"Training on {len(logs)} logs")

        X = self.extract_features(logs)
        X_scaled = self.scaler.fit_transform(X)

        self.isolation_forest = IsolationForest(
            n_estimators=200,
            contamination=contamination,
            random_state=42
        )
        self.isolation_forest.fit(X_scaled)

        self.dbscan = DBSCAN(eps=1.2, min_samples=5)
        self.dbscan.fit(X_scaled)

        self.trained = True
        logger.info("Training complete")

    # --------------------------------------------------
    # Detection
    # --------------------------------------------------
    def detect_anomalies(self, logs: List[Dict]) -> List[Dict]:
        if not self.trained:
            raise RuntimeError("Model not trained")

        X = self.extract_features(logs)
        X_scaled = self.scaler.transform(X)

        if_preds = self.isolation_forest.predict(X_scaled)
        if_scores = self.isolation_forest.score_samples(X_scaled)
        db_labels = self.dbscan.fit_predict(X_scaled)

        results = []

        for i, log in enumerate(logs):
            is_if_anomaly = if_preds[i] == -1
            is_db_anomaly = db_labels[i] == -1
            is_anomaly = is_if_anomaly or is_db_anomaly

            score = float(-if_scores[i])
            anomaly_type = None

            if is_anomaly:
                msg = log.get("message", "").lower()
                if "database" in msg or "connection" in msg:
                    anomaly_type = "database_issue"
                elif re.search(r"ddos|dos|flood", msg):
                    anomaly_type = "ddos_attack"
                elif log.get("request_rate", 0) > 100:
                    anomaly_type = "rate_limit_exceeded"
                elif "unauthorized" in msg or "forbidden" in msg:
                    anomaly_type = "security_breach"
                elif "memory" in msg or "cpu" in msg:
                    anomaly_type = "resource_exhaustion"
                else:
                    anomaly_type = "unknown_anomaly"

            results.append({
                "log_id": log.get("_id", str(i)),
                "is_anomaly": is_anomaly,
                "anomaly_type": anomaly_type,
                "anomaly_score": round(score, 4)
            })

        return results

    # --------------------------------------------------
    # Save / Load
    # --------------------------------------------------
    def save_model(self, path="models"):
        import os
        os.makedirs(path, exist_ok=True)

        joblib.dump(self.isolation_forest, f"{path}/isolation_forest.pkl")
        joblib.dump(self.dbscan, f"{path}/dbscan.pkl")
        joblib.dump(self.scaler, f"{path}/scaler.pkl")
        joblib.dump(self.feature_names, f"{path}/features.pkl")

        logger.info("Model saved")

    def load_model(self, path="models"):
        self.isolation_forest = joblib.load(f"{path}/isolation_forest.pkl")
        self.dbscan = joblib.load(f"{path}/dbscan.pkl")
        self.scaler = joblib.load(f"{path}/scaler.pkl")
        self.feature_names = joblib.load(f"{path}/features.pkl")
        self.trained = True
        logger.info("Model loaded")
# Example usage:
# detector = LogAnomalyDetector()   
sample_logs = [
    {
        "_id": "1",
        "timestamp": "2025-01-03T08:30:00Z",
        "level": "INFO",
        "message": "User login successful",
        "request_rate": 5,
        "metadata": {"response_time": 120, "status_code": 200}
    },
    {
        "_id": "2",
        "timestamp": "2025-01-03T08:31:00Z",
        "level": "ERROR",
        "message": "Database connection failed after timeout",
        "request_rate": 120,
        "metadata": {"response_time": 5000, "status_code": 500}
    }
]

detector = LogAnomalyDetector()
detector.train(sample_logs * 50)

results = detector.detect_anomalies(sample_logs)

for r in results:
    print(r)

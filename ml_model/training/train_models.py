import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from inference.anomaly_detector import LogAnomalyDetector, AutoencoderAnomalyDetector
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_synthetic_logs(num_samples: int = 10000) -> list:
    """
    Generate synthetic log data for training
    Includes normal patterns and anomalies
    """
    logs = []
    services = ['api', 'database', 'auth', 'frontend', 'cache']
    levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
    
    normal_messages = [
        'Request processed successfully',
        'User authenticated',
        'Database query executed',
        'Cache hit',
        'API response sent',
        'Session created',
        'Data validated',
        'Transaction completed'
    ]
    
    anomaly_messages = [
        'Database connection timeout error',
        'Memory allocation failed - Out of memory',
        'Potential SQL injection detected in query',
        'Unusual spike in traffic from IP 192.168.1.100',
        'Authentication failed 100 times from same IP',
        'DDoS attack pattern detected',
        'Critical: Server CPU at 99% for 10 minutes',
        'Database deadlock detected',
        'Unauthorized access attempt blocked',
        'SSL certificate validation failed'
    ]
    
    base_time = datetime.utcnow() - timedelta(days=7)
    
    for i in range(num_samples):
        is_anomaly = random.random() < 0.1  # 10% anomalies
        
        log = {
            '_id': str(i),
            'timestamp': base_time + timedelta(
                minutes=random.randint(0, 10080)  # 7 days
            ),
            'service': random.choice(services),
            'level': random.choice(levels) if not is_anomaly else random.choice(['ERROR', 'CRITICAL']),
            'message': random.choice(anomaly_messages if is_anomaly else normal_messages),
            'request_rate': random.randint(500, 1000) if is_anomaly else random.randint(1, 50),
            'metadata': {
                'response_time': random.randint(1000, 5000) if is_anomaly else random.randint(50, 500),
                'status_code': random.choice([500, 503, 429]) if is_anomaly else random.choice([200, 201, 304])
            }
        }
        
        logs.append(log)
    
    return logs

def main():
    """Main training pipeline"""
    logger.info("Starting ML model training pipeline...")
    
    # Generate training data
    logger.info("Generating synthetic training data...")
    training_logs = generate_synthetic_logs(10000)
    
    # Split into train and validation
    split_idx = int(len(training_logs) * 0.8)
    train_logs = training_logs[:split_idx]
    val_logs = training_logs[split_idx:]
    
    logger.info(f"Training samples: {len(train_logs)}")
    logger.info(f"Validation samples: {len(val_logs)}")
    
    # Train Isolation Forest + DBSCAN detector
    logger.info("\n=== Training Isolation Forest + DBSCAN ===")
    detector = LogAnomalyDetector()
    detector.train(train_logs, contamination=0.1)
    
    # Validate
    logger.info("\nValidating on test set...")
    val_results = detector.detect_anomalies(val_logs)
    
    anomalies_detected = sum(1 for r in val_results if r['is_anomaly'])
    logger.info(f"Anomalies detected: {anomalies_detected}/{len(val_results)} "
                f"({anomalies_detected/len(val_results)*100:.1f}%)")
    
    # Save model
    logger.info("\nSaving models...")
    detector.save_model('models/')
    
    # Train Autoencoder (if TensorFlow available)
    logger.info("\n=== Training Autoencoder ===")
    ae_detector = AutoencoderAnomalyDetector(encoding_dim=10)
    
    # Extract features for autoencoder
    train_features = detector.extract_features(train_logs)
    val_features = detector.extract_features(val_logs)
    
    success = ae_detector.train(train_features.values, epochs=50, batch_size=32)
    
    if success:
        # Validate autoencoder
        is_anomaly, scores = ae_detector.detect(val_features.values)
        ae_anomalies = np.sum(is_anomaly)
        logger.info(f"Autoencoder anomalies: {ae_anomalies}/{len(val_logs)} "
                    f"({ae_anomalies/len(val_logs)*100:.1f}%)")
    
    logger.info("\n=== Training Complete ===")
    logger.info("Models saved to ./models/ directory")
    
    # Print sample detections
    logger.info("\nSample anomaly detections:")
    for result in val_results[:10]:
        if result['is_anomaly']:
            logger.info(f"  - Type: {result['anomaly_type']}, "
                       f"Score: {result['anomaly_score']:.3f}, "
                       f"Preview: {result['details']['message_preview']}")

if __name__ == "__main__":
    main()
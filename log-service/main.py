from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import asyncio
import json
import jwt
import uvicorn
from datetime import datetime, timedelta
import redis.asyncio as redis
from motor.motor_asyncio import AsyncIOMotorClient
import numpy as np
import logging
from collections import deque
import re

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Log Analyzer Service")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()
JWT_SECRET = "your-secret-key-change-in-production"

# Database connections
mongodb_client = None
redis_client = None

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"Client connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"Client disconnected. Total connections: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting message: {e}")

manager = ConnectionManager()

# Models
class LogEntry(BaseModel):
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    level: str
    service: str
    message: str
    metadata: Optional[Dict] = {}
    ip_address: Optional[str] = None
    user_id: Optional[str] = None

class AnomalyResult(BaseModel):
    log_id: str
    is_anomaly: bool
    anomaly_score: float
    anomaly_type: Optional[str] = None
    detected_at: datetime = Field(default_factory=datetime.utcnow)
    details: Dict = {}

# JWT verification
def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        token = credentials.credentials
        payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

# Log pattern detection
class LogPatternAnalyzer:
    def __init__(self):
        self.error_patterns = [
            r'error|exception|fail|fatal|critical',
            r'database.*error|connection.*refused',
            r'timeout|timed out',
            r'out of memory|oom|memory leak',
            r'null pointer|segfault',
        ]
        self.security_patterns = [
            r'unauthorized|forbidden|access denied',
            r'sql injection|xss|csrf',
            r'brute force|ddos|dos attack',
            r'malware|virus|trojan',
        ]
        
    def analyze(self, log_message: str) -> Dict:
        log_lower = log_message.lower()
        
        result = {
            'has_error': False,
            'has_security_issue': False,
            'pattern_matches': []
        }
        
        for pattern in self.error_patterns:
            if re.search(pattern, log_lower):
                result['has_error'] = True
                result['pattern_matches'].append(f"error: {pattern}")
                
        for pattern in self.security_patterns:
            if re.search(pattern, log_lower):
                result['has_security_issue'] = True
                result['pattern_matches'].append(f"security: {pattern}")
                
        return result

pattern_analyzer = LogPatternAnalyzer()

# Rate limiting detector
class RateLimitDetector:
    def __init__(self, threshold: int = 100, window: int = 60):
        self.threshold = threshold
        self.window = window
        self.ip_requests = {}
        
    def check_rate_limit(self, ip_address: str) -> bool:
        current_time = datetime.utcnow()
        
        if ip_address not in self.ip_requests:
            self.ip_requests[ip_address] = deque()
            
        # Remove old requests outside the time window
        while self.ip_requests[ip_address] and \
              (current_time - self.ip_requests[ip_address][0]).seconds > self.window:
            self.ip_requests[ip_address].popleft()
            
        # Add current request
        self.ip_requests[ip_address].append(current_time)
        
        # Check if threshold exceeded
        return len(self.ip_requests[ip_address]) > self.threshold

rate_detector = RateLimitDetector()

# Startup and shutdown events
@app.on_event("startup")
async def startup_db_client():
    global mongodb_client, redis_client
    mongodb_client = AsyncIOMotorClient("mongodb://mongo:27017")
    redis_client = await redis.from_url("redis://redis:6379")
    logger.info("Connected to databases")

@app.on_event("shutdown")
async def shutdown_db_client():
    mongodb_client.close()
    await redis_client.close()
    logger.info("Closed database connections")

# Endpoints
@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "log-service"}

@app.post("/api/logs/ingest")
async def ingest_log(log: LogEntry, user=Depends(verify_token)):
    """Ingest a single log entry"""
    try:
        # Analyze log patterns
        pattern_analysis = pattern_analyzer.analyze(log.message)
        
        # Check for DDoS-like patterns
        is_rate_limited = False
        if log.ip_address:
            is_rate_limited = rate_detector.check_rate_limit(log.ip_address)
        
        # Prepare log document
        log_dict = log.dict()
        log_dict['pattern_analysis'] = pattern_analysis
        log_dict['is_rate_limited'] = is_rate_limited
        log_dict['processed_at'] = datetime.utcnow()
        
        # Store in MongoDB
        db = mongodb_client.loganalyzer
        result = await db.logs.insert_one(log_dict)
        log_dict['_id'] = str(result.inserted_id)
        
        # Cache in Redis for real-time access
        await redis_client.setex(
            f"log:{result.inserted_id}",
            3600,
            json.dumps(log_dict, default=str)
        )
        
        # Detect if this log needs immediate attention
        needs_attention = (
            pattern_analysis['has_error'] or 
            pattern_analysis['has_security_issue'] or 
            is_rate_limited
        )
        
        if needs_attention:
            # Broadcast to connected clients
            await manager.broadcast({
                "type": "alert",
                "log": log_dict,
                "timestamp": datetime.utcnow().isoformat()
            })
        
        return {
            "status": "success",
            "log_id": str(result.inserted_id),
            "needs_attention": needs_attention
        }
        
    except Exception as e:
        logger.error(f"Error ingesting log: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/logs")
async def get_logs(
    limit: int = 100,
    level: Optional[str] = None,
    service: Optional[str] = None,
    user=Depends(verify_token)
):
    """Retrieve logs with optional filtering"""
    try:
        db = mongodb_client.loganalyzer
        
        # Build query
        query = {}
        if level:
            query['level'] = level
        if service:
            query['service'] = service
            
        # Fetch logs
        cursor = db.logs.find(query).sort('timestamp', -1).limit(limit)
        logs = await cursor.to_list(length=limit)
        
        # Convert ObjectId to string
        for log in logs:
            log['_id'] = str(log['_id'])
            
        return {"logs": logs, "count": len(logs)}
        
    except Exception as e:
        logger.error(f"Error fetching logs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/logs/anomalies")
async def get_anomalies(
    limit: int = 50,
    user=Depends(verify_token)
):
    """Retrieve detected anomalies"""
    try:
        db = mongodb_client.loganalyzer
        
        # Fetch anomalies
        cursor = db.anomalies.find().sort('detected_at', -1).limit(limit)
        anomalies = await cursor.to_list(length=limit)
        
        for anomaly in anomalies:
            anomaly['_id'] = str(anomaly['_id'])
            
        return {"anomalies": anomalies, "count": len(anomalies)}
        
    except Exception as e:
        logger.error(f"Error fetching anomalies: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/logs/stats")
async def get_stats(user=Depends(verify_token)):
    """Get log statistics"""
    try:
        db = mongodb_client.loganalyzer
        
        # Count logs by level
        pipeline = [
            {
                "$group": {
                    "_id": "$level",
                    "count": {"$sum": 1}
                }
            }
        ]
        level_stats = await db.logs.aggregate(pipeline).to_list(length=None)
        
        # Count total logs
        total_logs = await db.logs.count_documents({})
        
        # Count anomalies
        total_anomalies = await db.anomalies.count_documents({})
        
        # Get recent activity (last hour)
        one_hour_ago = datetime.utcnow() - timedelta(hours=1)
        recent_logs = await db.logs.count_documents(
            {"timestamp": {"$gte": one_hour_ago}}
        )
        
        return {
            "total_logs": total_logs,
            "total_anomalies": total_anomalies,
            "recent_logs": recent_logs,
            "level_distribution": {stat['_id']: stat['count'] for stat in level_stats}
        }
        
    except Exception as e:
        logger.error(f"Error fetching stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/logs")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time log streaming"""
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive
            data = await websocket.receive_text()
            
            # Echo back to confirm connection
            if data == "ping":
                await websocket.send_text("pong")
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

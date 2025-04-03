#!/usr/bin/env python3
"""
Model API for AI Serving

This script demonstrates how to build a robust API for serving AI models,
with best practices for production-ready deployments.

Key functionalities:
- RESTful endpoint design
- Authentication and rate limiting
- Request validation and error handling
- Async processing and batching
- Monitoring and logging
"""

import os
import sys
import time
import json
import uuid
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
from functools import wraps
from collections import defaultdict, deque

# FastAPI for robust API creation
from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

# For async functionality
import uvicorn
import httpx

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set up paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output" / "model_api"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# ==========================================================
# Configuration
# ==========================================================

class APIConfig:
    """Configuration settings for the API"""
    # API settings
    API_TITLE = "AI Model API"
    API_DESCRIPTION = "A production-ready API for serving AI models"
    API_VERSION = "1.0.0"
    
    # Authentication
    API_KEY_NAME = "X-API-Key"
    API_KEYS = {
        "test-key": {"name": "Test User", "rate_limit": 100},
        "premium-key": {"name": "Premium User", "rate_limit": 1000}
    }
    
    # Rate limiting
    RATE_LIMIT_WINDOW = 3600  # 1 hour in seconds
    DEFAULT_RATE_LIMIT = 10  # requests per window for users without specific limits
    
    # Performance settings
    MAX_BATCH_SIZE = 16
    BATCH_TIMEOUT = 0.1  # seconds to wait before processing small batches
    
    # Model settings
    DEFAULT_MODEL = "text-generation"
    AVAILABLE_MODELS = ["text-generation", "sentiment-analysis", "summarization"]
    
    # Monitoring
    ENABLE_METRICS = True
    METRICS_LOG_INTERVAL = 60  # log metrics every 60 seconds

# ==========================================================
# Models and DTOs
# ==========================================================

class ModelInput(BaseModel):
    """Input data for model inference"""
    text: str = Field(..., min_length=1, max_length=10000, example="This is a sample text for prediction")
    parameters: Dict[str, Any] = Field(default_factory=dict, example={"max_length": 50})
    
    @validator('parameters')
    def validate_parameters(cls, v):
        """Validate model parameters"""
        allowed_params = ["max_length", "temperature", "top_p", "top_k", "num_beams"]
        for key in v.keys():
            if key not in allowed_params:
                raise ValueError(f"Parameter '{key}' is not supported. Allowed parameters: {allowed_params}")
        return v

class BatchModelInput(BaseModel):
    """Input data for batch inference"""
    inputs: List[ModelInput] = Field(..., min_items=1, max_items=APIConfig.MAX_BATCH_SIZE)

class ModelOutput(BaseModel):
    """Output data from model inference"""
    id: str = Field(..., example="pred_123456789")
    result: str = Field(..., example="This is a prediction result")
    model: str = Field(..., example="text-generation")
    processing_time: float = Field(..., example=0.1234)
    created_at: str = Field(..., example="2023-01-01T12:00:00Z")

class BatchModelOutput(BaseModel):
    """Output data from batch inference"""
    batch_id: str = Field(..., example="batch_123456789")
    results: List[ModelOutput] = Field(...)
    total_processing_time: float = Field(..., example=0.5678)

class ErrorResponse(BaseModel):
    """Standardized error response"""
    error: str = Field(..., example="Invalid input")
    detail: Optional[str] = Field(None, example="Text length exceeds maximum")
    code: int = Field(..., example=400)

class HealthStatus(BaseModel):
    """Health check response"""
    status: str = Field(..., example="healthy")
    version: str = Field(..., example="1.0.0")
    models: Dict[str, str] = Field(..., example={"text-generation": "ready"})
    uptime: float = Field(..., example=3600.5)

class MetricsResponse(BaseModel):
    """Metrics response"""
    requests_total: int = Field(..., example=1000)
    requests_by_endpoint: Dict[str, int] = Field(..., example={"/predict": 800, "/batch": 200})
    average_latency: float = Field(..., example=0.123)
    error_rate: float = Field(..., example=0.02)
    requests_by_model: Dict[str, int] = Field(..., example={"text-generation": 800})

# ==========================================================
# Authentication and Security
# ==========================================================

api_key_header = APIKeyHeader(name=APIConfig.API_KEY_NAME, auto_error=False)

async def get_api_key(
    api_key: str = Depends(api_key_header)
) -> Dict[str, Any]:
    """Validate API key and return user info"""
    if api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key",
            headers={"WWW-Authenticate": APIConfig.API_KEY_NAME},
        )
    
    if api_key not in APIConfig.API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": APIConfig.API_KEY_NAME},
        )
    
    return {"key": api_key, **APIConfig.API_KEYS[api_key]}

# ==========================================================
# Rate Limiting
# ==========================================================

class RateLimiter:
    """Simple in-memory rate limiter"""
    
    def __init__(self, window: int = APIConfig.RATE_LIMIT_WINDOW):
        self.window = window  # Time window in seconds
        self.requests = defaultdict(lambda: deque())
    
    def check_rate_limit(self, key: str, limit: int) -> Tuple[bool, int, int]:
        """
        Check if the request should be rate limited
        
        Returns:
            Tuple[bool, int, int]: (is_allowed, current_count, reset_in_seconds)
        """
        now = time.time()
        requests = self.requests[key]
        
        # Remove expired timestamps
        while requests and requests[0] < now - self.window:
            requests.popleft()
        
        # Check if under limit
        current_count = len(requests)
        is_allowed = current_count < limit
        
        # Calculate reset time
        reset_in = self.window if not requests else int(self.window - (now - requests[0]))
        
        # Add current request timestamp if allowed
        if is_allowed:
            requests.append(now)
        
        return is_allowed, current_count, reset_in

# Initialize rate limiter
rate_limiter = RateLimiter()

async def check_rate_limit(user_info: Dict[str, Any] = Depends(get_api_key)):
    """Dependency to check rate limit for a user"""
    rate_limit = user_info.get("rate_limit", APIConfig.DEFAULT_RATE_LIMIT)
    is_allowed, current, reset_in = rate_limiter.check_rate_limit(user_info["key"], rate_limit)
    
    if not is_allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Try again in {reset_in} seconds.",
            headers={"X-Rate-Limit-Reset": str(reset_in)}
        )
    
    return user_info

# ==========================================================
# Monitoring and Metrics
# ==========================================================

class APIMetrics:
    """Track API usage and performance metrics"""
    
    def __init__(self):
        self.start_time = time.time()
        self.requests_total = 0
        self.requests_by_endpoint = defaultdict(int)
        self.requests_by_model = defaultdict(int)
        self.processing_times = []
        self.errors = 0
    
    def track_request(self, endpoint: str, model: str, processing_time: float, is_error: bool = False):
        """Track a request for metrics"""
        self.requests_total += 1
        self.requests_by_endpoint[endpoint] += 1
        self.requests_by_model[model] += 1
        self.processing_times.append(processing_time)
        if is_error:
            self.errors += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        avg_latency = sum(self.processing_times) / max(len(self.processing_times), 1)
        error_rate = self.errors / max(self.requests_total, 1)
        
        return {
            "requests_total": self.requests_total,
            "requests_by_endpoint": dict(self.requests_by_endpoint),
            "requests_by_model": dict(self.requests_by_model),
            "average_latency": avg_latency,
            "error_rate": error_rate,
            "uptime": time.time() - self.start_time
        }

# Initialize metrics
metrics = APIMetrics()

# Middleware to track request timing
@asyncio.coroutine
async def metrics_middleware(request: Request, call_next):
    """Middleware to track request timing"""
    start_time = time.time()
    
    try:
        response = await call_next(request)
        is_error = response.status_code >= 400
    except Exception:
        is_error = True
        raise
    finally:
        process_time = time.time() - start_time
        endpoint = request.url.path
        model = request.path_params.get("model_name", APIConfig.DEFAULT_MODEL)
        metrics.track_request(endpoint, model, process_time, is_error)
    
    return response

# ==========================================================
# Model Implementations (Simulated)
# ==========================================================

async def simulate_model_processing(text: str, model_name: str, parameters: Dict[str, Any]) -> str:
    """Simulate model processing with realistic delays"""
    # Simulate different processing times based on model and parameters
    processing_time = 0.2
    
    # Adjust based on model
    if model_name == "sentiment-analysis":
        processing_time = 0.1
    elif model_name == "summarization":
        # Longer texts take more time to summarize
        processing_time = 0.3 + (len(text) / 5000)
    
    # Adjust based on parameters
    if "max_length" in parameters:
        processing_time += parameters["max_length"] / 1000
    
    # Add some random variation
    import random
    processing_time *= random.uniform(0.8, 1.2)
    
    # Simulate processing
    await asyncio.sleep(processing_time)
    
    # Generate a simulated response based on the model
    if model_name == "sentiment-analysis":
        sentiments = ["positive", "negative", "neutral"]
        return f"Sentiment: {random.choice(sentiments)}"
    elif model_name == "summarization":
        return f"Summary: {text[:50]}..."
    else:  # text-generation
        return f"Generated: {text[:30]}... followed by AI generated continuation."

class ModelManager:
    """Manager for AI models"""
    
    def __init__(self):
        self.models = {name: "ready" for name in APIConfig.AVAILABLE_MODELS}
        self.batch_queue = {}
        self.batch_events = {}
    
    async def predict(self, input_data: ModelInput, model_name: str) -> ModelOutput:
        """Run single prediction"""
        start_time = time.time()
        
        # Check if model exists
        if model_name not in self.models:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model '{model_name}' not found"
            )
        
        # Process the prediction
        result = await simulate_model_processing(input_data.text, model_name, input_data.parameters)
        
        processing_time = time.time() - start_time
        
        return ModelOutput(
            id=f"pred_{uuid.uuid4().hex[:10]}",
            result=result,
            model=model_name,
            processing_time=processing_time,
            created_at=datetime.utcnow().isoformat()
        )
    
    async def predict_batch(self, batch_input: BatchModelInput, model_name: str) -> BatchModelOutput:
        """Run batch prediction"""
        start_time = time.time()
        
        # Process each prediction
        tasks = []
        for input_item in batch_input.inputs:
            tasks.append(self.predict(input_item, model_name))
        
        # Run all predictions concurrently
        results = await asyncio.gather(*tasks)
        
        total_processing_time = time.time() - start_time
        
        return BatchModelOutput(
            batch_id=f"batch_{uuid.uuid4().hex[:10]}",
            results=results,
            total_processing_time=total_processing_time
        )
    
    async def add_to_batch_queue(
        self, 
        input_data: ModelInput, 
        model_name: str, 
        max_batch_size: int = APIConfig.MAX_BATCH_SIZE,
        timeout: float = APIConfig.BATCH_TIMEOUT
    ) -> ModelOutput:
        """Add a request to batch queue and process when batch is full or timeout occurs"""
        # Initialize batch queue for model if it doesn't exist
        if model_name not in self.batch_queue:
            self.batch_queue[model_name] = []
            self.batch_events[model_name] = asyncio.Event()
        
        # Create a future to return the result when ready
        result_future = asyncio.Future()
        self.batch_queue[model_name].append((input_data, result_future))
        
        # If batch is full, trigger processing
        if len(self.batch_queue[model_name]) >= max_batch_size:
            self.batch_events[model_name].set()
        else:
            # Otherwise, set a timeout
            asyncio.create_task(self._trigger_batch_after_timeout(model_name, timeout))
        
        # Wait for result
        return await result_future
    
    async def _trigger_batch_after_timeout(self, model_name: str, timeout: float):
        """Trigger batch processing after timeout"""
        await asyncio.sleep(timeout)
        if model_name in self.batch_events and not self.batch_events[model_name].is_set():
            self.batch_events[model_name].set()
    
    async def batch_processor(self, model_name: str):
        """Background task to process batches"""
        while True:
            # Wait for batch to be ready
            if model_name not in self.batch_events:
                self.batch_events[model_name] = asyncio.Event()
            
            await self.batch_events[model_name].wait()
            self.batch_events[model_name].clear()
            
            # Get current batch
            current_batch = self.batch_queue.get(model_name, [])
            self.batch_queue[model_name] = []
            
            if not current_batch:
                continue
            
            # Process batch
            inputs = [item[0] for item in current_batch]
            futures = [item[1] for item in current_batch]
            
            try:
                batch_input = BatchModelInput(inputs=inputs)
                batch_result = await self.predict_batch(batch_input, model_name)
                
                # Set results for each future
                for i, future in enumerate(futures):
                    if not future.done():
                        future.set_result(batch_result.results[i])
            except Exception as e:
                # If error, set exception for all futures
                error = HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Batch processing error: {str(e)}"
                )
                for future in futures:
                    if not future.done():
                        future.set_exception(error)

# Initialize model manager
model_manager = ModelManager()

# ==========================================================
# FastAPI Application
# ==========================================================

app = FastAPI(
    title=APIConfig.API_TITLE,
    description=APIConfig.API_DESCRIPTION,
    version=APIConfig.API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add metrics middleware if enabled
if APIConfig.ENABLE_METRICS:
    app.middleware("http")(metrics_middleware)

# Start batch processors
@app.on_event("startup")
async def startup_event():
    """Start background tasks on startup"""
    # Start batch processors for each model
    for model_name in APIConfig.AVAILABLE_MODELS:
        asyncio.create_task(model_manager.batch_processor(model_name))
    
    # Log startup
    logger.info(f"API started with {len(APIConfig.AVAILABLE_MODELS)} models available")

# ==========================================================
# API Endpoints
# ==========================================================

@app.get("/health", response_model=HealthStatus)
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": APIConfig.API_VERSION,
        "models": model_manager.models,
        "uptime": time.time() - metrics.start_time
    }

@app.get("/metrics", response_model=MetricsResponse, dependencies=[Depends(get_api_key)])
async def get_metrics():
    """Get API metrics"""
    return metrics.get_metrics()

@app.post(
    "/predict/{model_name}",
    response_model=ModelOutput,
    dependencies=[Depends(check_rate_limit)],
    responses={
        400: {"model": ErrorResponse},
        401: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        429: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
async def predict(
    model_name: str,
    input_data: ModelInput,
    batch: bool = False,
    user_info: Dict[str, Any] = Depends(check_rate_limit)
):
    """
    Run prediction with specified model
    
    - **model_name**: Name of the model to use
    - **input_data**: Text and parameters for prediction
    - **batch**: If true, request will be batched with others
    """
    try:
        logger.info(f"Prediction request for model '{model_name}' from user '{user_info['name']}'")
        
        if model_name not in APIConfig.AVAILABLE_MODELS:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model '{model_name}' not found. Available models: {APIConfig.AVAILABLE_MODELS}"
            )
        
        if batch:
            # Add to batch queue
            return await model_manager.add_to_batch_queue(input_data, model_name)
        else:
            # Process immediately
            return await model_manager.predict(input_data, model_name)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction error: {str(e)}"
        )

@app.post(
    "/batch/{model_name}",
    response_model=BatchModelOutput,
    dependencies=[Depends(check_rate_limit)],
    responses={
        400: {"model": ErrorResponse},
        401: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        429: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
async def batch_predict(
    model_name: str,
    batch_input: BatchModelInput,
    user_info: Dict[str, Any] = Depends(check_rate_limit)
):
    """
    Run batch prediction with specified model
    
    - **model_name**: Name of the model to use
    - **batch_input**: List of inputs for batch prediction
    """
    try:
        logger.info(f"Batch prediction request for model '{model_name}' with {len(batch_input.inputs)} inputs")
        
        if model_name not in APIConfig.AVAILABLE_MODELS:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model '{model_name}' not found. Available models: {APIConfig.AVAILABLE_MODELS}"
            )
        
        # Process batch
        return await model_manager.predict_batch(batch_input, model_name)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in batch prediction: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction error: {str(e)}"
        )

@app.get("/models", dependencies=[Depends(get_api_key)])
async def list_models():
    """List available models and their status"""
    return {
        "models": {
            name: {"status": status}
            for name, status in model_manager.models.items()
        },
        "default_model": APIConfig.DEFAULT_MODEL
    }

# Custom exception handler for more detailed errors
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom exception handler for HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": f"HTTP {exc.status_code}",
            "detail": exc.detail,
            "code": exc.status_code
        }
    )

# ==========================================================
# Main Application Entry
# ==========================================================

def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the FastAPI server"""
    uvicorn.run("model_api:app", host=host, port=port, reload=False)

if __name__ == "__main__":
    # Example of how to run the server
    run_server() 
#!/usr/bin/env python3
"""
Docker Deployment for AI Models

This script demonstrates how to containerize AI models using Docker,
including best practices for efficient and production-ready deployments.

Key functionalities:
- Creating optimized Dockerfiles for ML workloads
- Managing Docker builds and configuration
- Setting up multi-stage builds
- Local testing with Docker Compose
"""

import os
import sys
import argparse
import subprocess
import logging
import yaml
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set up paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
TEMPLATES_DIR = SCRIPT_DIR / "templates"
OUTPUT_DIR = PROJECT_ROOT / "output" / "docker_deployment"

# Ensure directories exist
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
TEMPLATES_DIR.mkdir(exist_ok=True)

# ==========================================================
# Docker File Templates
# ==========================================================

# Base Dockerfile template for a Python-based ML model
BASE_DOCKERFILE = """# Optimized Dockerfile for ML model deployment
# Multi-stage build for smaller final image

# Stage 1: Build environment
FROM python:3.10-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \\
    build-essential \\
    && apt-get clean \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Runtime environment
FROM python:3.10-slim

WORKDIR /app

# Copy installed packages from builder stage
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy application code
COPY . .

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser
RUN chown -R appuser:appuser /app
USER appuser

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=/app/models
ENV LOG_LEVEL=INFO

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["python", "app.py"]
"""

# Dockerfile template for GPU support
GPU_DOCKERFILE = """# GPU-enabled Dockerfile for ML model deployment
# Multi-stage build for smaller final image

# Stage 1: Build environment
FROM python:3.10-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \\
    build-essential \\
    && apt-get clean \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Runtime environment
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# Set up Python
RUN apt-get update && apt-get install -y --no-install-recommends \\
    python3 \\
    python3-pip \\
    && apt-get clean \\
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed packages from builder stage
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy application code
COPY . .

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser
RUN chown -R appuser:appuser /app
USER appuser

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=/app/models
ENV LOG_LEVEL=INFO
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
    CMD python3 -c "import requests; requests.get('http://localhost:8000/health')"

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["python3", "app.py"]
"""

# Docker Compose template
DOCKER_COMPOSE_TEMPLATE = """version: '3.8'

services:
  model-api:
    build:
      context: .
      dockerfile: {dockerfile}
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
    environment:
      - MODEL_NAME={model_name}
      - LOG_LEVEL=INFO
      - WORKERS={workers}
    restart: unless-stopped
    {gpu_config}

  # Optional Redis for caching (uncomment if needed)
  # redis:
  #   image: redis:alpine
  #   ports:
  #     - "6379:6379"
  #   volumes:
  #     - redis-data:/data
  #   restart: unless-stopped

# Optional volume for Redis (uncomment if needed)
# volumes:
#   redis-data:
"""

# Sample FastAPI app
FASTAPI_APP_TEMPLATE = """import os
import time
import logging
from typing import Dict, Any, List, Optional

import torch
import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Configure logging
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="ML Model API",
    description="API for serving machine learning model predictions",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model cache
MODEL = None
MODEL_NAME = os.environ.get("MODEL_NAME", "")
MODEL_PATH = os.environ.get("MODEL_PATH", "/app/models")

# Request model
class PredictionRequest(BaseModel):
    inputs: str
    parameters: Optional[Dict[str, Any]] = None

# Response model
class PredictionResponse(BaseModel):
    result: Any
    model_name: str
    processing_time: float


@app.on_event("startup")
async def startup_event():
    \"""Load the ML model on startup\"""
    global MODEL
    try:
        # Here you would load your model
        # MODEL = load_model(os.path.join(MODEL_PATH, MODEL_NAME))
        # For demo purposes:
        logger.info(f"Loading model {MODEL_NAME}...")
        time.sleep(2)  # Simulate model loading
        MODEL = "loaded"
        logger.info(f"Model {MODEL_NAME} loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        # Continue without model, will return error on prediction requests


@app.get("/health")
async def health_check():
    \"""Health check endpoint\"""
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model": MODEL_NAME}


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest, background_tasks: BackgroundTasks):
    \"""Run prediction with the model\"""
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = time.time()
    
    try:
        # Here you would run your model inference
        # result = MODEL.predict(request.inputs, **request.parameters)
        # For demo purposes:
        logger.info(f"Running prediction with input: {request.inputs[:50]}...")
        time.sleep(0.5)  # Simulate prediction time
        result = f"Prediction for: {request.inputs[:50]}..."
        
        processing_time = time.time() - start_time
        
        # Log request asynchronously
        background_tasks.add_task(log_request, request.inputs, processing_time)
        
        return {
            "result": result,
            "model_name": MODEL_NAME,
            "processing_time": processing_time
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


def log_request(input_text: str, processing_time: float):
    \"""Log request details (non-blocking)\"""
    logger.info(f"Processed request in {processing_time:.4f} seconds")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app", 
        host="0.0.0.0", 
        port=8000, 
        workers=int(os.environ.get("WORKERS", 1))
    )
"""

# Sample requirements file
REQUIREMENTS_TEMPLATE = """fastapi>=0.95.0
uvicorn>=0.22.0
pydantic>=2.0.0
python-multipart>=0.0.6
numpy>=1.24.0
torch>=2.0.0
transformers>=4.30.0
requests>=2.28.0
"""

# ==========================================================
# Docker Deployment Helper Functions
# ==========================================================

def create_project_structure(project_dir: Path, model_name: str, use_gpu: bool = False) -> None:
    """Create the project structure for Docker deployment"""
    logger.info(f"Creating project structure in {project_dir}")
    
    # Create project directory
    project_dir.mkdir(exist_ok=True, parents=True)
    
    # Create Dockerfile
    dockerfile_content = GPU_DOCKERFILE if use_gpu else BASE_DOCKERFILE
    dockerfile_name = "Dockerfile.gpu" if use_gpu else "Dockerfile"
    with open(project_dir / dockerfile_name, 'w') as f:
        f.write(dockerfile_content)
    
    # Create docker-compose.yml
    gpu_config = 'deploy:\n      resources:\n        reservations:\n          devices:\n            - driver: nvidia\n              count: 1\n              capabilities: [gpu]' if use_gpu else ''
    docker_compose_content = DOCKER_COMPOSE_TEMPLATE.format(
        dockerfile=dockerfile_name,
        model_name=model_name,
        workers=2,
        gpu_config=gpu_config
    )
    with open(project_dir / "docker-compose.yml", 'w') as f:
        f.write(docker_compose_content)
    
    # Create app.py
    with open(project_dir / "app.py", 'w') as f:
        f.write(FASTAPI_APP_TEMPLATE)
    
    # Create requirements.txt
    with open(project_dir / "requirements.txt", 'w') as f:
        f.write(REQUIREMENTS_TEMPLATE)
    
    # Create models directory
    models_dir = project_dir / "models"
    models_dir.mkdir(exist_ok=True)
    
    # Create README.md with instructions
    readme_content = f"""# Docker Deployment for {model_name}

## Quick Start

1. Build and start the container:
   ```
   docker-compose up -d
   ```

2. Check the API:
   ```
   curl http://localhost:8000/health
   ```

3. Make a prediction:
   ```
   curl -X POST http://localhost:8000/predict \\
     -H "Content-Type: application/json" \\
     -d '{{"inputs": "Your input text here", "parameters": {{"max_length": 50}}}}'
   ```

## Customization

- Modify `app.py` to implement your specific model loading and inference logic
- Adjust environment variables in `docker-compose.yml` as needed
- For production deployment, consider setting up proper authentication

## Resource Configuration

- Adjust the `deploy` section in `docker-compose.yml` for resource allocation
- For GPU deployment, ensure you have nvidia-docker installed
"""
    with open(project_dir / "README.md", 'w') as f:
        f.write(readme_content)
    
    logger.info(f"Project structure created successfully in {project_dir}")

def build_docker_image(project_dir: Path, image_name: str, tag: str = "latest") -> bool:
    """Build a Docker image from the project directory"""
    try:
        logger.info(f"Building Docker image {image_name}:{tag}")
        result = subprocess.run(
            ["docker", "build", "-t", f"{image_name}:{tag}", "."],
            cwd=project_dir,
            check=True,
            capture_output=True,
            text=True
        )
        logger.info(f"Docker image built successfully: {image_name}:{tag}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error building Docker image: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error building Docker image: {e}")
        return False

def run_docker_compose(project_dir: Path, detached: bool = True) -> bool:
    """Run docker-compose in the project directory"""
    try:
        cmd = ["docker-compose", "up"]
        if detached:
            cmd.append("-d")
        
        logger.info(f"Starting Docker containers with {'detached mode' if detached else 'interactive mode'}")
        result = subprocess.run(
            cmd,
            cwd=project_dir,
            check=True,
            capture_output=True,
            text=True
        )
        logger.info("Docker containers started successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error starting Docker containers: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error starting Docker containers: {e}")
        return False

# ==========================================================
# Deployment Strategies
# ==========================================================

def deploy_fastapi_model(output_dir: Path, model_name: str, use_gpu: bool = False) -> Path:
    """Deploy a model using FastAPI in a Docker container"""
    # Create a unique project name based on model name
    safe_model_name = model_name.lower().replace(" ", "-")
    project_name = f"{safe_model_name}-api"
    project_dir = output_dir / project_name
    
    # Create project structure
    create_project_structure(project_dir, model_name, use_gpu)
    
    logger.info(f"""
    Model deployment project created at: {project_dir}
    
    To build and run the Docker container:
    1. Navigate to the project directory:
       cd {project_dir}
       
    2. Build and start the container:
       docker-compose up -d
       
    3. Check the API:
       curl http://localhost:8000/health
       
    4. Make a prediction:
       curl -X POST http://localhost:8000/predict \\
         -H "Content-Type: application/json" \\
         -d '{{"inputs": "Your input text here", "parameters": {{"max_length": 50}}}}'
    """)
    
    return project_dir

# ==========================================================
# Main CLI Interface
# ==========================================================

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Docker Deployment for AI Models")
    
    parser.add_argument(
        "--model-name",
        type=str,
        default="huggingface-model",
        help="Name of the model to deploy"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(OUTPUT_DIR),
        help="Directory to output the deployment files"
    )
    
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Configure the deployment for GPU support"
    )
    
    parser.add_argument(
        "--build",
        action="store_true",
        help="Build the Docker image after creating the project"
    )
    
    parser.add_argument(
        "--run",
        action="store_true",
        help="Run the Docker container after building (implies --build)"
    )
    
    return parser.parse_args()

def main():
    """Main entry point"""
    args = parse_args()
    
    # Convert output_dir to Path and ensure it exists
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Deploy the model
    project_dir = deploy_fastapi_model(
        output_dir=output_dir,
        model_name=args.model_name,
        use_gpu=args.use_gpu
    )
    
    # Build and run if requested
    if args.build or args.run:
        image_name = f"{args.model_name.lower().replace(' ', '-')}-api"
        success = build_docker_image(project_dir, image_name)
        
        if success and args.run:
            run_docker_compose(project_dir)

# ==========================================================
# Example Usage and Demo
# ==========================================================

def demo():
    """Run a demonstration of Docker deployment"""
    print("Docker Deployment Demo for AI Models")
    print("===================================\n")
    
    model_name = "sentiment-analysis"
    output_dir = OUTPUT_DIR / "demo"
    
    print(f"Creating a Docker deployment project for '{model_name}' model\n")
    
    project_dir = deploy_fastapi_model(
        output_dir=output_dir,
        model_name=model_name,
        use_gpu=False
    )
    
    print("\nDeployment project created successfully!")
    print(f"Project directory: {project_dir}")
    print("\nThe project includes:")
    print("- Dockerfile with multi-stage build for optimized size")
    print("- docker-compose.yml for easy deployment")
    print("- FastAPI application with health check and prediction endpoints")
    print("- Requirements file with necessary dependencies")
    print("- README.md with deployment instructions")
    
    print("\nTo deploy this project, follow the instructions in the README.md file.")
    print("Demo completed successfully!")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
    else:
        demo() 
#!/usr/bin/env python3
"""
Fireworks.ai Functions Deployment

This script demonstrates how to deploy AI models using Fireworks.ai Functions,
a specialized platform for hosting and serving AI models.

Key functionalities:
- Setting up Fireworks.ai Function configurations
- Deploying models with different configurations
- Testing and monitoring deployed functions
- Cost optimization strategies
"""

import os
import sys
import json
import time
import logging
import requests
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set up paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output" / "fireworks_deployment"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# ==========================================================
# Fireworks.ai Configuration
# ==========================================================

class FireworksConfig:
    """Configuration for Fireworks.ai deployments"""
    
    # API configuration
    BASE_URL = "https://api.fireworks.ai/inference/v1"
    AUTH_URL = "https://api.fireworks.ai/auth/v1"
    FUNCTION_URL = "https://api.fireworks.ai/function/v1"
    
    # Default model configurations
    DEFAULT_MODEL_CONFIG = {
        "model": "accounts/fireworks/models/mixtral-8x7b-instruct",
        "max_tokens": 1024,
        "temperature": 0.7,
        "top_p": 0.9,
    }
    
    # Available models on Fireworks.ai
    AVAILABLE_MODELS = {
        "mixtral-8x7b": "accounts/fireworks/models/mixtral-8x7b-instruct",
        "llama-v2-70b": "accounts/fireworks/models/llama-v2-70b-chat",
        "llama-v2-13b": "accounts/fireworks/models/llama-v2-13b-chat",
        "llama-v2-7b": "accounts/fireworks/models/llama-v2-7b-chat",
        "mistral-7b": "accounts/fireworks/models/mistral-7b-instruct-4k",
    }
    
    # Function deployment configurations
    DEPLOYMENT_CONFIGS = {
        "development": {
            "replicas": 1,
            "scaling_config": {
                "min_replicas": 0,
                "max_replicas": 2,
                "target_num_ongoing_requests_per_replica": 10,
                "scale_down_delay_secs": 300,  # 5 minutes
            }
        },
        "production": {
            "replicas": 2,
            "scaling_config": {
                "min_replicas": 1,
                "max_replicas": 5,
                "target_num_ongoing_requests_per_replica": 20,
                "scale_down_delay_secs": 600,  # 10 minutes
            }
        }
    }

# ==========================================================
# Fireworks.ai API Client (Simulated)
# ==========================================================

class FireworksClient:
    """Client for interacting with Fireworks.ai API (simulated)"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the client with API key"""
        self.api_key = api_key or os.environ.get("FIREWORKS_API_KEY")
        if not self.api_key:
            logger.warning("No Fireworks API key provided. Using simulation mode.")
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def create_function(
        self, 
        name: str,
        model: str,
        prompt_template: str,
        configuration: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create a new function on Fireworks.ai (simulated)
        
        Args:
            name: Function name
            model: Model ID to use
            prompt_template: Template for prompt construction
            configuration: Deployment configuration
            
        Returns:
            Dict with function details
        """
        # In a real implementation, this would make an API call to Fireworks.ai
        # For this demo, we simulate the response
        
        if not self.api_key:
            # Simulate success response
            function_id = f"fn_{int(time.time())}"
            return {
                "id": function_id,
                "name": name,
                "model": model,
                "status": "creating",
                "endpoint": f"https://api.fireworks.ai/function/v1/{function_id}",
                "configuration": configuration
            }
        
        # In a real implementation with API key:
        # response = requests.post(
        #     f"{FireworksConfig.FUNCTION_URL}/functions",
        #     headers=self.headers,
        #     json={
        #         "name": name,
        #         "model": model,
        #         "prompt_template": prompt_template,
        #         "configuration": configuration
        #     }
        # )
        # return response.json()
        
        # Simulate response for demo purposes
        function_id = f"fn_{int(time.time())}"
        return {
            "id": function_id,
            "name": name,
            "model": model,
            "status": "creating",
            "endpoint": f"https://api.fireworks.ai/function/v1/{function_id}",
            "configuration": configuration
        }
    
    def get_function(self, function_id: str) -> Dict[str, Any]:
        """Get function details (simulated)"""
        # Simulate response
        return {
            "id": function_id,
            "name": "Example Function",
            "model": "accounts/fireworks/models/mixtral-8x7b-instruct",
            "status": "ready",
            "endpoint": f"https://api.fireworks.ai/function/v1/{function_id}",
            "created_at": time.time() - 300,  # 5 minutes ago
            "updated_at": time.time() - 60,  # 1 minute ago
            "metrics": {
                "requests_last_hour": 52,
                "average_latency_ms": 856,
                "p95_latency_ms": 1234
            }
        }
    
    def invoke_function(
        self, 
        function_id: str, 
        inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Invoke a function (simulated)"""
        # Simulate processing time based on input complexity
        processing_time = 0.5 + (len(json.dumps(inputs)) / 1000)
        time.sleep(processing_time)
        
        # Simulate response
        return {
            "id": f"resp_{int(time.time())}",
            "function_id": function_id,
            "output": "This is a simulated response from the Fireworks.ai function. In a real deployment, this would be generated by the model.",
            "model": "accounts/fireworks/models/mixtral-8x7b-instruct",
            "processing_time": processing_time,
            "token_usage": {
                "prompt_tokens": 42,
                "completion_tokens": 128,
                "total_tokens": 170
            }
        }
    
    def list_functions(self) -> List[Dict[str, Any]]:
        """List all functions (simulated)"""
        # Simulate response with a few functions
        return [
            {
                "id": f"fn_{100000 + i}",
                "name": f"Example Function {i}",
                "model": list(FireworksConfig.AVAILABLE_MODELS.values())[i % len(FireworksConfig.AVAILABLE_MODELS)],
                "status": "ready",
                "created_at": time.time() - (i * 86400)  # i days ago
            }
            for i in range(3)
        ]
    
    def delete_function(self, function_id: str) -> Dict[str, Any]:
        """Delete a function (simulated)"""
        # Simulate successful deletion
        return {
            "id": function_id,
            "status": "deleting"
        }
    
    def update_function(
        self, 
        function_id: str,
        configuration: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update function configuration (simulated)"""
        # Simulate successful update
        return {
            "id": function_id,
            "status": "updating",
            "configuration": configuration
        }

# ==========================================================
# Prompt Templates
# ==========================================================

# Example prompt templates for different use cases
PROMPT_TEMPLATES = {
    "text-generation": """
    <|system|>
    You are a helpful assistant that generates high-quality text based on the user's instructions.
    Your responses should be well-structured, coherent, and tailored to the specified topic and style.
    </|system|>
    
    <|user|>
    Please generate text about the following topic: {{topic}}
    
    Length: {{length}}
    Style: {{style}}
    </|user|>
    
    <|assistant|>
    """,
    
    "summarization": """
    <|system|>
    You are a summarization assistant that creates concise, accurate summaries of longer texts.
    Focus on the key points while maintaining the original meaning and context.
    </|system|>
    
    <|user|>
    Please summarize the following text:
    
    {{text}}
    
    Length: {{length}}
    </|user|>
    
    <|assistant|>
    """,
    
    "sentiment-analysis": """
    <|system|>
    You are a sentiment analysis assistant. Analyze the sentiment of the provided text and classify it as positive, negative, or neutral.
    Provide a brief explanation for your classification.
    </|system|>
    
    <|user|>
    Please analyze the sentiment of the following text:
    
    {{text}}
    </|user|>
    
    <|assistant|>
    """
}

# ==========================================================
# Deployment Functions
# ==========================================================

def create_function_deployment(
    client: FireworksClient,
    name: str,
    function_type: str,
    model_name: str,
    environment: str = "development"
) -> Dict[str, Any]:
    """
    Create a new function deployment on Fireworks.ai
    
    Args:
        client: Initialized FireworksClient
        name: Name for the function
        function_type: Type of function (e.g., text-generation, summarization)
        model_name: Short name of model to use (must be in FireworksConfig.AVAILABLE_MODELS)
        environment: Environment type (development or production)
        
    Returns:
        Dict with deployment details
    """
    # Validate inputs
    if function_type not in PROMPT_TEMPLATES:
        raise ValueError(f"Invalid function type: {function_type}. Must be one of {list(PROMPT_TEMPLATES.keys())}")
    
    if model_name not in FireworksConfig.AVAILABLE_MODELS:
        raise ValueError(f"Invalid model name: {model_name}. Must be one of {list(FireworksConfig.AVAILABLE_MODELS.keys())}")
    
    if environment not in FireworksConfig.DEPLOYMENT_CONFIGS:
        raise ValueError(f"Invalid environment: {environment}. Must be one of {list(FireworksConfig.DEPLOYMENT_CONFIGS.keys())}")
    
    # Get full model ID
    model_id = FireworksConfig.AVAILABLE_MODELS[model_name]
    
    # Get prompt template
    prompt_template = PROMPT_TEMPLATES[function_type]
    
    # Get deployment configuration
    deployment_config = FireworksConfig.DEPLOYMENT_CONFIGS[environment]
    
    # Create configuration
    configuration = {
        **FireworksConfig.DEFAULT_MODEL_CONFIG,
        "model": model_id,
        **deployment_config
    }
    
    # Create function
    logger.info(f"Creating '{name}' function with model {model_name} in {environment} environment")
    result = client.create_function(name, model_id, prompt_template, configuration)
    
    # Save deployment details to file
    output_file = OUTPUT_DIR / f"{name.lower().replace(' ', '_')}_deployment.json"
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    logger.info(f"Deployment details saved to {output_file}")
    
    return result

def test_function(
    client: FireworksClient,
    function_id: str,
    test_inputs: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Test a deployed function with sample inputs
    
    Args:
        client: Initialized FireworksClient
        function_id: ID of the function to test
        test_inputs: Input data for testing
        
    Returns:
        Dict with test results
    """
    logger.info(f"Testing function {function_id} with sample inputs")
    
    # Get function details to check status
    function_details = client.get_function(function_id)
    
    if function_details["status"] != "ready":
        logger.warning(f"Function {function_id} is not ready (status: {function_details['status']})")
        return {
            "success": False,
            "status": function_details["status"],
            "message": "Function is not ready for testing"
        }
    
    # Invoke the function
    start_time = time.time()
    result = client.invoke_function(function_id, test_inputs)
    total_time = time.time() - start_time
    
    # Create test report
    test_report = {
        "success": True,
        "function_id": function_id,
        "test_inputs": test_inputs,
        "result": result,
        "total_time": total_time,
        "timestamp": time.time()
    }
    
    # Save test report
    output_file = OUTPUT_DIR / f"test_report_{function_id}.json"
    with open(output_file, 'w') as f:
        json.dump(test_report, f, indent=2)
    
    logger.info(f"Test completed in {total_time:.2f} seconds. Report saved to {output_file}")
    
    return test_report

def estimate_costs(
    token_usage: Dict[str, int],
    model_name: str,
    estimated_requests_per_day: int
) -> Dict[str, Any]:
    """
    Estimate costs for a deployment
    
    Args:
        token_usage: Dict with token usage (prompt_tokens, completion_tokens)
        model_name: Model name for pricing lookup
        estimated_requests_per_day: Estimated number of daily requests
        
    Returns:
        Dict with cost estimates
    """
    # These are example pricing rates - actual rates should be obtained from Fireworks.ai
    model_pricing = {
        "mixtral-8x7b": {"input": 0.0027, "output": 0.0027},  # per 1K tokens
        "llama-v2-70b": {"input": 0.0087, "output": 0.0087},
        "llama-v2-13b": {"input": 0.0027, "output": 0.0027},
        "llama-v2-7b": {"input": 0.002, "output": 0.002},
        "mistral-7b": {"input": 0.002, "output": 0.002},
    }
    
    if model_name not in model_pricing:
        raise ValueError(f"No pricing information for model: {model_name}")
    
    pricing = model_pricing[model_name]
    
    # Calculate daily cost
    prompt_cost = (token_usage["prompt_tokens"] / 1000) * pricing["input"] * estimated_requests_per_day
    completion_cost = (token_usage["completion_tokens"] / 1000) * pricing["output"] * estimated_requests_per_day
    daily_cost = prompt_cost + completion_cost
    
    # Calculate monthly cost (approximate)
    monthly_cost = daily_cost * 30
    
    return {
        "model": model_name,
        "daily_requests": estimated_requests_per_day,
        "token_usage_per_request": token_usage,
        "cost_per_request": {
            "prompt_cost": (token_usage["prompt_tokens"] / 1000) * pricing["input"],
            "completion_cost": (token_usage["completion_tokens"] / 1000) * pricing["output"],
            "total": (token_usage["prompt_tokens"] / 1000) * pricing["input"] + 
                    (token_usage["completion_tokens"] / 1000) * pricing["output"]
        },
        "daily_cost": daily_cost,
        "monthly_cost": monthly_cost,
        "pricing_rates": {
            "input_per_1k": pricing["input"],
            "output_per_1k": pricing["output"]
        }
    }

# ==========================================================
# Main CLI Interface
# ==========================================================

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Fireworks.ai Function Deployment Tool")
    
    # Setup subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Create function command
    create_parser = subparsers.add_parser("create", help="Create a new Fireworks.ai function")
    create_parser.add_argument("--name", type=str, required=True, help="Function name")
    create_parser.add_argument(
        "--type", 
        type=str, 
        required=True, 
        choices=list(PROMPT_TEMPLATES.keys()),
        help="Function type"
    )
    create_parser.add_argument(
        "--model", 
        type=str, 
        default="mixtral-8x7b",
        choices=list(FireworksConfig.AVAILABLE_MODELS.keys()),
        help="Model to use"
    )
    create_parser.add_argument(
        "--env", 
        type=str, 
        default="development", 
        choices=list(FireworksConfig.DEPLOYMENT_CONFIGS.keys()),
        help="Deployment environment"
    )
    
    # Test function command
    test_parser = subparsers.add_parser("test", help="Test a deployed function")
    test_parser.add_argument("--id", type=str, required=True, help="Function ID to test")
    test_parser.add_argument("--inputs", type=str, required=True, help="JSON string or file path with test inputs")
    
    # List functions command
    list_parser = subparsers.add_parser("list", help="List deployed functions")
    
    # Estimate costs command
    cost_parser = subparsers.add_parser("estimate-cost", help="Estimate deployment costs")
    cost_parser.add_argument(
        "--model", 
        type=str, 
        required=True,
        choices=list(FireworksConfig.AVAILABLE_MODELS.keys()),
        help="Model to estimate costs for"
    )
    cost_parser.add_argument(
        "--prompt-tokens", 
        type=int, 
        default=500,
        help="Average prompt tokens per request"
    )
    cost_parser.add_argument(
        "--completion-tokens", 
        type=int, 
        default=200,
        help="Average completion tokens per request"
    )
    cost_parser.add_argument(
        "--requests-per-day", 
        type=int, 
        default=1000,
        help="Estimated requests per day"
    )
    
    return parser.parse_args()

def main():
    """Main entry point"""
    args = parse_args()
    
    # Initialize client
    client = FireworksClient()
    
    if args.command == "create":
        # Create a new function
        result = create_function_deployment(
            client=client,
            name=args.name,
            function_type=args.type,
            model_name=args.model,
            environment=args.env
        )
        print(f"Function created with ID: {result['id']}")
        print(f"Status: {result['status']}")
        
    elif args.command == "test":
        # Parse inputs
        if args.inputs.endswith(".json") and os.path.isfile(args.inputs):
            with open(args.inputs, 'r') as f:
                test_inputs = json.load(f)
        else:
            test_inputs = json.loads(args.inputs)
        
        # Test the function
        result = test_function(client, args.id, test_inputs)
        if result["success"]:
            print(f"Test completed successfully in {result['total_time']:.2f} seconds")
            print(f"Output: {result['result']['output']}")
        else:
            print(f"Test failed: {result['message']}")
            
    elif args.command == "list":
        # List deployed functions
        functions = client.list_functions()
        print(f"Found {len(functions)} deployed functions:")
        for i, func in enumerate(functions, 1):
            print(f"{i}. {func['name']} (ID: {func['id']}, Status: {func['status']})")
            
    elif args.command == "estimate-cost":
        # Estimate deployment costs
        token_usage = {
            "prompt_tokens": args.prompt_tokens,
            "completion_tokens": args.completion_tokens,
            "total_tokens": args.prompt_tokens + args.completion_tokens
        }
        
        cost_estimate = estimate_costs(
            token_usage=token_usage,
            model_name=args.model,
            estimated_requests_per_day=args.requests_per_day
        )
        
        print(f"Cost Estimate for {args.model}:")
        print(f"Daily Requests: {cost_estimate['daily_requests']:,}")
        print(f"Cost per Request: ${cost_estimate['cost_per_request']['total']:.6f}")
        print(f"Daily Cost: ${cost_estimate['daily_cost']:.2f}")
        print(f"Monthly Cost: ${cost_estimate['monthly_cost']:.2f}")
        
    else:
        print("No command specified. Use --help to see available commands.")

# ==========================================================
# Example Usage and Demo
# ==========================================================

def demo():
    """Run a demonstration of Fireworks.ai Function deployment"""
    print("Fireworks.ai Function Deployment Demo")
    print("=====================================\n")
    
    # Initialize client
    client = FireworksClient()
    
    # Step 1: Create a function
    print("1. Creating a text generation function...\n")
    function_result = create_function_deployment(
        client=client,
        name="Demo Text Generator",
        function_type="text-generation",
        model_name="mixtral-8x7b",
        environment="development"
    )
    
    function_id = function_result["id"]
    print(f"   Function created with ID: {function_id}")
    print(f"   Status: {function_result['status']}")
    print(f"   Endpoint: {function_result['endpoint']}\n")
    
    # Step 2: Test the function
    print("2. Testing the function with sample inputs...\n")
    test_inputs = {
        "topic": "artificial intelligence and its impact on society",
        "length": "medium",
        "style": "informative"
    }
    
    test_result = test_function(client, function_id, test_inputs)
    print(f"   Test completed successfully in {test_result['total_time']:.2f} seconds")
    print(f"   Sample output: {test_result['result']['output'][:150]}...\n")
    
    # Step 3: Estimate costs
    print("3. Estimating deployment costs...\n")
    token_usage = test_result["result"]["token_usage"]
    
    cost_estimate = estimate_costs(
        token_usage=token_usage,
        model_name="mixtral-8x7b",
        estimated_requests_per_day=1000
    )
    
    print(f"   Estimated costs for 1,000 requests per day:")
    print(f"   - Cost per request: ${cost_estimate['cost_per_request']['total']:.6f}")
    print(f"   - Daily cost: ${cost_estimate['daily_cost']:.2f}")
    print(f"   - Monthly cost: ${cost_estimate['monthly_cost']:.2f}\n")
    
    # Step 4: Show optimization tips
    print("4. Cost Optimization Tips:\n")
    print("   - Use smaller models for simpler tasks (e.g., mistral-7b instead of mixtral-8x7b)")
    print("   - Implement caching for common requests to reduce API calls")
    print("   - Use efficient prompt engineering to reduce token usage")
    print("   - Configure proper scaling parameters to avoid over-provisioning")
    print("   - Consider batching requests for better throughput and cost efficiency\n")
    
    print("Demo completed successfully!")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
    else:
        demo() 
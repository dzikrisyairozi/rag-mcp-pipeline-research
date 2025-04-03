#!/usr/bin/env python3
"""
Fireworks.ai Function Testing Script

This script provides examples of how to test Fireworks.ai functions using Python's requests library.
It demonstrates different ways to interact with deployed functions, handling authentication,
and processing responses.
"""

import os
import sys
import json
import time
import requests
from typing import Dict, Any, Optional
from pathlib import Path

# Set up paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output" / "fireworks_functions"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Example API key - you would normally load this from an environment variable
# Set your API key with: export FIREWORKS_API_KEY=your_api_key_here
API_KEY = os.environ.get("FIREWORKS_API_KEY", "demo_key")

# Base URLs for Fireworks.ai
BASE_URL = "https://api.fireworks.ai/inference/v1"
FUNCTION_URL = "https://api.fireworks.ai/function/v1"

# Headers for API requests
DEFAULT_HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

def test_text_generation_function(function_id: str, topic: str, length: str = "medium", style: str = "informative") -> Dict[str, Any]:
    """
    Test a text generation function with the given parameters
    
    Args:
        function_id: The ID of the deployed function
        topic: Topic to generate text about
        length: Desired length (short, medium, long)
        style: Writing style (informative, persuasive, creative, etc.)
        
    Returns:
        Dict containing the function response
    """
    # Construct the API endpoint URL
    url = f"{FUNCTION_URL}/{function_id}"
    
    # Construct the request payload with input parameters
    payload = {
        "inputs": {
            "topic": topic,
            "length": length,
            "style": style
        }
    }
    
    # Print request information
    print(f"Making request to: {url}")
    print(f"Request payload: {json.dumps(payload, indent=2)}")
    
    try:
        # Make the API request
        start_time = time.time()
        response = requests.post(url, headers=DEFAULT_HEADERS, json=payload)
        elapsed_time = time.time() - start_time
        
        # Check if the request was successful
        response.raise_for_status()
        
        # Parse the JSON response
        result = response.json()
        
        print(f"Request completed in {elapsed_time:.2f} seconds")
        print(f"Response status: {response.status_code}")
        
        # Save response to file
        output_file = OUTPUT_DIR / f"text_generation_response_{int(time.time())}.json"
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)
        
        print(f"Response saved to: {output_file}")
        
        return result
    
    except requests.exceptions.RequestException as e:
        print(f"Error making request: {e}")
        if hasattr(e, "response") and e.response is not None:
            print(f"Response status code: {e.response.status_code}")
            print(f"Response body: {e.response.text}")
        return {"error": str(e)}

def test_summarization_function(function_id: str, text: str, length: str = "short") -> Dict[str, Any]:
    """
    Test a summarization function with the given parameters
    
    Args:
        function_id: The ID of the deployed function
        text: The text to summarize
        length: Desired summary length (short, medium, long)
        
    Returns:
        Dict containing the function response
    """
    # Construct the API endpoint URL
    url = f"{FUNCTION_URL}/{function_id}"
    
    # Construct the request payload with input parameters
    payload = {
        "inputs": {
            "text": text,
            "length": length
        }
    }
    
    # Print request information
    print(f"Making request to: {url}")
    print(f"Request payload text length: {len(text)} characters")
    
    try:
        # Make the API request
        start_time = time.time()
        response = requests.post(url, headers=DEFAULT_HEADERS, json=payload)
        elapsed_time = time.time() - start_time
        
        # Check if the request was successful
        response.raise_for_status()
        
        # Parse the JSON response
        result = response.json()
        
        print(f"Request completed in {elapsed_time:.2f} seconds")
        print(f"Response status: {response.status_code}")
        
        # Save response to file
        output_file = OUTPUT_DIR / f"summarization_response_{int(time.time())}.json"
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)
        
        print(f"Response saved to: {output_file}")
        
        return result
    
    except requests.exceptions.RequestException as e:
        print(f"Error making request: {e}")
        if hasattr(e, "response") and e.response is not None:
            print(f"Response status code: {e.response.status_code}")
            print(f"Response body: {e.response.text}")
        return {"error": str(e)}

def test_sentiment_analysis_function(function_id: str, text: str) -> Dict[str, Any]:
    """
    Test a sentiment analysis function with the given parameters
    
    Args:
        function_id: The ID of the deployed function
        text: The text to analyze
        
    Returns:
        Dict containing the function response
    """
    # Construct the API endpoint URL
    url = f"{FUNCTION_URL}/{function_id}"
    
    # Construct the request payload with input parameters
    payload = {
        "inputs": {
            "text": text
        }
    }
    
    # Print request information
    print(f"Making request to: {url}")
    print(f"Text for sentiment analysis: '{text[:100]}...' ({len(text)} chars)")
    
    try:
        # Make the API request
        start_time = time.time()
        response = requests.post(url, headers=DEFAULT_HEADERS, json=payload)
        elapsed_time = time.time() - start_time
        
        # Check if the request was successful
        response.raise_for_status()
        
        # Parse the JSON response
        result = response.json()
        
        print(f"Request completed in {elapsed_time:.2f} seconds")
        print(f"Response status: {response.status_code}")
        
        # Save response to file
        output_file = OUTPUT_DIR / f"sentiment_analysis_response_{int(time.time())}.json"
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)
        
        print(f"Response saved to: {output_file}")
        
        return result
    
    except requests.exceptions.RequestException as e:
        print(f"Error making request: {e}")
        if hasattr(e, "response") and e.response is not None:
            print(f"Response status code: {e.response.status_code}")
            print(f"Response body: {e.response.text}")
        return {"error": str(e)}

def get_function_details(function_id: str) -> Dict[str, Any]:
    """
    Get details about a deployed function
    
    Args:
        function_id: The ID of the deployed function
        
    Returns:
        Dict containing the function details
    """
    # Construct the API endpoint URL
    url = f"{FUNCTION_URL}/functions/{function_id}"
    
    try:
        # Make the API request
        response = requests.get(url, headers=DEFAULT_HEADERS)
        
        # Check if the request was successful
        response.raise_for_status()
        
        # Parse the JSON response
        result = response.json()
        
        # Save function details to file
        output_file = OUTPUT_DIR / f"function_details_{function_id}.json"
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)
        
        print(f"Function details saved to: {output_file}")
        
        return result
    
    except requests.exceptions.RequestException as e:
        print(f"Error getting function details: {e}")
        if hasattr(e, "response") and e.response is not None:
            print(f"Response status code: {e.response.status_code}")
            print(f"Response body: {e.response.text}")
        return {"error": str(e)}

def list_deployed_functions() -> Dict[str, Any]:
    """
    List all deployed functions
    
    Returns:
        Dict containing the list of functions
    """
    # Construct the API endpoint URL
    url = f"{FUNCTION_URL}/functions"
    
    try:
        # Make the API request
        response = requests.get(url, headers=DEFAULT_HEADERS)
        
        # Check if the request was successful
        response.raise_for_status()
        
        # Parse the JSON response
        result = response.json()
        
        # Save function list to file
        output_file = OUTPUT_DIR / f"function_list_{int(time.time())}.json"
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)
        
        print(f"Function list saved to: {output_file}")
        
        return result
    
    except requests.exceptions.RequestException as e:
        print(f"Error listing functions: {e}")
        if hasattr(e, "response") and e.response is not None:
            print(f"Response status code: {e.response.status_code}")
            print(f"Response body: {e.response.text}")
        return {"error": str(e)}

def main():
    """Main function with usage examples"""
    # Check if API key is set
    if API_KEY == "demo_key":
        print("Warning: Using demo API key. Set FIREWORKS_API_KEY environment variable for actual use.")
        print("Example commands will be shown but not executed.")
        
        # Example function IDs (for demonstration purposes)
        text_gen_function_id = "fn_example_text_gen_123"
        summarization_function_id = "fn_example_summarization_456"
        sentiment_function_id = "fn_example_sentiment_789"
        
        print("\n=== Example Usage ===\n")
        
        print("1. Test Text Generation Function:")
        print(f"   function_id = '{text_gen_function_id}'")
        print("   topic = 'artificial intelligence ethics'")
        print("   length = 'medium'")
        print("   style = 'informative'")
        
        print("\n2. Test Summarization Function:")
        print(f"   function_id = '{summarization_function_id}'")
        print("   text = 'Long text to be summarized...'")
        print("   length = 'short'")
        
        print("\n3. Test Sentiment Analysis Function:")
        print(f"   function_id = '{sentiment_function_id}'")
        print("   text = 'I really enjoyed using this product! It exceeded my expectations.'")
        
        return
    
    # If API key is set, you can uncomment and use these examples
    
    # Example 1: List deployed functions
    # functions = list_deployed_functions()
    # print(f"Found {len(functions.get('functions', []))} deployed functions")
    
    # Example 2: Test a text generation function
    # Replace with an actual function ID
    # function_id = "your_function_id_here"
    # result = test_text_generation_function(
    #     function_id=function_id,
    #     topic="artificial intelligence ethics",
    #     length="medium",
    #     style="informative"
    # )
    # print(f"Output: {result.get('output', 'No output')[:200]}...")
    
    # Example 3: Test a summarization function
    # Replace with an actual function ID
    # function_id = "your_function_id_here"
    # with open("sample_text.txt", "r") as f:
    #     text = f.read()
    # result = test_summarization_function(
    #     function_id=function_id,
    #     text=text,
    #     length="short"
    # )
    # print(f"Summary: {result.get('output', 'No output')}")

if __name__ == "__main__":
    main() 
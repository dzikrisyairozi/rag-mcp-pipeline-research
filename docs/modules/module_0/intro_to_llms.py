#!/usr/bin/env python3
"""
Introduction to Large Language Models (LLMs)

This script provides a hands-on introduction to working with Large Language Models (LLMs).
It covers the basics of interacting with LLMs through APIs, understanding prompts,
and setting up a foundation for more advanced topics.

Note: This script can be run directly or converted to a Jupyter notebook
with: jupyter nbconvert --to notebook --execute intro_to_llms.py
"""

# 1. Setting Up the Environment
# Install required packages if not already installed
# %pip install openai python-dotenv langchain

# 2. Setting Up API Access
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Access API key
api_key = os.getenv("OPENAI_API_KEY")

# Check if API key is available
if not api_key:
    print("⚠️ API key not found! Please create a .env file in the root directory with your OpenAI API key:")
    print("OPENAI_API_KEY=your_api_key_here")
else:
    print("✅ API key loaded successfully!")

# 3. Basic Interaction with an LLM
from openai import OpenAI

# Initialize the client
client = OpenAI(api_key=api_key)

# Simple completion request
def get_completion(prompt, model="gpt-3.5-turbo"):
    """Get a completion from the OpenAI API"""
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7,
    )
    return response.choices[0].message.content

# Test the function
prompt = "Explain what a Large Language Model is in simple terms."
print("\n=== Basic LLM Interaction ===")
print(f"Prompt: {prompt}")
response = get_completion(prompt) if api_key else "API key not available. Skipping API call."
print(f"Response: {response}")

# 4. Understanding Prompt Engineering
print("\n=== Prompt Engineering Examples ===")

# Basic prompt
basic_prompt = "Write a poem about AI."
print("Basic prompt:")
print(f"- Prompt: {basic_prompt}")
basic_response = get_completion(basic_prompt) if api_key else "API key not available. Skipping API call."
print(f"- Response: {basic_response}")
print()

# More detailed prompt with specific instructions
detailed_prompt = """Write a short poem about artificial intelligence with the following characteristics:
- Four lines only
- Rhyming scheme AABB
- Include a metaphor comparing AI to a river
- End with a thought-provoking question
"""
print("Detailed prompt:")
print(f"- Prompt: {detailed_prompt}")
detailed_response = get_completion(detailed_prompt) if api_key else "API key not available. Skipping API call."
print(f"- Response: {detailed_response}")

# 5. Introduction to RAG (Retrieval Augmented Generation)
print("\n=== Simple RAG Example ===")

# Simulated knowledge base (in a real RAG system, this would be retrieved from a vector database)
knowledge_base = {
    "MCP server": "Multi-Cloud Processing (MCP) servers are systems designed to manage API interactions across multiple cloud services. They standardize command execution and authentication, allowing for consistent interactions with different third-party services.",
    "RAG": "Retrieval Augmented Generation (RAG) is an AI framework that enhances large language models by retrieving relevant information from external knowledge sources before generating responses. This helps provide more accurate, up-to-date, and contextually relevant outputs."
}

def simple_rag(query):
    """A very simple RAG implementation"""
    # Step 1: Determine what knowledge to retrieve (in a real system, this would use embeddings and similarity)
    retrieved_context = ""
    for key, value in knowledge_base.items():
        if key.lower() in query.lower():
            retrieved_context += f"{key}: {value}\n"
    
    # Step 2: If we found relevant information, augment the prompt with it
    if retrieved_context:
        augmented_prompt = f"""Based on the following information, please answer the query.
        
        Information:
        {retrieved_context}
        
        Query: {query}
        """
    else:
        augmented_prompt = query
    
    # Step 3: Generate a response using the augmented prompt
    return get_completion(augmented_prompt) if api_key else "API key not available. Skipping API call."

# Test our simple RAG implementation
rag_query = "What is a MCP server and how does it relate to API integration?"
print(f"Query: {rag_query}")
rag_response = simple_rag(rag_query)
print(f"RAG-augmented response: {rag_response}")

# 6. Next Steps
print("\n=== Next Steps ===")
print("""
This script provided a very basic introduction to working with LLMs and a simple conceptual example of RAG. 
To continue your learning:

1. Experiment with different prompts and observe how the outputs change
2. Try different models (if available through your API)
3. Explore the OpenAI or equivalent documentation for more advanced features
4. Move on to more advanced concepts in Module 1

Remember, this is just the beginning of your journey into LLMs, RAG, and MCP servers!
""")

if __name__ == "__main__":
    print("\nScript completed successfully!")
    print("To convert this to a Jupyter notebook, run:")
    print("jupyter nbconvert --to notebook --execute intro_to_llms.py") 
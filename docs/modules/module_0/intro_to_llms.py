#!/usr/bin/env python3
"""
Introduction to Large Language Models (LLMs)

This script provides a hands-on introduction to working with Large Language Models (LLMs).
It covers the basics of interacting with LLMs through models from Hugging Face (free),
understanding prompts, and setting up a foundation for more advanced topics.

Why use Hugging Face instead of OpenAI?
1. Free to use - no API key or payment required
2. Open-source models accessible to everyone
3. Runs models locally on your machine
4. Greater transparency in how models work
5. Variety of model sizes to fit your hardware capabilities

Note: This script can be run directly or converted to a Jupyter notebook
with: jupyter nbconvert --to notebook --execute intro_to_llms.py
"""

# 1. Setting Up the Environment
# Install required packages if not already installed
# Run this in your terminal:
# pip install transformers torch sentence-transformers

print("=== Setting up LLM environment using Hugging Face (Free Models) ===\n")

# 2. Import the required libraries
import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import time

# 3. Basic Interaction with an LLM
print("Loading a small pretrained model from Hugging Face...")
print("Note: The first time this runs, it will download the model files (may take a few minutes).")

try:
    # Using a small GPT-2 model that works on most hardware
    generator = pipeline('text-generation', model='distilgpt2')
    print("✅ Model loaded successfully!\n")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("Please check your internet connection and try again.")
    exit(1)

# Simple completion function using Hugging Face models
def get_completion(prompt, max_length=100):
    """Get a completion from a Hugging Face model"""
    try:
        # Generate text based on the prompt
        result = generator(prompt, max_length=max_length, num_return_sequences=1, 
                          pad_token_id=generator.tokenizer.eos_token_id)
        
        # Extract and return the generated text
        return result[0]['generated_text']
    except Exception as e:
        print(f"Error generating completion: {e}")
        return f"Error: {str(e)}"

# Test the function
prompt = "Explain what a Large Language Model is in simple terms."
print("=== Basic LLM Interaction ===")
print(f"Prompt: {prompt}")
print("Generating response (this may take a moment)...")
response = get_completion(prompt)
print(f"Response: {response}\n")

# 4. Understanding Prompt Engineering
print("=== Prompt Engineering Examples ===")

# Basic prompt
basic_prompt = "Write a poem about AI."
print("Basic prompt:")
print(f"- Prompt: {basic_prompt}")
print("Generating response...")
basic_response = get_completion(basic_prompt, max_length=150)
print(f"- Response: {basic_response}\n")

# More detailed prompt with specific instructions
detailed_prompt = """Write a short poem about artificial intelligence with the following characteristics:
- Four lines only
- Rhyming scheme AABB
- Include a metaphor comparing AI to a river
- End with a thought-provoking question
"""
print("Detailed prompt:")
print(f"- Prompt: {detailed_prompt}")
print("Generating response...")
detailed_response = get_completion(detailed_prompt, max_length=200)
print(f"- Response: {detailed_response}\n")

# 5. Introduction to RAG (Retrieval Augmented Generation)
print("=== Simple RAG Example ===")
print("Loading sentence embedding model for RAG...")

try:
    # Load a small sentence embedding model
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    print("✅ Embedding model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading embedding model: {e}")
    print("Continuing with simulated embeddings...")
    embedding_model = None

# Simulated knowledge base (in a real RAG system, this would be retrieved from a vector database)
knowledge_base = {
    "MCP server": "Multi-Cloud Processing (MCP) servers are systems designed to manage API interactions across multiple cloud services. They standardize command execution and authentication, allowing for consistent interactions with different third-party services.",
    "RAG": "Retrieval Augmented Generation (RAG) is an AI framework that enhances large language models by retrieving relevant information from external knowledge sources before generating responses. This helps provide more accurate, up-to-date, and contextually relevant outputs."
}

def simple_rag(query):
    """A very simple RAG implementation"""
    print(f"Processing query: '{query}'")
    
    # Step 1: Determine what knowledge to retrieve
    retrieved_context = ""
    
    if embedding_model:
        # Using embeddings for more accurate semantic search
        print("Using sentence embeddings for semantic retrieval...")
        query_embedding = embedding_model.encode(query)
        
        # Calculate similarity with each knowledge base entry
        best_match = None
        best_score = -1
        
        for key, value in knowledge_base.items():
            key_embedding = embedding_model.encode(key)
            # Cosine similarity between query and key
            similarity = torch.nn.functional.cosine_similarity(
                torch.tensor(query_embedding).unsqueeze(0),
                torch.tensor(key_embedding).unsqueeze(0)
            ).item()
            
            if similarity > best_score:
                best_score = similarity
                best_match = key
                
        if best_score > 0.5:  # Threshold for relevance
            retrieved_context = f"{best_match}: {knowledge_base[best_match]}\n"
            print(f"Found relevant information about: {best_match}")
    else:
        # Simple keyword matching as fallback
        print("Using keyword matching for retrieval...")
        for key, value in knowledge_base.items():
            if key.lower() in query.lower():
                retrieved_context += f"{key}: {value}\n"
                print(f"Found keyword match: {key}")
    
    # Step 2: If we found relevant information, augment the prompt with it
    if retrieved_context:
        print("Creating augmented prompt with retrieved information...")
        augmented_prompt = f"""Based on the following information, please answer the query.
        
        Information:
        {retrieved_context}
        
        Query: {query}
        """
    else:
        print("No relevant information found. Using original query.")
        augmented_prompt = query
    
    # Step 3: Generate a response using the augmented prompt
    print("Generating response with augmented context...")
    return get_completion(augmented_prompt, max_length=200)

# Test our simple RAG implementation
rag_query = "What is a MCP server and how does it relate to API integration?"
print(f"Query: {rag_query}")
rag_response = simple_rag(rag_query)
print(f"RAG-augmented response: {rag_response}\n")

# 6. Next Steps
print("=== Next Steps ===")
print("""
This script provided a very basic introduction to working with LLMs and a simple conceptual example of RAG. 
To continue your learning:

1. Experiment with different prompts and observe how the outputs change
2. Try different models from Hugging Face (e.g., 'gpt2', 'gpt2-medium', 'EleutherAI/gpt-neo-125M')
3. Explore the Hugging Face documentation for more advanced features
4. Move on to more advanced concepts in Module 1

Remember, this is just the beginning of your journey into LLMs, RAG, and MCP servers!
""")

if __name__ == "__main__":
    print("\nScript completed successfully!")
    print("\n=== IMPORTANT NOTES ===")
    print("1. The first run is slower as it downloads models from Hugging Face")
    print("2. Outputs from these smaller models may not be as coherent as commercial APIs")
    print("3. For best results, run on a computer with at least 4GB of RAM")
    print("4. Using a GPU will significantly speed up processing time if available")
    print("\nTo convert this to a Jupyter notebook, run:")
    print("jupyter nbconvert --to notebook --execute intro_to_llms.py") 
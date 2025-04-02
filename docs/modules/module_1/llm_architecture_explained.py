#!/usr/bin/env python3
"""
LLM Architecture Explained

This script provides an interactive explanation of how Large Language Models work,
with visualizations and examples to help you understand the core concepts.

Run this script to learn about:
- Tokenization: How text is broken down into tokens
- Attention mechanisms: How models focus on relevant information
- Context management: How models keep track of what's been said
- Output generation: How models produce responses

No external API keys required - uses local Hugging Face models.
"""

import os
import sys
import numpy as np
import gradio as gr
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Create output directory if it doesn't exist
script_path = Path(__file__).resolve()
project_root = script_path.parent.parent.parent.parent
output_dir = project_root / "output"
output_dir.mkdir(exist_ok=True)

# Initialize models and tokenizers
print("Loading models and tokenizers...")
try:
    # Load a small model for demonstration
    model_name = "distilgpt2"  # A smaller GPT-2 version
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    print(f"✅ Successfully loaded {model_name}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("Please check your internet connection or try a different model.")
    sys.exit(1)

# Function to visualize tokenization
def visualize_tokenization(text):
    """Tokenize text and visualize the tokens"""
    # Tokenize the input text
    tokens = tokenizer.tokenize(text)
    token_ids = tokenizer.encode(text)
    
    # Create a visualization
    fig, ax = plt.subplots(figsize=(12, len(tokens) * 0.4 + 1))
    
    # Display tokens and their IDs
    for i, (token, token_id) in enumerate(zip(tokens, token_ids[1:-1])):  # Skip special tokens
        token_display = token.replace('Ġ', ' ')  # Replace GPT-2 space character for display
        ax.text(0.1, i, f"Token {i+1}: '{token_display}'", fontsize=12)
        ax.text(0.6, i, f"ID: {token_id}", fontsize=12)
    
    ax.set_ylim(len(tokens), -1)
    ax.set_xlim(0, 1)
    ax.axis('off')
    plt.tight_layout()
    
    # Save the image
    output_path = output_dir / "tokenization.png"
    plt.savefig(output_path)
    plt.close()
    
    return {
        "tokens": tokens,
        "token_ids": token_ids[1:-1],  # Skip special tokens
        "visualization": str(output_path)
    }

# Function to demonstrate attention mechanisms
def visualize_attention(text):
    """Generate and visualize attention patterns for text"""
    inputs = tokenizer(text, return_tensors="pt")
    
    # Generate output while capturing attention
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
        
    # Get attention from the last layer
    attentions = outputs.attentions[-1].mean(dim=1)[0].cpu().numpy()
    
    # Create a visualization of the attention map
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(attentions, cmap='viridis')
    
    # Add labels
    ax.set_xticks(np.arange(len(tokens)))
    ax.set_yticks(np.arange(len(tokens)))
    ax.set_xticklabels([t.replace('Ġ', ' ') for t in tokens], rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticklabels([t.replace('Ġ', ' ') for t in tokens])
    
    # Add a colorbar
    plt.colorbar(im)
    
    plt.title("Attention Visualization")
    plt.tight_layout()
    
    # Save the image
    output_path = output_dir / "attention.png"
    plt.savefig(output_path)
    plt.close()
    
    return str(output_path)

# Function to demonstrate output generation
def generate_text(prompt, max_length=50, temperature=0.7):
    """Generate text from a prompt with the model"""
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Generate text with configurable parameters
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_length=max_length,
            temperature=temperature,
            do_sample=True,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True
        )
    
    # Get the generated text
    generated_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    
    # Get token probabilities for visualization
    token_probs = []
    
    # Skip the input tokens
    input_length = inputs["input_ids"].shape[1]
    
    # Process each new token's scores
    for i, scores in enumerate(outputs.scores):
        # Get probabilities from logits for the selected token
        token_id = outputs.sequences[0][input_length + i].item()
        probs = torch.nn.functional.softmax(scores[0], dim=-1)
        token_prob = probs[token_id].item()
        token_probs.append((tokenizer.decode([token_id]), token_prob))
    
    # Visualize token probabilities
    fig, ax = plt.subplots(figsize=(12, len(token_probs) * 0.4 + 1))
    
    tokens = [t[0] for t in token_probs]
    probs = [t[1] for t in token_probs]
    
    # Create colormap based on probabilities
    colors = [(0.7, 0.7, 1), (0, 0.5, 1)]  # Light blue to blue
    cmap = LinearSegmentedColormap.from_list("prob_cmap", colors, N=100)
    
    # Create colored bars for each token probability
    for i, (token, prob) in enumerate(zip(tokens, probs)):
        color = cmap(prob)
        ax.barh(i, prob, color=color)
        ax.text(prob + 0.01, i, f"{token} ({prob:.2f})", va='center')
    
    ax.set_yticks(range(len(tokens)))
    ax.set_yticklabels([])
    ax.set_xlim(0, 1.1)
    ax.set_xlabel("Token Probability")
    ax.set_title("Token Generation Probabilities")
    
    plt.tight_layout()
    
    # Save the image
    output_path = output_dir / "token_probabilities.png"
    plt.savefig(output_path)
    plt.close()
    
    # Return results including the visualization
    return {
        "generated_text": generated_text,
        "token_probs_viz": str(output_path)
    }

# Define the interactive interface
def create_interface():
    with gr.Blocks(title="LLM Architecture Explained") as interface:
        gr.Markdown("# Understanding LLM Architecture")
        gr.Markdown("""
        This interactive tutorial explains how Large Language Models (LLMs) work through visualizations and examples.
        
        ## 1. Tokenization: Breaking Text into Units
        
        Just like you learned to read by recognizing individual letters and then words, LLMs process text as "tokens."
        Tokens can be words, parts of words, or even individual characters.
        """)
        
        with gr.Tab("Tokenization"):
            with gr.Row():
                with gr.Column(scale=3):
                    tokenize_input = gr.Textbox(
                        label="Enter text to tokenize",
                        placeholder="Type something here...",
                        value="Hello world! How do language models work?"
                    )
                    tokenize_button = gr.Button("Tokenize")
                
                with gr.Column(scale=4):
                    tokenize_output = gr.Json(label="Tokenization Results")
                    token_image = gr.Image(label="Token Visualization", type="filepath")
            
            gr.Markdown("""
            ### How Tokenization Works
            
            1. The model breaks your text into pieces (tokens) based on its vocabulary
            2. Common words often get their own token, while rare words are split into pieces
            3. Each token gets converted to a number (token ID) that the model can process
            4. For GPT models, spaces are often represented by 'Ġ' in the raw tokens
            
            **Try it yourself!** Enter different texts and see how they get tokenized.
            """)
        
        gr.Markdown("""
        ## 2. Attention: How Models Focus on Relevant Information
        
        Attention mechanisms allow the model to "pay attention" to different parts of the input when generating each token.
        This is similar to how you might focus on different parts of a sentence to understand its meaning.
        """)
        
        with gr.Tab("Attention Mechanism"):
            with gr.Row():
                with gr.Column(scale=3):
                    attention_input = gr.Textbox(
                        label="Enter text to visualize attention",
                        placeholder="Short texts work best for visualization",
                        value="The cat sat on the mat."
                    )
                    attention_button = gr.Button("Visualize Attention")
                
                with gr.Column(scale=4):
                    attention_image = gr.Image(label="Attention Visualization", type="filepath")
            
            gr.Markdown("""
            ### Understanding the Attention Map
            
            The image shows how each token (word piece) attends to every other token:
            
            - Brighter colors indicate stronger attention
            - Each row shows what a particular token is paying attention to
            - Diagonal elements are often bright as tokens attend to themselves
            - You can see how tokens like "the" might strongly attend to nouns that follow
            
            **Try it yourself!** Enter different sentences and observe the attention patterns.
            """)
        
        gr.Markdown("""
        ## 3. Text Generation: How Models Produce Responses
        
        LLMs generate text by predicting the next token based on all the tokens that came before it.
        This is like predicting the next word in a sentence based on all the previous words.
        """)
        
        with gr.Tab("Text Generation"):
            with gr.Row():
                with gr.Column(scale=3):
                    generate_input = gr.Textbox(
                        label="Enter a prompt",
                        placeholder="Start with a few words...",
                        value="Once upon a time"
                    )
                    with gr.Row():
                        max_length = gr.Slider(
                            minimum=10, maximum=100, value=50, step=1,
                            label="Max Length"
                        )
                        temperature = gr.Slider(
                            minimum=0.1, maximum=1.5, value=0.7, step=0.1,
                            label="Temperature (creativity)"
                        )
                    generate_button = gr.Button("Generate Text")
                
                with gr.Column(scale=4):
                    generate_output = gr.Textbox(label="Generated Text")
                    prob_image = gr.Image(label="Token Probabilities", type="filepath")
            
            gr.Markdown("""
            ### Understanding the Generation Process
            
            1. The model generates one token at a time
            2. For each token, it assigns probabilities to all possible next tokens
            3. Temperature controls randomness:
               - Low temperature (0.1-0.5): More deterministic, focused responses
               - Medium temperature (0.6-0.9): Balanced creativity
               - High temperature (1.0+): More random, creative responses
            4. The visualization shows the probability of each generated token
            
            **Try it yourself!** Adjust the temperature and see how it affects the output.
            """)
        
        gr.Markdown("""
        ## What We've Learned
        
        - **Tokenization** breaks text into units the model can process
        - **Attention mechanisms** allow the model to focus on relevant parts of the text
        - **Generation parameters** like temperature control the model's creativity and style
        
        These concepts are the building blocks for understanding LLMs and how to effectively work with them.
        
        Next, we'll look at different models and how they compare in terms of capabilities and performance.
        """)
        
        # Connect buttons to functions
        tokenize_button.click(
            fn=visualize_tokenization, 
            inputs=tokenize_input, 
            outputs=[tokenize_output, token_image]
        )
        
        attention_button.click(
            fn=visualize_attention,
            inputs=attention_input,
            outputs=attention_image
        )
        
        generate_button.click(
            fn=generate_text,
            inputs=[generate_input, max_length, temperature],
            outputs=[generate_output, prob_image]
        )
    
    return interface

# Main execution
if __name__ == "__main__":
    print("Starting LLM Architecture Visualization Interface...")
    print(f"Visualizations will be saved to: {output_dir}")
    
    interface = create_interface()
    interface.launch(share=False)  # Set to True to create a public link
    
    print("\nThank you for exploring LLM architecture!")
    print("Next, try running model_comparison.py to compare different models.") 
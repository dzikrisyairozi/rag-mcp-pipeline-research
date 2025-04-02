#!/usr/bin/env python3
"""
Model Comparison Workshop

This script provides a hands-on workshop for comparing different LLM models,
exploring their strengths, weaknesses, and performance characteristics.

Features:
- Compare responses from different models
- Test different parameter settings
- Measure generation speed and quality
- Explore size vs. performance tradeoffs

All models are run locally using Hugging Face's transformers library.
"""

import os
import sys
import time
import torch
import gradio as gr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

# Create output directory if it doesn't exist
script_path = Path(__file__).resolve()
project_root = script_path.parent.parent.parent.parent
output_dir = project_root / "output"
output_dir.mkdir(exist_ok=True)

# Model definitions - smaller, free models suitable for running locally
MODELS = {
    "DistilGPT2 (small)": {
        "name": "distilgpt2",
        "description": "A smaller, faster version of GPT-2",
        "size": "82M parameters",
        "good_for": "Basic text completion, simple queries",
        "limitations": "Limited context understanding, basic capabilities"
    },
    "GPT2 (medium)": {
        "name": "gpt2",
        "description": "The original GPT-2 small model",
        "size": "124M parameters",
        "good_for": "General text generation, medium complexity tasks",
        "limitations": "May struggle with complex reasoning"
    },
    "GPT2-Medium (large)": {
        "name": "gpt2-medium",
        "description": "A larger version of GPT-2",
        "size": "355M parameters",
        "good_for": "More nuanced text generation, better context understanding",
        "limitations": "Slower, requires more memory"
    }
}

# Load models - lazy loading to save memory
model_cache = {}

def load_model(model_key):
    """Load a model and its tokenizer on demand"""
    if model_key not in model_cache:
        print(f"Loading model: {model_key}")
        model_name = MODELS[model_key]["name"]
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            
            # Store in cache
            model_cache[model_key] = {
                "model": model,
                "tokenizer": tokenizer
            }
            print(f"✅ Successfully loaded {model_key}")
        except Exception as e:
            print(f"❌ Error loading model {model_key}: {e}")
            return None
    
    return model_cache[model_key]

def generate_text(model_key, prompt, max_length, temperature, top_p, repetition_penalty):
    """Generate text using the specified model and parameters"""
    start_time = time.time()
    
    # Load model
    model_data = load_model(model_key)
    if not model_data:
        return {
            "generated_text": f"Error: Failed to load model {model_key}",
            "generation_time": 0,
            "tokens_per_second": 0
        }
    
    model = model_data["model"]
    tokenizer = model_data["tokenizer"]
    
    # Encode prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    input_length = len(input_ids[0])
    
    # Set generation parameters
    gen_kwargs = {
        "max_length": max_length,
        "temperature": temperature,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
        "do_sample": temperature > 0,
        "pad_token_id": tokenizer.eos_token_id
    }
    
    # Generate
    with torch.no_grad():
        output = model.generate(input_ids, **gen_kwargs)
    
    # Decode and measure performance
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    tokens_generated = len(output[0]) - input_length
    generation_time = time.time() - start_time
    tokens_per_second = tokens_generated / generation_time if generation_time > 0 else 0
    
    # Format the output
    result = {
        "generated_text": generated_text,
        "generation_time": f"{generation_time:.2f} seconds",
        "tokens_per_second": f"{tokens_per_second:.2f} tokens/sec"
    }
    
    return result

def compare_models(prompt, max_length=50, temperature=0.7, top_p=0.9, repetition_penalty=1.0):
    """Generate responses from all models for comparison"""
    results = {}
    
    for model_key in MODELS.keys():
        print(f"Generating with {model_key}...")
        result = generate_text(
            model_key, prompt, max_length, temperature, top_p, repetition_penalty
        )
        results[model_key] = result
    
    # Create a comparison table
    comparison_df = pd.DataFrame({
        "Model": list(results.keys()),
        "Size": [MODELS[m]["size"] for m in results.keys()],
        "Generation Time": [results[m]["generation_time"] for m in results.keys()],
        "Speed (tokens/sec)": [results[m]["tokens_per_second"] for m in results.keys()]
    })
    
    # Plot performance comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract numeric values for plotting
    speeds = [float(s.split()[0]) for s in comparison_df["Speed (tokens/sec)"].values]
    sizes = [int(s.split("M")[0]) for s in comparison_df["Size"].values]
    
    # Create scatter plot
    scatter = ax.scatter(sizes, speeds, s=100, alpha=0.7)
    
    # Add labels to each point
    for i, model in enumerate(comparison_df["Model"]):
        ax.annotate(model, (sizes[i], speeds[i]), 
                   xytext=(5, 5), textcoords='offset points')
    
    ax.set_xlabel("Model Size (M parameters)")
    ax.set_ylabel("Speed (tokens/second)")
    ax.set_title("Model Size vs. Generation Speed")
    
    # Save the plot
    plot_path = output_dir / "model_comparison.png"
    plt.savefig(plot_path)
    plt.close()
    
    # Format the results
    formatted_results = {
        "table": comparison_df.to_dict('records'),
        "plot": str(plot_path),
        "responses": {model: results[model]["generated_text"] for model in results}
    }
    
    return formatted_results

def parameter_exploration(model_key, prompt, temperatures, top_ps, max_length=50):
    """Explore how different parameters affect generation"""
    model_data = load_model(model_key)
    if not model_data:
        return {"error": f"Failed to load model {model_key}"}
    
    model = model_data["model"]
    tokenizer = model_data["tokenizer"]
    
    # Encode prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    
    results = []
    
    # Generate with different parameters
    for temp in temperatures:
        for top_p in top_ps:
            # Set generation parameters
            gen_kwargs = {
                "max_length": max_length,
                "temperature": temp,
                "top_p": top_p,
                "do_sample": True,
                "pad_token_id": tokenizer.eos_token_id
            }
            
            # Generate
            with torch.no_grad():
                output = model.generate(input_ids, **gen_kwargs)
            
            # Decode
            generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
            
            # Store results
            results.append({
                "temperature": temp,
                "top_p": top_p,
                "text": generated_text
            })
    
    # Create a visualization
    fig, axes = plt.subplots(len(temperatures), len(top_ps), figsize=(15, 10))
    fig.suptitle(f"Parameter Exploration for {model_key}\nPrompt: {prompt}", fontsize=16)
    
    for i, temp in enumerate(temperatures):
        for j, top_p in enumerate(top_ps):
            idx = i * len(top_ps) + j
            text = results[idx]["text"]
            
            # Truncate text for display
            display_text = text[:100] + "..." if len(text) > 100 else text
            
            if len(temperatures) > 1 and len(top_ps) > 1:
                ax = axes[i, j]
            elif len(temperatures) > 1:
                ax = axes[i]
            elif len(top_ps) > 1:
                ax = axes[j]
            else:
                ax = axes
            
            # Create a text box
            ax.text(0.1, 0.5, display_text, wrap=True, fontsize=9)
            ax.set_title(f"T={temp}, P={top_p}")
            ax.axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for suptitle
    
    # Save the plot
    plot_path = output_dir / "parameter_exploration.png"
    plt.savefig(plot_path)
    plt.close()
    
    return {
        "results": results,
        "plot": str(plot_path)
    }

def create_interface():
    """Create the Gradio interface"""
    with gr.Blocks(title="LLM Model Comparison") as interface:
        gr.Markdown("# LLM Model Comparison Workshop")
        gr.Markdown("""
        This workshop lets you compare different language models and explore how they perform.
        All models run locally on your machine using free, open-source models from Hugging Face.
        """)
        
        with gr.Tab("Model Comparison"):
            gr.Markdown("""
            ## Compare Different Models
            
            See how different models perform on the same prompt. This will help you understand the tradeoffs
            between model size, speed, and quality.
            """)
            
            with gr.Row():
                with gr.Column(scale=3):
                    comparison_prompt = gr.Textbox(
                        label="Enter a prompt to test all models",
                        placeholder="Try something creative...",
                        value="Explain quantum computing to me like I'm five years old."
                    )
                    comparison_max_length = gr.Slider(
                        minimum=10, maximum=200, value=100, step=10,
                        label="Maximum Length"
                    )
                    comparison_temperature = gr.Slider(
                        minimum=0.1, maximum=1.5, value=0.7, step=0.1,
                        label="Temperature"
                    )
                    comparison_button = gr.Button("Compare Models")
                
                with gr.Column(scale=4):
                    comparison_results = gr.JSON(label="Comparison Results")
                    comparison_plot = gr.Image(label="Performance Comparison", type="filepath")
                    comparison_responses = gr.JSON(label="Model Responses")
            
            gr.Markdown("""
            ### Understanding the Results
            
            - **Size**: Number of parameters in the model - larger models are usually more capable but slower
            - **Generation Time**: How long it took to generate the response
            - **Speed**: Tokens generated per second - higher is faster
            
            The plot shows the relationship between model size and generation speed.
            """)
        
        with gr.Tab("Parameter Exploration"):
            gr.Markdown("""
            ## Explore Generation Parameters
            
            See how different parameter settings affect the same model. This helps you understand
            how to control model behavior.
            """)
            
            with gr.Row():
                with gr.Column(scale=3):
                    parameter_model = gr.Dropdown(
                        choices=list(MODELS.keys()), 
                        value=list(MODELS.keys())[0],
                        label="Select Model"
                    )
                    parameter_prompt = gr.Textbox(
                        label="Enter a prompt",
                        placeholder="Enter something open-ended...",
                        value="The future of artificial intelligence will be"
                    )
                    parameter_length = gr.Slider(
                        minimum=20, maximum=150, value=50, step=10,
                        label="Max Length"
                    )
                    parameter_button = gr.Button("Explore Parameters")
                
                with gr.Column(scale=4):
                    parameter_plot = gr.Image(label="Parameter Exploration", type="filepath")
                    parameter_results = gr.JSON(label="Detailed Results", visible=False)
            
            gr.Markdown("""
            ### Key Parameters Explained
            
            - **Temperature**: Controls randomness
              - Lower values (0.1-0.5): More deterministic, repetitive responses
              - Medium values (0.6-0.9): Balanced creativity 
              - Higher values (1.0+): More random, varied, and potentially incoherent
            
            - **Top-p (nucleus sampling)**: Controls diversity
              - Lower values: Model considers fewer options for each token
              - Higher values: Model considers more options for each token
              
            Explore how different combinations affect the quality and creativity of generations.
            """)
        
        with gr.Tab("Model Information"):
            model_info = [[k, v["size"], v["description"], v["good_for"], v["limitations"]] 
                         for k, v in MODELS.items()]
            
            gr.DataFrame(
                headers=["Model", "Size", "Description", "Good For", "Limitations"],
                datatype=["str", "str", "str", "str", "str"],
                value=model_info
            )
            
            gr.Markdown("""
            ### Model Selection Tips
            
            When choosing a model for your application, consider:
            
            1. **Task complexity**: More complex tasks need larger models
            2. **Speed requirements**: Time-sensitive applications need faster models
            3. **Memory constraints**: Larger models need more RAM
            4. **Quality needs**: Higher quality generally requires larger models
            
            For most applications, start with the smallest model that meets your needs and scale up if necessary.
            """)
        
        # Connect components to functions
        comparison_button.click(
            fn=compare_models,
            inputs=[comparison_prompt, comparison_max_length, comparison_temperature],
            outputs=[comparison_results, comparison_plot, comparison_responses]
        )
        
        parameter_button.click(
            fn=parameter_exploration,
            inputs=[
                parameter_model, 
                parameter_prompt, 
                gr.State([0.3, 0.7, 1.2]),  # Temperatures
                gr.State([0.5, 0.9]),        # Top-p values
                parameter_length
            ],
            outputs=[parameter_results, parameter_plot]
        )
    
    return interface

# Main execution
if __name__ == "__main__":
    print("Starting Model Comparison Workshop...")
    print(f"Visualizations and results will be saved to: {output_dir}")
    print("Models will be loaded on demand to save memory.")
    
    interface = create_interface()
    interface.launch(share=False)  # Set to True to create a public link
    
    print("\nWorkshop completed! Thank you for exploring LLM models.")
    print("Next, check out assistant_tutorial.py to build your first LLM application.") 
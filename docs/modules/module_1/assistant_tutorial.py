#!/usr/bin/env python3
"""
Assistant Tutorial

This script provides a guided tutorial for building your first LLM-based assistant.
It walks you through loading models, creating prompt templates, and handling conversations.

By the end of this tutorial, you'll understand:
- How to structure prompts for LLMs
- Basic assistant architecture
- Response formatting and parsing
- Simple conversation management

No API keys required - using local Hugging Face models.
"""

import os
import sys
import json
import time
import torch
import gradio as gr
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

# Create output directory if it doesn't exist
script_path = Path(__file__).resolve()
project_root = script_path.parent.parent.parent.parent
output_dir = project_root / "output"
output_dir.mkdir(exist_ok=True)

# Initialize model and tokenizer
print("Loading model and tokenizer...")
try:
    model_name = "distilgpt2"  # A smaller GPT-2 version
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    print(f"✅ Successfully loaded {model_name}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("Please check your internet connection or try a different model.")
    sys.exit(1)

# Basic prompt templates
PROMPT_TEMPLATES = {
    "general": """
    You are a helpful assistant. Provide accurate, concise, and useful information.
    
    User question: {question}
    
    Assistant:
    """,
    
    "teacher": """
    You are a patient teacher who explains complex topics in simple terms.
    Provide step-by-step explanations and use analogies when helpful.
    
    Student question: {question}
    
    Teacher:
    """,
    
    "creative": """
    You are a creative writer with a vivid imagination.
    Create engaging, original content based on the prompt.
    
    Writing prompt: {question}
    
    Creative response:
    """,
    
    "expert": """
    You are an expert in {domain} with deep knowledge of the field.
    Provide detailed, technical information when appropriate, but
    explain concepts clearly for someone with basic understanding.
    
    Question about {domain}: {question}
    
    Expert response:
    """
}

# Function to generate responses
def generate_response(prompt, max_length=100, temperature=0.7):
    """Generate a response using the model"""
    try:
        # Tokenize the prompt
        inputs = tokenizer(prompt, return_tensors="pt")
        input_length = len(inputs["input_ids"][0])
        
        # Generate text
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                max_length=input_length + max_length,
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode the generated text
        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated part (not including the prompt)
        response = full_text[len(tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)):]
        
        return response.strip()
    except Exception as e:
        return f"Error generating response: {str(e)}"

# Class to manage conversation history
class ConversationMemory:
    def __init__(self, max_history=5):
        self.history = []
        self.max_history = max_history
    
    def add(self, role, content):
        """Add a message to history"""
        self.history.append({"role": role, "content": content})
        # Keep history within limits
        if len(self.history) > self.max_history * 2:
            self.history = self.history[-self.max_history * 2:]
    
    def get_formatted_history(self):
        """Get history formatted as a string"""
        result = ""
        for msg in self.history:
            result += f"{msg['role']}: {msg['content']}\n\n"
        return result
    
    def clear(self):
        """Clear all history"""
        self.history = []

# Create an assistant that remembers conversation
class Assistant:
    def __init__(self, name="AI Assistant"):
        self.name = name
        self.memory = ConversationMemory()
        self.template = PROMPT_TEMPLATES["general"]
        self.domain = "general knowledge"
        self.max_length = 100
        self.temperature = 0.7
    
    def set_template(self, template_name, domain=None):
        """Set the prompt template to use"""
        if template_name in PROMPT_TEMPLATES:
            self.template = PROMPT_TEMPLATES[template_name]
            if domain and template_name == "expert":
                self.domain = domain
    
    def build_prompt(self, question, use_memory=True):
        """Build the full prompt including history if needed"""
        # Format the template
        if "domain" in self.template and "{domain}" in self.template:
            prompt = self.template.format(domain=self.domain, question=question)
        else:
            prompt = self.template.format(question=question)
        
        # Add history if needed
        if use_memory and self.memory.history:
            history_text = self.memory.get_formatted_history()
            # Add history before the current question
            parts = prompt.split(question)
            if len(parts) == 2:
                prompt = parts[0] + f"\n\nPrevious conversation:\n{history_text}\n\n" + question + parts[1]
        
        return prompt
    
    def respond(self, question, use_memory=True):
        """Generate a response to the question"""
        if not question.strip():
            return "Please ask me a question."
        
        # Build the prompt
        prompt = self.build_prompt(question, use_memory)
        
        # Generate the response
        response = generate_response(prompt, self.max_length, self.temperature)
        
        # Update memory if needed
        if use_memory:
            self.memory.add("User", question)
            self.memory.add(self.name, response)
        
        return response, prompt

# Create the Gradio interface
def create_interface():
    """Create the interface for the assistant tutorial"""
    # Create assistant instance
    assistant = Assistant()
    
    with gr.Blocks(title="Build Your First LLM Assistant") as interface:
        gr.Markdown("# Build Your First LLM Assistant")
        gr.Markdown("""
        This interactive tutorial teaches you how to build an AI assistant using language models.
        Try different prompt templates and settings to see how they affect responses.
        """)
        
        with gr.Tab("Basic Assistant"):
            with gr.Row():
                with gr.Column(scale=1):
                    template_dropdown = gr.Dropdown(
                        choices=list(PROMPT_TEMPLATES.keys()),
                        value="general",
                        label="Prompt Template"
                    )
                    
                    domain_input = gr.Textbox(
                        label="Domain (for Expert template)",
                        placeholder="e.g., physics, history, programming",
                        visible=False
                    )
                    
                    max_length_slider = gr.Slider(
                        minimum=10, maximum=150, value=50, step=5,
                        label="Response Length"
                    )
                    
                    temperature_slider = gr.Slider(
                        minimum=0.1, maximum=1.2, value=0.7, step=0.1,
                        label="Temperature (Creativity)"
                    )
                
                with gr.Column(scale=2):
                    question_input = gr.Textbox(
                        label="Your Question",
                        placeholder="Ask something...",
                        lines=2
                    )
                    
                    submit_button = gr.Button("Submit")
                    
                    with gr.Accordion("Full Prompt (click to view)", open=False):
                        prompt_display = gr.Textbox(
                            label="Complete Prompt",
                            lines=5,
                            interactive=False
                        )
                    
                    response_output = gr.Textbox(
                        label="Assistant Response",
                        lines=5,
                        interactive=False
                    )
        
        with gr.Tab("Conversation Assistant"):
            with gr.Row():
                with gr.Column(scale=1):
                    conv_template_dropdown = gr.Dropdown(
                        choices=list(PROMPT_TEMPLATES.keys()),
                        value="general",
                        label="Prompt Template"
                    )
                    
                    conv_domain_input = gr.Textbox(
                        label="Domain (for Expert template)",
                        placeholder="e.g., physics, history, programming",
                        visible=False
                    )
                    
                    memory_checkbox = gr.Checkbox(
                        label="Use Conversation Memory",
                        value=True
                    )
                    
                    clear_button = gr.Button("Clear Conversation")
                
                with gr.Column(scale=2):
                    chat_history = gr.Textbox(
                        label="Conversation History",
                        placeholder="Your conversation will appear here...",
                        lines=8,
                        interactive=False
                    )
                    
                    conv_question_input = gr.Textbox(
                        label="Your Message",
                        placeholder="Type your message here...",
                        lines=2
                    )
                    
                    conv_submit_button = gr.Button("Send")
        
        with gr.Tab("Tutorial Guide"):
            gr.Markdown("""
            ## Learning Guide
            
            ### Step 1: Understanding Prompt Templates
            Prompts give context to the model about what's expected. They include:
            - **Identity**: Who the assistant is (teacher, expert, etc.)
            - **Instructions**: How the assistant should respond
            - **Context**: Any background information
            - **Question**: The user's actual question
            
            Try different templates in the Basic Assistant tab to see how they change responses!
            
            ### Step 2: Parameter Tuning
            - **Response Length**: Controls how long the generated text will be
            - **Temperature**: Controls randomness and creativity
              - Low (0.1-0.4): More deterministic, focused responses
              - Medium (0.5-0.8): Balanced creativity
              - High (0.9+): More random, creative responses
            
            ### Step 3: Adding Memory
            In the Conversation Assistant tab, try having a multi-turn conversation.
            - With memory enabled, the assistant remembers previous exchanges
            - This creates more coherent conversations
            - Try asking follow-up questions that reference previous responses
            
            ### Next Steps
            Once you're comfortable with this tutorial:
            1. Try the more advanced `build_conversational_assistant.py` script
            2. Experiment with domain specialization in `domain_specialization.py`
            3. Build a practical application that integrates with MCP servers
            """)
        
        # Basic tab functions
        def update_template_visibility(template):
            return gr.update(visible=template == "expert")
        
        def submit_basic_question(template, question, domain, max_length, temperature):
            # Update assistant settings
            assistant.set_template(template, domain)
            assistant.max_length = max_length
            assistant.temperature = temperature
            
            # Get response without using memory
            response, prompt = assistant.respond(question, use_memory=False)
            return response, prompt
        
        # Conversation tab functions
        def update_conversation_template(template):
            return gr.update(visible=template == "expert")
        
        def submit_conversation_message(template, question, domain, use_memory):
            # Update assistant settings
            assistant.set_template(template, domain)
            
            # Get response with memory if enabled
            response, _ = assistant.respond(question, use_memory=use_memory)
            
            # Update display
            history_text = assistant.memory.get_formatted_history() if use_memory else ""
            return history_text
        
        def clear_conversation():
            assistant.memory.clear()
            return ""
        
        # Connect the interface components
        template_dropdown.change(
            fn=update_template_visibility,
            inputs=template_dropdown,
            outputs=domain_input
        )
        
        submit_button.click(
            fn=submit_basic_question,
            inputs=[template_dropdown, question_input, domain_input, max_length_slider, temperature_slider],
            outputs=[response_output, prompt_display]
        )
        
        conv_template_dropdown.change(
            fn=update_conversation_template,
            inputs=conv_template_dropdown,
            outputs=conv_domain_input
        )
        
        conv_submit_button.click(
            fn=submit_conversation_message,
            inputs=[conv_template_dropdown, conv_question_input, conv_domain_input, memory_checkbox],
            outputs=chat_history
        )
        
        clear_button.click(
            fn=clear_conversation,
            inputs=[],
            outputs=chat_history
        )
    
    return interface

# Save example templates to file
def save_example_templates():
    # Add some additional example templates
    extended_templates = PROMPT_TEMPLATES.copy()
    extended_templates.update({
        "code_assistant": """
        You are a skilled programming assistant. Write clean, efficient code with good documentation.
        
        Programming task: {question}
        Preferred language: {language}
        
        Code solution:
        """,
        
        "step_by_step": """
        You are a helpful assistant that explains processes step by step.
        Break down complex tasks into clear, numbered steps.
        
        Task to explain: {question}
        
        Step-by-step guide:
        """
    })
    
    # Save to file
    templates_path = output_dir / "example_templates.json"
    with open(templates_path, 'w') as f:
        json.dump(extended_templates, f, indent=2)
    
    print(f"Saved example templates to: {templates_path}")

# Main execution
if __name__ == "__main__":
    print("Starting Assistant Tutorial...")
    print(f"Outputs will be saved to: {output_dir}")
    
    # Save example templates
    save_example_templates()
    
    # Create and launch the interface
    interface = create_interface()
    interface.launch(share=False)  # Set to True to create a public link
    
    print("\nTutorial completed!")
    print("Next, check out build_conversational_assistant.py for a more advanced assistant.") 
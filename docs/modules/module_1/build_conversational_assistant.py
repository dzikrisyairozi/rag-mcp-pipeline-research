#!/usr/bin/env python3
"""
Build a Conversational Assistant

This script demonstrates how to build a more advanced conversational AI assistant
with improved memory management, persona customization, and specialized capabilities.

Features:
- Advanced conversation memory with summarization
- Persona customization and role-based responses
- Tool integration (calculator, basic weather info)
- Response formatting and filtering
- Performance optimization options

No API keys required - using local Hugging Face models.
"""

import os
import sys
import json
import time
import random
import math
import re
import torch
import gradio as gr
import datetime
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from transformers import AutoModelForCausalLM, AutoTokenizer

# Create output directory if it doesn't exist
script_path = Path(__file__).resolve()
project_root = script_path.parent.parent.parent.parent
output_dir = project_root / "output"
output_dir.mkdir(exist_ok=True)

# Initialize model and tokenizer
print("Loading model and tokenizer...")
try:
    # A medium-sized model that can run on most computers
    model_name = "gpt2"  # 124M parameters
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    print(f"✅ Successfully loaded {model_name}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("Please check your internet connection or try a different model.")
    sys.exit(1)

# ==========================================================
# Advanced Memory Management
# ==========================================================

class MemoryItem:
    """A single piece of conversation memory"""
    def __init__(self, role: str, content: str, timestamp: float = None):
        self.role = role
        self.content = content
        self.timestamp = timestamp or time.time()
    
    def __str__(self):
        return f"{self.role}: {self.content}"
    
    def to_dict(self):
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp
        }

class ConversationMemory:
    """Advanced memory management for conversations"""
    def __init__(
        self, 
        max_items: int = 10, 
        memory_decay: bool = True,
        summarize_threshold: int = 20
    ):
        self.items: List[MemoryItem] = []
        self.max_items = max_items
        self.memory_decay = memory_decay
        self.summarize_threshold = summarize_threshold
        self.summary: Optional[str] = None
    
    def add(self, role: str, content: str) -> None:
        """Add an item to memory"""
        self.items.append(MemoryItem(role, content))
        
        # Apply memory management
        if len(self.items) > self.max_items * 2:
            self._manage_memory()
    
    def _manage_memory(self) -> None:
        """Apply memory management strategies"""
        if len(self.items) <= self.max_items:
            return
            
        if len(self.items) >= self.summarize_threshold:
            # Create summary of older items (simplified, in production you'd use an LLM)
            to_summarize = self.items[:-self.max_items]
            summary_text = "Previous conversation summary: "
            
            # Extract key points (simplified approach)
            user_questions = [item.content for item in to_summarize if item.role == "User"]
            summary_text += f"User asked about {', '.join(user_questions[:3])}... "
            
            # Keep only recent items
            self.items = self.items[-self.max_items:]
            self.summary = summary_text
        elif self.memory_decay:
            # Keep every conversation turn from recent history
            # For older history, keep only important turns (simplistic approach)
            recent = self.items[-self.max_items:]
            older = self.items[:-self.max_items]
            
            # Keep only user questions from older history (simplistic importance filter)
            kept_older = [item for item in older if item.role == "User"]
            
            self.items = kept_older + recent
    
    def get_formatted_history(self, include_summary: bool = True) -> str:
        """Get conversation history formatted as a string"""
        result = ""
        
        # Add summary if available
        if include_summary and self.summary:
            result += f"{self.summary}\n\n"
        
        # Add conversation items
        for item in self.items:
            result += f"{item.role}: {item.content}\n\n"
        
        return result
    
    def clear(self) -> None:
        """Clear all memory"""
        self.items = []
        self.summary = None
    
    def save(self, filename: str = "conversation_history.json") -> str:
        """Save conversation to file"""
        path = output_dir / filename
        data = {
            "items": [item.to_dict() for item in self.items],
            "summary": self.summary
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        
        return str(path)
    
    def load(self, filename: str = "conversation_history.json") -> bool:
        """Load conversation from file"""
        path = output_dir / filename
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            
            self.items = [
                MemoryItem(item["role"], item["content"], item["timestamp"]) 
                for item in data["items"]
            ]
            self.summary = data.get("summary")
            return True
        except Exception as e:
            print(f"Error loading conversation: {e}")
            return False

# ==========================================================
# Assistant Persona System
# ==========================================================

class Persona:
    """Defines the assistant's personality and behavior"""
    def __init__(
        self,
        name: str,
        description: str,
        prompt_template: str,
        greeting: str = None,
        token_limit: int = 150
    ):
        self.name = name
        self.description = description
        self.prompt_template = prompt_template
        self.greeting = greeting or f"Hello, I'm {name}. How can I help you today?"
        self.token_limit = token_limit
    
    @staticmethod
    def load_personas() -> Dict[str, "Persona"]:
        """Load predefined personas"""
        return {
            "general": Persona(
                name="Assistant",
                description="A helpful, balanced AI assistant",
                prompt_template="""
                You are a helpful assistant named {name}. Provide accurate, concise, and useful information.
                
                {memory}
                
                User: {question}
                
                Assistant:
                """
            ),
            "teacher": Persona(
                name="Professor",
                description="A patient, educational AI that explains concepts clearly",
                prompt_template="""
                You are a patient teacher named {name} who explains complex topics in simple terms.
                Provide step-by-step explanations and use analogies when helpful.
                
                {memory}
                
                Student: {question}
                
                Professor:
                """
            ),
            "creative": Persona(
                name="Muse",
                description="A creative AI that generates imaginative content",
                prompt_template="""
                You are a creative writer named {name} with a vivid imagination.
                Create engaging, original content based on the prompt.
                
                {memory}
                
                Prompt: {question}
                
                Muse:
                """,
                token_limit=200
            ),
            "tech_expert": Persona(
                name="TechGuru",
                description="A technical AI expert that provides detailed explanations",
                prompt_template="""
                You are a technical expert named {name} with extensive knowledge in programming, 
                data science, and technology. Provide detailed, accurate information with code
                examples when applicable.
                
                {memory}
                
                Question: {question}
                
                TechGuru:
                """
            ),
            "counselor": Persona(
                name="Advisor",
                description="A supportive AI that offers guidance and reflection",
                prompt_template="""
                You are a supportive advisor named {name} who helps people reflect on their
                challenges and develop practical solutions. You are empathetic but also
                provide structured guidance.
                
                {memory}
                
                Person: {question}
                
                Advisor:
                """
            )
        }

# ==========================================================
# Tools Integration
# ==========================================================

class Tool:
    """A tool that the assistant can use to perform actions"""
    def __init__(self, name: str, description: str, function: Callable):
        self.name = name
        self.description = description
        self.function = function
    
    def execute(self, *args, **kwargs) -> str:
        """Execute the tool function"""
        try:
            return self.function(*args, **kwargs)
        except Exception as e:
            return f"Error using {self.name}: {str(e)}"

def calculator_tool(expression: str) -> str:
    """Simple calculator tool"""
    try:
        # Clean the expression and use safe eval
        clean_expr = re.sub(r'[^0-9+\-*/().%\s]', '', expression)
        result = eval(clean_expr, {"__builtins__": {}}, {"math": math})
        return f"Calculator result: {expression} = {result}"
    except Exception as e:
        return f"Calculator error: Could not calculate '{expression}'. {str(e)}"

def weather_tool(location: str) -> str:
    """Simulated weather tool (no actual API)"""
    # In a real implementation, this would call a weather API
    # For this example, we'll just simulate random weather
    conditions = ["sunny", "cloudy", "rainy", "snowy", "windy", "foggy"]
    temperature = random.randint(0, 35)
    condition = random.choice(conditions)
    
    return f"Weather for {location}: {temperature}°C and {condition}"

def time_tool() -> str:
    """Return the current time"""
    now = datetime.datetime.now()
    return f"Current time: {now.strftime('%H:%M:%S')}"

def current_date_tool() -> str:
    """Return the current date"""
    now = datetime.datetime.now()
    return f"Current date: {now.strftime('%A, %B %d, %Y')}"

# ==========================================================
# Conversational Assistant Class
# ==========================================================

class ConversationalAssistant:
    """Advanced conversational assistant with tools and personas"""
    def __init__(self):
        self.memory = ConversationMemory()
        self.personas = Persona.load_personas()
        self.current_persona = self.personas["general"]
        
        # Initialize tools
        self.tools = {
            "calculator": Tool("Calculator", "Perform calculations", calculator_tool),
            "weather": Tool("Weather", "Get weather information", weather_tool),
            "time": Tool("Time", "Get current time", time_tool),
            "date": Tool("Date", "Get current date", current_date_tool),
        }
        
        # Generation settings
        self.temperature = 0.7
        self.use_tools = True
        self.max_tool_uses = 2
    
    def set_persona(self, persona_key: str) -> bool:
        """Set the assistant's persona"""
        if persona_key in self.personas:
            self.current_persona = self.personas[persona_key]
            return True
        return False
    
    def detect_tool_request(self, question: str) -> Optional[Dict[str, str]]:
        """Detect if a tool should be used for this question"""
        if not self.use_tools:
            return None
            
        question_lower = question.lower()
        
        # Simplistic tool detection (in a real system, use NLU)
        if any(x in question_lower for x in ["calculate", "compute", "what is", "result of"]):
            match = re.search(r'calculate[d]?\s+([0-9+\-*/().%\s]+)', question_lower)
            if match:
                return {"tool": "calculator", "args": match.group(1)}
            
            # Look for expressions like "what is 5+3"
            match = re.search(r'what\s+is\s+([0-9+\-*/().%\s]+)', question_lower)
            if match and any(x in match.group(1) for x in ['+', '-', '*', '/', '%']):
                return {"tool": "calculator", "args": match.group(1)}
        
        if any(x in question_lower for x in ["weather", "temperature", "forecast"]):
            match = re.search(r'weather\s+(?:in|at|for)\s+([a-zA-Z\s,]+)', question_lower)
            if match:
                return {"tool": "weather", "args": match.group(1)}
        
        if any(x in question_lower for x in ["time", "what time"]):
            if "date" not in question_lower:
                return {"tool": "time", "args": ""}
        
        if any(x in question_lower for x in ["date", "today", "what day"]):
            return {"tool": "date", "args": ""}
            
        return None
    
    def generate_response(self, prompt: str) -> str:
        """Generate a response using the model"""
        try:
            # Tokenize the prompt
            inputs = tokenizer(prompt, return_tensors="pt")
            input_length = len(inputs["input_ids"][0])
            
            # Get the token limit from the current persona
            max_new_tokens = self.current_persona.token_limit
            
            # Generate text
            with torch.no_grad():
                outputs = model.generate(
                    inputs["input_ids"],
                    max_length=input_length + max_new_tokens,
                    temperature=self.temperature,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Decode the generated text
            full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the generated part (not including the prompt)
            response = full_text[len(tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)):]
            
            # Remove any trailing "User:" or similar that the model might generate
            cutoff_phrases = ["User:", "Student:", "Person:", "Prompt:"]
            for phrase in cutoff_phrases:
                if phrase in response:
                    response = response.split(phrase)[0]
            
            return response.strip()
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def build_prompt(self, question: str) -> str:
        """Build the full prompt for the model"""
        # Get memory formatted as text
        memory_text = self.memory.get_formatted_history()
        
        # Format the template
        return self.current_persona.prompt_template.format(
            name=self.current_persona.name,
            memory=memory_text,
            question=question
        )
    
    def respond(self, question: str) -> str:
        """Process a question and generate a response"""
        if not question.strip():
            return "Please ask me a question."
        
        # Add user question to memory
        self.memory.add("User", question)
        
        # Check for tool usage
        tool_use_outputs = []
        tool_request = self.detect_tool_request(question)
        
        if tool_request and self.use_tools:
            tool = self.tools.get(tool_request["tool"])
            if tool:
                tool_output = tool.execute(tool_request["args"])
                tool_use_outputs.append(tool_output)
        
        # Build the prompt
        prompt = self.build_prompt(question)
        
        # If using tools, append tool outputs to the prompt
        if tool_use_outputs:
            tool_section = "\n".join(tool_use_outputs)
            prompt += f"\n\nAvailable information:\n{tool_section}\n\n"
        
        # Generate response
        response = self.generate_response(prompt)
        
        # Add response to memory
        self.memory.add(self.current_persona.name, response)
        
        return response

# ==========================================================
# Gradio Interface
# ==========================================================

def create_interface():
    """Create a Gradio interface for the assistant"""
    assistant = ConversationalAssistant()
    
    with gr.Blocks(title="Advanced Conversational Assistant") as interface:
        gr.Markdown("# Advanced Conversational Assistant")
        gr.Markdown("""
        This demo showcases a more advanced AI assistant with persona customization,
        conversation memory, and integrated tools like calculator and weather information.
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                persona_dropdown = gr.Dropdown(
                    choices=list(assistant.personas.keys()),
                    value="general",
                    label="Assistant Persona"
                )
                
                with gr.Accordion("Persona Description", open=False):
                    persona_description = gr.Textbox(
                        value=assistant.current_persona.description,
                        label="Description",
                        interactive=False
                    )
                
                temperature_slider = gr.Slider(
                    minimum=0.1, maximum=1.2, value=0.7, step=0.1,
                    label="Temperature (Creativity)"
                )
                
                use_tools_checkbox = gr.Checkbox(
                    label="Enable Tools",
                    value=True
                )
                
                save_button = gr.Button("Save Conversation")
                load_button = gr.Button("Load Conversation")
                clear_button = gr.Button("Clear Conversation")
            
            with gr.Column(scale=2):
                chat_history = gr.Textbox(
                    label="Conversation",
                    placeholder="Your conversation will appear here...",
                    lines=15,
                    interactive=False
                )
                
                with gr.Row():
                    question_input = gr.Textbox(
                        label="Your Message",
                        placeholder="Type your message here...",
                        lines=2
                    )
                    
                    submit_button = gr.Button("Send")
        
        # Show greeting when persona changes
        def change_persona(persona_key):
            if persona_key in assistant.personas:
                assistant.set_persona(persona_key)
                return assistant.current_persona.greeting, assistant.current_persona.description
            return "", ""
        
        # Submit a question and get response
        def submit_question(question, temperature, use_tools):
            if not question.strip():
                return chat_history.value
            
            # Update settings
            assistant.temperature = temperature
            assistant.use_tools = use_tools
            
            # Get response
            response = assistant.respond(question)
            
            # Update history display
            history_text = assistant.memory.get_formatted_history()
            return history_text
        
        # Save conversation
        def save_conversation():
            path = assistant.memory.save()
            return f"Conversation saved to: {path}"
        
        # Load conversation
        def load_conversation():
            success = assistant.memory.load()
            if success:
                return assistant.memory.get_formatted_history()
            return "Failed to load conversation."
        
        # Clear conversation
        def clear_conversation():
            assistant.memory.clear()
            return ""
        
        # Connect the interface components
        persona_dropdown.change(
            fn=change_persona,
            inputs=persona_dropdown,
            outputs=[chat_history, persona_description]
        )
        
        submit_button.click(
            fn=submit_question,
            inputs=[question_input, temperature_slider, use_tools_checkbox],
            outputs=chat_history
        )
        
        question_input.submit(
            fn=submit_question,
            inputs=[question_input, temperature_slider, use_tools_checkbox],
            outputs=chat_history
        )
        
        save_button.click(
            fn=save_conversation,
            inputs=[],
            outputs=chat_history
        )
        
        load_button.click(
            fn=load_conversation,
            inputs=[],
            outputs=chat_history
        )
        
        clear_button.click(
            fn=clear_conversation,
            inputs=[],
            outputs=chat_history
        )
    
    return interface

# ==========================================================
# Main Execution
# ==========================================================

if __name__ == "__main__":
    print("Starting Advanced Conversational Assistant...")
    print(f"Outputs will be saved to: {output_dir}")
    
    # Create and launch the interface
    interface = create_interface()
    interface.launch(share=False)  # Set to True to create a public link
    
    print("\nAssistant is running!")
    print("Next, check out domain_specialization.py for customizing your assistant for specific domains.") 
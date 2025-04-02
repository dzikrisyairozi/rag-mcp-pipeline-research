#!/usr/bin/env python3
"""
Domain Specialization for LLMs

This script demonstrates how to specialize a general-purpose LLM for specific domains
through techniques like knowledge injection, domain-specific prompting, and retrieval augmentation.

Features:
- Knowledge base creation and management
- Domain-specific prompt templates
- Retrieval-augmented generation (simple version)
- Domain adaptation techniques
- Demo interface for comparing general vs. specialized responses

No API keys required - using local Hugging Face models.
"""

import os
import sys
import json
import time
import torch
import gradio as gr
import numpy as np
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer

# Create output directory if it doesn't exist
script_path = Path(__file__).resolve()
project_root = script_path.parent.parent.parent.parent
output_dir = project_root / "output"
docs_dir = project_root / "docs" / "knowledge_bases"
output_dir.mkdir(exist_ok=True)
docs_dir.mkdir(exist_ok=True)

# Initialize models
print("Loading models...")
try:
    # Main generation model - using a smaller model for practicality
    gen_model_name = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(gen_model_name)
    model = AutoModelForCausalLM.from_pretrained(gen_model_name)
    
    # Embedding model for retrieval (small but effective)
    embed_model_name = "all-MiniLM-L6-v2"
    embed_model = SentenceTransformer(embed_model_name)
    
    print(f"✅ Successfully loaded models")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    print("Please check your internet connection or try different models.")
    sys.exit(1)

# ==========================================================
# Knowledge Base Management
# ==========================================================

class KnowledgeBase:
    """Knowledge base for domain-specific information retrieval"""
    def __init__(self, domain_name: str):
        self.domain_name = domain_name
        self.documents: List[Dict[str, Any]] = []
        self.embeddings = None
        self.domain_dir = docs_dir / domain_name
        self.domain_dir.mkdir(exist_ok=True)
    
    def add_document(self, title: str, content: str, source: str = "") -> int:
        """Add a document to the knowledge base"""
        doc_id = len(self.documents)
        doc = {
            "id": doc_id,
            "title": title,
            "content": content,
            "source": source
        }
        self.documents.append(doc)
        self._save_document(doc)
        return doc_id
    
    def _save_document(self, doc: Dict[str, Any]) -> None:
        """Save document to file"""
        filename = f"{doc['id']:04d}_{re.sub(r'[^\w]', '_', doc['title'])[:30]}.json"
        filepath = self.domain_dir / filename
        with open(filepath, 'w') as f:
            json.dump(doc, f, indent=2)
    
    def load_documents(self) -> None:
        """Load all documents from the domain directory"""
        self.documents = []
        for file in self.domain_dir.glob("*.json"):
            try:
                with open(file, 'r') as f:
                    doc = json.load(f)
                    self.documents.append(doc)
            except Exception as e:
                print(f"Error loading document {file}: {e}")
        
        # Sort by ID
        self.documents.sort(key=lambda x: x["id"])
        print(f"Loaded {len(self.documents)} documents for domain '{self.domain_name}'")
    
    def compute_embeddings(self) -> None:
        """Compute embeddings for all documents"""
        if not self.documents:
            print("No documents to embed.")
            return
        
        try:
            texts = [doc["content"] for doc in self.documents]
            self.embeddings = embed_model.encode(texts)
            print(f"✅ Computed embeddings for {len(texts)} documents")
        except Exception as e:
            print(f"Error computing embeddings: {e}")
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve relevant documents for a query"""
        if not self.documents:
            return []
        
        if self.embeddings is None:
            self.compute_embeddings()
        
        # Compute query embedding
        query_embedding = embed_model.encode([query])[0]
        
        # Compute similarities
        similarities = np.dot(self.embeddings, query_embedding)
        
        # Get top k results
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        return [
            {
                **self.documents[idx],
                "similarity": float(similarities[idx])
            }
            for idx in top_indices
        ]

# Sample knowledge bases for different domains
def create_sample_knowledge_bases() -> Dict[str, KnowledgeBase]:
    """Create and populate sample knowledge bases for demonstration"""
    kb_dict = {}
    
    # Medical Knowledge Base
    medical_kb = KnowledgeBase("medical")
    medical_kb.add_document(
        "Diabetes Overview",
        """Diabetes is a chronic health condition that affects how your body turns food into energy.
        Most of the food you eat is broken down into sugar (glucose) and released into your bloodstream.
        When your blood sugar goes up, it signals your pancreas to release insulin.
        Type 1 diabetes is caused by an autoimmune reaction where the body attacks itself by mistake.
        Type 2 diabetes occurs when your body doesn't use insulin well and can't keep blood sugar at normal levels.""",
        "CDC Health Information"
    )
    
    medical_kb.add_document(
        "Hypertension Management",
        """Hypertension, or high blood pressure, is a condition where the force of blood against artery walls is too high.
        Management strategies include:
        1. Regular physical activity (150 minutes per week)
        2. DASH diet (rich in fruits, vegetables, whole grains, lean protein)
        3. Reducing sodium intake to less than 2,300mg per day
        4. Limiting alcohol consumption
        5. Maintaining a healthy weight
        6. Medication as prescribed by healthcare professionals""",
        "American Heart Association"
    )
    
    medical_kb.add_document(
        "Vaccine Information",
        """Vaccines contain weakened or inactive parts of a particular organism that triggers an immune response.
        They stimulate the body's immune system to recognize the agent as a threat, destroy it, and keep
        a record of it to recognize and destroy any similar microorganisms encountered in the future.
        Vaccines undergo rigorous safety testing and continued monitoring.
        Common vaccines include MMR, DTaP, influenza, HPV, and COVID-19 vaccines.""",
        "WHO Immunization Information"
    )
    
    # Add to dictionary
    kb_dict["medical"] = medical_kb
    
    # Technology Knowledge Base
    tech_kb = KnowledgeBase("technology")
    tech_kb.add_document(
        "Machine Learning Basics",
        """Machine learning is a subset of artificial intelligence that provides systems the ability to 
        learn and improve from experience without being explicitly programmed.
        The learning process begins with observations or data, such as examples, direct experience, or instruction.
        Common types include:
        - Supervised learning: Algorithm trained on labeled data
        - Unsupervised learning: Algorithm finds patterns in unlabeled data
        - Reinforcement learning: Algorithm learns through reward/penalty feedback""",
        "Introduction to Machine Learning"
    )
    
    tech_kb.add_document(
        "Cloud Computing Types",
        """Cloud computing services are broadly divided into three types:
        1. Infrastructure as a Service (IaaS): Provides virtualized computing resources over the internet
        2. Platform as a Service (PaaS): Provides a platform allowing customers to develop, run, and manage applications
        3. Software as a Service (SaaS): Delivers software applications over the internet, on-demand and typically subscription-based
        
        Major providers include AWS, Microsoft Azure, Google Cloud Platform, and IBM Cloud.""",
        "Cloud Computing Handbook"
    )
    
    tech_kb.add_document(
        "Cybersecurity Best Practices",
        """Key cybersecurity best practices include:
        1. Use strong, unique passwords and password managers
        2. Enable multi-factor authentication (MFA)
        3. Keep systems and software updated
        4. Use encrypted connections (HTTPS, VPN)
        5. Regularly backup important data
        6. Be cautious of phishing attempts
        7. Implement principle of least privilege for access control
        8. Use endpoint protection software
        9. Conduct regular security audits
        10. Educate users about security awareness""",
        "NIST Cybersecurity Framework"
    )
    
    # Add to dictionary
    kb_dict["technology"] = tech_kb
    
    # Finance Knowledge Base
    finance_kb = KnowledgeBase("finance")
    finance_kb.add_document(
        "Investment Types",
        """Common investment types include:
        1. Stocks: Ownership shares in a company
        2. Bonds: Debt securities where investors loan money to an entity
        3. Mutual Funds: Pooled funds managed by professionals
        4. ETFs: Exchange-traded funds that track indices, commodities, or baskets of assets
        5. Real Estate: Property investments, either direct or through REITs
        6. Certificates of Deposit: Time deposits with fixed terms and interest rates
        7. Cryptocurrencies: Digital assets using cryptography for security""",
        "Investment Fundamentals Guide"
    )
    
    finance_kb.add_document(
        "Retirement Planning",
        """Effective retirement planning includes:
        1. Start saving early to benefit from compound interest
        2. Contribute to tax-advantaged accounts (401(k), IRA)
        3. Create a diversified portfolio appropriate for your age and risk tolerance
        4. Gradually shift to more conservative investments as retirement approaches
        5. Consider healthcare costs in retirement planning
        6. Plan for multiple income streams in retirement
        7. Regularly review and adjust your retirement plan""",
        "Retirement Planning Association"
    )
    
    finance_kb.add_document(
        "Tax Efficiency Strategies",
        """Tax efficiency strategies for investors include:
        1. Maximize tax-advantaged accounts like 401(k)s and IRAs
        2. Hold tax-efficient investments in taxable accounts
        3. Consider municipal bonds for tax-free income
        4. Use tax-loss harvesting to offset capital gains
        5. Be mindful of holding periods for capital gains treatment
        6. Consider the timing of income and deductions
        7. Donate appreciated assets to charity rather than cash
        8. Implement estate planning strategies to minimize inheritance taxes""",
        "Tax Planning Guide"
    )
    
    # Add to dictionary
    kb_dict["finance"] = finance_kb
    
    return kb_dict

# ==========================================================
# Domain-Specific Prompt Engineering
# ==========================================================

class DomainPromptTemplates:
    """Collection of domain-specific prompt templates"""
    
    @staticmethod
    def get_general_template() -> str:
        """Generic prompt template"""
        return """
        You are a helpful assistant. Provide a clear, accurate response to the question.
        
        Question: {question}
        
        Answer:
        """
    
    @staticmethod
    def get_domain_template(domain: str) -> str:
        """Get domain-specific prompt template"""
        templates = {
            "medical": """
            You are a medical information assistant. Provide accurate, evidence-based health information.
            Remember to:
            - Use precise medical terminology where appropriate
            - Explain complex concepts in plain language
            - Avoid making specific diagnoses
            - Encourage consulting healthcare professionals for personal medical advice
            
            Question: {question}
            
            Relevant information:
            {context}
            
            Medical Information:
            """,
            
            "technology": """
            You are a technology expert assistant. Provide clear, accurate technical information.
            Remember to:
            - Be precise about technical specifications and capabilities
            - Explain technical concepts in an accessible way
            - Include practical examples where helpful
            - Consider both advantages and limitations of technologies
            
            Question: {question}
            
            Relevant information:
            {context}
            
            Technical Response:
            """,
            
            "finance": """
            You are a financial information assistant. Provide clear, accurate financial information.
            Remember to:
            - Use standard financial terminology appropriately
            - Explain financial concepts clearly
            - Avoid making specific investment recommendations
            - Note that information is educational and not personalized advice
            
            Question: {question}
            
            Relevant information:
            {context}
            
            Financial Information:
            """
        }
        
        return templates.get(domain, DomainPromptTemplates.get_general_template())

# ==========================================================
# Domain Specialization Methods
# ==========================================================

class DomainAdapter:
    """Methods for adapting LLMs to specific domains"""
    def __init__(self, domain: str, knowledge_base: KnowledgeBase):
        self.domain = domain
        self.knowledge_base = knowledge_base
        self.prompt_template = DomainPromptTemplates.get_domain_template(domain)
        self.general_template = DomainPromptTemplates.get_general_template()
    
    def retrieve_context(self, query: str, top_k: int = 2) -> str:
        """Retrieve relevant context from knowledge base"""
        retrieved_docs = self.knowledge_base.retrieve(query, top_k)
        
        if not retrieved_docs:
            return "No relevant information found."
        
        context = ""
        for i, doc in enumerate(retrieved_docs, 1):
            context += f"[Document {i}: {doc['title']}]\n"
            context += f"{doc['content']}\n\n"
        
        return context
    
    def generate_specialized_response(self, query: str, temperature: float = 0.7) -> str:
        """Generate a domain-specialized response using RAG"""
        # Retrieve relevant context
        context = self.retrieve_context(query)
        
        # Format prompt with domain template and context
        prompt = self.prompt_template.format(question=query, context=context)
        
        # Generate response
        response = self._generate_text(prompt, temperature)
        
        return response
    
    def generate_general_response(self, query: str, temperature: float = 0.7) -> str:
        """Generate a general (non-specialized) response"""
        # Format prompt with general template
        prompt = self.general_template.format(question=query)
        
        # Generate response
        response = self._generate_text(prompt, temperature)
        
        return response
    
    def _generate_text(self, prompt: str, temperature: float) -> str:
        """Generate text using the model"""
        try:
            # Tokenize the prompt
            inputs = tokenizer(prompt, return_tensors="pt")
            input_length = len(inputs["input_ids"][0])
            
            # Generate text
            with torch.no_grad():
                outputs = model.generate(
                    inputs["input_ids"],
                    max_length=input_length + 150,  # Limit response length
                    temperature=temperature,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Decode the generated text
            full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the generated part (not including the prompt)
            response = full_text[len(tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)):]
            
            # Clean up the response
            response = response.strip()
            
            # Remove any trailing content that might be generating new questions
            cutoff_phrases = ["Question:", "User:", "Human:"]
            for phrase in cutoff_phrases:
                if phrase in response:
                    response = response.split(phrase)[0]
            
            return response.strip()
        except Exception as e:
            return f"Error generating response: {str(e)}"

# ==========================================================
# Gradio Interface
# ==========================================================

def create_interface(knowledge_bases: Dict[str, KnowledgeBase]):
    """Create a Gradio interface for domain specialization demo"""
    domain_adapters = {
        domain: DomainAdapter(domain, kb) 
        for domain, kb in knowledge_bases.items()
    }
    
    with gr.Blocks(title="LLM Domain Specialization") as interface:
        gr.Markdown("# LLM Domain Specialization Demo")
        gr.Markdown("""
        This demo shows how to specialize a general LLM for specific domains 
        using retrieval-augmented generation and domain-specific prompting.
        
        Compare responses from a general model versus a domain-specialized approach.
        """)
        
        with gr.Row():
            domain_dropdown = gr.Dropdown(
                choices=list(domain_adapters.keys()),
                value=list(domain_adapters.keys())[0] if domain_adapters else None,
                label="Select Domain"
            )
            
            temperature_slider = gr.Slider(
                minimum=0.1, maximum=1.2, value=0.7, step=0.1,
                label="Temperature (Creativity)"
            )
        
        query_input = gr.Textbox(
            label="Your Question",
            placeholder="Ask a domain-specific question...",
            lines=2
        )
        
        submit_button = gr.Button("Generate Responses")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### General LLM Response")
                general_response = gr.Textbox(
                    label="Standard Response",
                    lines=10,
                    interactive=False
                )
            
            with gr.Column():
                gr.Markdown("### Domain-Specialized Response")
                specialized_response = gr.Textbox(
                    label="Specialized Response",
                    lines=10,
                    interactive=False
                )
        
        with gr.Accordion("Retrieved Knowledge", open=False):
            retrieved_context = gr.Textbox(
                label="Context Used",
                lines=8,
                interactive=False
            )
        
        # Function to handle domain changes
        def change_domain(domain):
            if domain in knowledge_bases:
                kb = knowledge_bases[domain]
                docs = [f"{doc['title']} - {doc['source']}" for doc in kb.documents]
                return gr.update(value=f"Selected domain: {domain} ({len(docs)} documents)")
            return ""
        
        # Function to generate responses
        def generate_responses(domain, query, temperature):
            if not query.strip() or domain not in domain_adapters:
                return "Please enter a question.", "Please enter a question.", ""
            
            adapter = domain_adapters[domain]
            
            # Get general response
            general = adapter.generate_general_response(query, temperature)
            
            # Get specialized response
            specialized = adapter.generate_specialized_response(query, temperature)
            
            # Get context used
            context = adapter.retrieve_context(query)
            
            return general, specialized, context
        
        # Connect the components
        submit_button.click(
            fn=generate_responses,
            inputs=[domain_dropdown, query_input, temperature_slider],
            outputs=[general_response, specialized_response, retrieved_context]
        )
        
        domain_dropdown.change(
            fn=change_domain,
            inputs=domain_dropdown,
            outputs=retrieved_context
        )
    
    return interface

# ==========================================================
# Main Execution
# ==========================================================

if __name__ == "__main__":
    print("Starting Domain Specialization Demo...")
    print(f"Outputs will be saved to: {output_dir}")
    
    # Create sample knowledge bases
    knowledge_bases = create_sample_knowledge_bases()
    
    # Compute embeddings for each knowledge base
    for domain, kb in knowledge_bases.items():
        print(f"Computing embeddings for {domain} domain...")
        kb.compute_embeddings()
    
    # Create and launch the interface
    interface = create_interface(knowledge_bases)
    interface.launch(share=False)  # Set to True to create a public link
    
    print("\nDomain specialization demo is running!")
    print("Next, explore integrating this specialized assistant with MCP servers.") 
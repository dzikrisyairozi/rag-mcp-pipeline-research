# Module 0: Prerequisites for RAG & MCP Server Integration

## Objective
Establish a solid foundation in AI, LLMs, and MCP server concepts before diving into more advanced topics. This module uses only free and open-source resources to make the learning experience accessible to everyone.

## Why We Use Free Models (Not OpenAI)
This module intentionally avoids using OpenAI's paid API services for the following reasons:

1. **Accessibility**: Anyone can follow this module without financial barriers
2. **Practicality**: Learning the concepts doesn't require commercial-grade models
3. **Transparency**: Open-source models provide better insight into how LLMs work
4. **Flexibility**: You can run everything locally without internet (after initial download)
5. **Skill building**: Working with open-source models builds more transferable skills

## Step-by-Step Setup Guide

### 1. Set Up Python Environment (15-20 minutes)

1. **Install Python 3.8+** if not already installed
   - Download from [python.org](https://www.python.org/downloads/)
   - During installation on Windows, check "Add Python to PATH"
   - Verify installation by opening a terminal/command prompt and typing:
     ```bash
     python --version
     ```

2. **Create a dedicated virtual environment**
   - Open a terminal/command prompt
   - Navigate to your project directory
   - Create the virtual environment:
     ```bash
     # On Windows
     python -m venv venv
     venv\Scripts\activate

     # On macOS/Linux
     python3 -m venv venv
     source venv/bin/activate
     ```
   - You should see `(venv)` at the beginning of your command line

3. **Install required packages**
   - With your virtual environment activated, run:
     ```bash
     pip install transformers torch sentence-transformers fastapi uvicorn jupyter
     ```
   - This might take a few minutes, especially on slower connections

### 2. Run the Example Scripts (30-45 minutes)

1. **LLM Introduction Script**
   - With your virtual environment activated, run:
     ```bash
     # Navigate to the module directory
     cd docs/modules/module_0
     
     # Run the script
     python intro_to_llms.py
     ```
   - **Important**: The first run will download model files (300MB-1GB) and may take 5-10 minutes depending on your internet connection
   - Follow along with the script output to understand each step of LLM interaction
   - **New Feature**: All outputs are saved to an `output/response_[timestamp].txt` file at the project root for easier reading
   - The script will display the location of this file when it runs

2. **Simple MCP Server Example**
   - In a terminal with your virtual environment activated:
     ```bash
     # Navigate to the module directory if not already there
     cd docs/modules/module_0
     
     # Run the server
     python simple_mcp_server_example.py
     ```
   - Open a web browser and go to: `http://127.0.0.1:8000/`
   - Visit `http://127.0.0.1:8000/examples` to see example API requests
   - Visit `http://127.0.0.1:8000/integration` to understand how an AI would use this MCP server
   - Use a tool like [Postman](https://www.postman.com/downloads/) or [curl](https://curl.se/) to make test requests to the server

### 3. Understand the Conceptual Framework (1-2 hours)

1. **Read the Comprehensive RAG-MCP Integration Document**
   - Open and carefully read `rag_mcp_integration.md`
   - Study the diagrams to understand how components work together
   - Take notes on key concepts and their relationships

## Learning Path Breakdown

This section breaks down the recommended learning order and time commitment.

### Week 1: Programming Fundamentals

**Day 1-2: Python Basics** (if needed)
- Complete [Python Official Tutorial](https://docs.python.org/3/tutorial/) or [W3Schools Python Tutorial](https://www.w3schools.com/python/)
- Practice basic data structures and functions
- Ensure you understand how to install packages with pip

**Day 3-4: Version Control & Development Environment**
- Install Git and VSCode
- Learn basic Git commands (clone, add, commit, push)
- Set up VSCode with Python extensions
- Create and activate virtual environments

**Day 5-7: Web APIs & HTTP**
- Learn RESTful API concepts
- Understand HTTP methods and status codes
- Practice making API requests with Python's requests library
- Explore basic FastAPI examples

### Week 2: AI & LLM Concepts

**Day 8-10: Machine Learning Fundamentals**
- Complete a basic ML introduction like [Google's Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course)
- Understand key concepts: models, training/inference, embeddings
- Learn what makes LLMs different from traditional ML

**Day 11-14: LLM Hands-on Experience**
- Work through the provided `intro_to_llms.py` script
- Experiment with different prompts and models
- Try implementing a simple conversation agent
- Read about prompt engineering techniques

### Week 3: RAG & MCP Server Concepts

**Day 15-17: RAG Implementation**
- Understand the RAG architecture in detail
- Explore the code examples in `intro_to_llms.py`
- Try implementing your own knowledge base
- Experiment with different retrieval strategies

**Day 18-21: MCP Server Implementation**
- Run and understand the `simple_mcp_server_example.py`
- Make test requests to the server
- Try adding a new service to the example
- Connect the MCP server to a simple web frontend

## Hands-On Practice Exercises

Complete these exercises to reinforce your learning:

1. **Modify the LLM script**
   - Try different models from Hugging Face (e.g., 'gpt2', 'EleutherAI/gpt-neo-125M')
   - Add the ability to save and load conversation history
   - Implement a simple chat interface with the model

2. **Enhance the MCP server**
   - Add a new service (e.g., "gmail" or "twitter")
   - Implement proper error handling and logging
   - Add request validation and sanitization
   - Create a simple web UI for interacting with the server

3. **Create a simple RAG implementation**
   - Gather a small dataset of text documents
   - Index them using sentence embeddings
   - Create a query interface that retrieves relevant documents
   - Augment LLM prompts with the retrieved information

## Troubleshooting Common Issues

**Problem**: Package installation fails
**Solution**: Try upgrading pip (`pip install --upgrade pip`) and then retry. If using Windows, ensure you have C++ build tools installed.

**Problem**: Models are downloading slowly
**Solution**: Be patient on the first run. Models are cached locally, so subsequent runs will be faster.

**Problem**: Out of memory errors when loading models
**Solution**: Try using smaller models or increasing your system's swap space. For Windows, see [this guide](https://support.microsoft.com/en-us/help/3521/how-to-manage-virtual-memory-paging-file-in-windows-10).

**Problem**: Script execution hangs
**Solution**: Some operations take time, especially on first run. If it's truly stuck for more than 10 minutes, try restarting and using smaller models.

## Next Steps

After completing this module, you'll be ready to move on to:
- **Module 1**: AI Modeling & LLM Integration
- **Module 2**: Hosting & Deployment Strategies for AI

## Resources for Further Learning

- [Hugging Face Documentation](https://huggingface.co/docs)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) (excellent visual explanation)
- [Full Stack Deep Learning Course](https://fullstackdeeplearning.com/) (free comprehensive course)
- [Papers with Code](https://paperswithcode.com/) (implementations of research papers)
- [Awesome RAG GitHub repository](https://github.com/explodinggradients/awesome-rag) (curated resources) 
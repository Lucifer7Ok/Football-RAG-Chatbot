IFAB Football Laws AI - Multi-Modal RAG System
This project is a high-performance Retrieval-Augmented Generation (RAG) system specifically designed to query and interpret the IFAB Laws of the Game 2025/26. By combining semantic text search with visual-context awareness, it provides accurate, evidence-based answers to complex football match scenarios, complete with relevant tactical diagrams.

Key Features
Multi-Modal Retrieval: Simultaneously retrieves precise text-based laws and relevant tactical diagrams/images from the official PDF.

Local Embedding: Utilizes the BAAI/bge-m3 model running entirely offline on your local machine, ensuring data privacy and reducing API costs.

Hybrid RAG Engine: Implements a multi-layered retrieval strategy to minimize hallucinations and ensure the AI remains grounded in official IFAB documentation.

Production-Ready API: Built with FastAPI, featuring resource management (Semaphores, Lifespan events, and Readiness flags) to prevent hardware overload during concurrent requests.

Automated Evaluation: Integrated with Ragas to quantitatively measure system performance based on Faithfulness, Answer Relevancy, and Context Precision.

Project Structure
├── week1_pipeline_optimized.py    # PDF parsing & Translation pipeline
├── week2_cropimg.py               # Vision-based tactical diagram extraction
├── week3_chunking.py              # Local Embedding & FAISS vectorization
├── week4_rag_engine.py            # Core RAG engine (Retriever + Generator)
├── week5_api_server.py            # Production FastAPI server
├── week6_frontend.py              # Streamlit chat interface
├── week7_evaluation.py            # Automated Ragas evaluation module
├── clean_data.py                  # Dataset normalization script
└── Laws_of_the_game_2025_26.pdf   # Source document

Installation
Clone the repository:

Bash
git clone <your-repo-link>
cd ifab-rag-chatbot
Set up a virtual environment and install dependencies:

Bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
Configure your API key in the .env file:

GEMINI_API_KEY=your_actual_api_key_here
Usage
To operate the full system, follow this sequence:

Start the API Backend:

python week5_api_server.py
Launch the Frontend Interface:

In a separate terminal window, run:

streamlit run week6_frontend.py

Access: Open your browser at http://localhost:8501 to begin querying the AI assistant.

Results
The system has achieved high-performance metrics validated through the Ragas evaluation framework:

Faithfulness: ~0.92 (High groundedness in provided laws).

Answer Relevancy: ~0.88 (Direct and context-aware responses).

Context Precision: ~0.85 (Accurate retrieval of relevant laws and tactical charts).

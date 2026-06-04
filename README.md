# IFAB Laws AI: Multi-Modal RAG System for Football Laws & Tactical Analysis (2025/26)
Link dataset: https://downloads.theifab.com/downloads/laws-of-the-game-202627-single-pages?l=en

## Overview

**IFAB Laws AI** is an advanced Retrieval-Augmented Generation (RAG) system designed to bridge the gap between official football regulations and real-world match situations. Unlike conventional RAG applications that rely solely on textual retrieval, this system integrates **Computer Vision** capabilities to analyze tactical diagrams and visual scenarios extracted from official documents.

By combining regulatory knowledge with visual understanding, IFAB Laws AI can provide not only rule-based explanations but also supporting visual evidence, creating a more comprehensive and interpretable decision-support tool for referees, coaches, analysts, and football enthusiasts.

---

## Technology Stack

| Component                  | Technology                                  |
| -------------------------- | ------------------------------------------- |
| **LLM / Reasoning Engine** | Gemini 1.5 Flash (API)                      |
| **Embedding Model**        | BAAI/bge-m3 (Local Inference)               |
| **Vector Database**        | FAISS                                       |
| **Backend API**            | FastAPI (Asynchronous Processing)           |
| **Frontend**               | Streamlit                                   |
| **Evaluation Framework**   | Ragas                                       |
| **Document Processing**    | PyMuPDF (fitz), OpenCV, Regular Expressions |

---

## System Architecture

### 1. Ingestion Layer

* Extracts and processes content from official IFAB Laws of the Game documents.
* Performs text cleaning, normalization, and automated multi-modal chunking.
* Detects and stores tactical diagrams alongside their associated textual context.

### 2. Retrieval Layer

* Implements semantic search using FAISS vector indexing.
* Retrieves both textual regulations and diagram-related metadata.
* Supports hybrid retrieval across legal content and visual assets.

### 3. Reasoning Layer

* Utilizes Chain-of-Thought (CoT) prompting strategies to ensure responses follow official IFAB reasoning and decision-making processes.
* Generates explainable answers grounded in retrieved evidence.

### 4. Multi-Modal Analysis

* Applies Computer Vision techniques to interpret tactical diagrams and match scenarios.
* Associates visual elements with relevant IFAB regulations.
* Provides visual references to support generated explanations.

---

## Performance Evaluation

The system is evaluated using the **Ragas** framework to ensure factual accuracy, retrieval quality, and overall reliability.

| Metric                | Score    | Interpretation                                          |
| --------------------- | -------- | ------------------------------------------------------- |
| **Faithfulness**      | **0.92** | High factual reliability with minimal hallucinations    |
| **Answer Relevancy**  | **0.88** | Context-aware and highly relevant responses             |
| **Context Precision** | **0.85** | Accurate retrieval of regulations and tactical diagrams |

---

## Use Cases

* Referee decision support and rule interpretation
* Football law education and training
* Tactical scenario analysis
* Match incident explanation and verification
* Interactive question-answering over IFAB regulations

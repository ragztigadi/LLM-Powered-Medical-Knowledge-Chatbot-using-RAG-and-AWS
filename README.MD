
# **LLM-Powered Medical Knowledge Chatbot using RAG and AWS**

A production-ready **Retrieval-Augmented Generation (RAG)** medical knowledge chatbot built using **LangChain**, **Pinecone**, and **AWS**.
The system ingests medical PDF documents, generates semantic embeddings, stores them in a vector database, and serves accurate, grounded responses through an interactive chat interface.
The application is fully containerized and deployed on **AWS EC2** using a **GitHub Actions CI/CD pipeline**.

---

## **Table of Contents**

* System Architecture
* Features
* Technology Stack
* Project Structure
* Configuration
* Indexing & Data Flow
* Application Usage
* Results & Validation
* API Endpoints
* Performance Optimization
* Troubleshooting
* License
* Acknowledgments

---

## **System Architecture**

![alt text](<system architectur Advanced-RAG.png>)

The system follows a standard **RAG pipeline** with two primary phases:

---

### **1. Indexing Phase**

![alt text](<01 vector db.png>)

* Medical PDF documents are loaded from the `Data/` directory
* Text is split into overlapping chunks using `RecursiveCharacterTextSplitter`
* Chunks are converted into embeddings using **intfloat/e5-large-v2** (1024-dimensional)
* Embeddings are stored in **Pinecone Serverless (AWS, us-east-1)**

![alt text](<02 emebdeings store into vdb.png>)

---

### **2. Retrieval & Generation Phase**

* User query is embedded using the same embedding model
* Pinecone performs similarity search (Top-K = 3)
* Retrieved chunks are injected into a prompt
* The LLM generates a grounded response based strictly on retrieved context

---

## **Features**

* **Semantic Vector Search** using Pinecone
* **Context-grounded responses** (no hallucinations)
* **Medical-domain document ingestion (PDFs)**
* **Local / containerized inference** (privacy-preserving)
* **Modular, production-ready codebase**
* **Dockerized deployment**
* **Automated CI/CD to AWS EC2 via GitHub Actions**

---

## **Technology Stack**

### **Backend**

* Python
* Flask
* LangChain
* Pinecone Vector Database

### **Machine Learning**

* HuggingFace Embeddings (`intfloat/e5-large-v2`)
* Recursive Character Text Splitter
* PyPDF

### **Cloud & DevOps**

* AWS EC2
* AWS ECR
* Docker
* GitHub Actions (CI/CD)

### **Frontend**

* HTML / CSS
* JavaScript
* Bootstrap

---

## **Project Structure**

```
LLM-Powered-Medical-Knowledge-Chatbot-using-RAG-and-AWS/
│
├── Data/
│   └── Medical reference PDFs
│
├── Documents/
│   ├── system_architectur_Advanced-RAG.png
│   ├── 01_vector_db.png
│   ├── 02_emebdeings_store_into_vdb.png
│   ├── 03_test_input_out_put_1.png
│   ├── 03_test_input_out_put_2.png
│   ├── 04_ecr.png
│   ├── 05_ec2_instance.png
│   └── 06_CICD_deployment.png
│
├── src/
│   ├── helper.py          # PDF loading, chunking, embeddings
│   ├── prompt.py          # Prompt templates
│   └── store_index.py     # Pinecone index creation & ingestion
│
├── templates/
│   └── chat.html
│
├── static/
│   └── style.css
│
├── app.py                 # Flask entry point
├── requirements.txt
├── Dockerfile
├── .env
└── README.md
```

---

## **Configuration**

### **Environment Variables**

```
PINECONE_API_KEY=your_pinecone_api_key
```

---

### **Pinecone Index Configuration**

* **Index Name:** `llm-med-bot`
* **Cloud:** AWS
* **Region:** us-east-1
* **Dimension:** 1024
* **Embedding Model:** `llama-text-embed-v2`
* **Metric:** Cosine similarity
* **Capacity Mode:** Serverless (On-Demand)

---

## **Indexing & Data Flow**

### **Index Creation & Ingestion**

`store_index.py` performs:

* PDF ingestion
* Text chunking
* Embedding generation
* Pinecone index creation (if not exists)
* Vector upsert

✔ **Idempotent design** — rerunning does **not duplicate data**

---

## **Application Usage**

* The Flask application exposes a chat interface on **port 8080**
* Users submit medical queries
* Relevant document chunks are retrieved
* LLM generates responses strictly grounded in indexed knowledge

---

## **Results & Validation**

### **Test Case 1 — Medical Query**

![alt text](<03 test input out put 1.png>)

**Query:**
`what is acne and solution to prevent`

**Result:**
Accurate medical explanation with prevention strategies sourced directly from indexed documents.

---

### **Test Case 2 — Out-of-Scope Query**

![alt text](<03 test input out put 2.png>)
**Query:**
`what is Gen-AI?`

**Result:**
System correctly declined to answer due to missing context, demonstrating **hallucination-free behavior**.

---

## **CI/CD & Deployment**

### **Docker & ECR**

![alt text](<04 ecr.png>)

* Application image built via GitHub Actions
* Image pushed to **Amazon ECR (Private Repository)**

---

### **AWS EC2 Deployment**

![alt text](<05 ec2 instance.png>)

* Container pulled and executed on EC2
* Security Group exposes **port 8080**

---

### **GitHub Actions Pipeline**

![alt text](<06 CICD deploymenet.png>)

Pipeline stages:

* Continuous Integration (Docker build & test)
* Continuous Deployment (Push to ECR → Deploy to EC2)

✔ Pipeline completed successfully
✔ Application running on AWS

---

## **API Endpoints**

### **GET /**

* Serves chat UI

### **POST /get**

```json
{
  "msg": "your medical question"
}
```

Returns chatbot response as plain text.

---

## **Performance Optimization**

* **Top-K Retrieval:** 3
* **Chunk Size:** 500 characters
* **Overlap:** 20 characters
* **Embedding Model:** Optimized for semantic medical text
* **Serverless Vector Store:** Low-latency retrieval

---

## **Troubleshooting**

* **Embedding dimension mismatch:** Ensure index dimension = 1024
* **Empty responses:** Confirm documents are indexed
* **Pinecone auth error:** Verify API key in `.env`
* **LLM not responding:** Confirm container is running and port 8080 is exposed

---

## **License**

This project is licensed under the **MIT License**.

---

## **Acknowledgments**

* **LangChain** — RAG orchestration
* **Pinecone** — Scalable vector database
* **HuggingFace** — High-quality embeddings
* **AWS** — Cloud infrastructure
* **GALE Encyclopedia of Science** — Medical reference corpus

---
# LLM-Powered Medical Knowledge Chatbot using RAG and AWS

A retrieval-augmented generation (RAG) based medical knowledge chatbot built with LangChain, Pinecone vector database, and Ollama LLM. The system processes medical PDF documents, creates semantic embeddings, and provides accurate responses to medical queries through an interactive web interface.

## Table of Contents

- [System Architecture](#system-architecture)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## System Architecture

![System Architecture](Documents/system_architectur_Advanced-RAG.png)

The system implements an advanced RAG pipeline with the following components:

### Indexing Phase

1. **Document Processing**: Medical PDF documents are loaded from the data directory
2. **Text Chunking**: Documents are split into manageable chunks (500 characters with 20-character overlap) using RecursiveCharacterTextSplitter
3. **Embedding Generation**: Text chunks are converted to vector embeddings using HuggingFace's `intfloat/e5-large-v2` model (1024 dimensions)
4. **Vector Storage**: Embeddings are stored in Pinecone vector database for efficient similarity search

### Retrieval and Generation Phase

1. **Query Processing**: User queries are converted to embeddings using the same embedding model
2. **Similarity Search**: The system performs vector similarity search in Pinecone to retrieve the top 3 most relevant document chunks
3. **Context Augmentation**: Retrieved documents are combined with the user query to create enriched context
4. **Response Generation**: The Ollama LLM (Llama 3.2) generates accurate responses based on the retrieved context
5. **User Interface**: Responses are delivered through a Flask-based web application

## Features

- **Semantic Search**: Advanced vector similarity search for accurate document retrieval
- **Contextual Responses**: LLM generates answers grounded in retrieved medical knowledge
- **Scalable Architecture**: Pinecone serverless infrastructure for efficient vector storage
- **Local LLM**: Privacy-focused deployment using Ollama (no external API calls for inference)
- **Interactive UI**: Clean, responsive chat interface for seamless user interaction
- **Modular Design**: Separation of concerns with dedicated modules for helpers, prompts, and indexing

## Technology Stack

### Backend
- **Python 3.8+**
- **Flask**: Web framework for API endpoints and frontend serving
- **LangChain**: Framework for LLM application development
- **Pinecone**: Vector database for embeddings storage and retrieval
- **Ollama**: Local LLM inference (Llama 3.2)

### Machine Learning
- **HuggingFace Embeddings**: `intfloat/e5-large-v2` for text embeddings
- **LangChain Text Splitters**: Document chunking and preprocessing
- **PyPDF**: PDF document parsing

### Frontend
- **HTML5/CSS3**: Responsive chat interface
- **JavaScript/jQuery**: Asynchronous message handling
- **Bootstrap 4**: UI components and styling

## Project Structure

```
LLM-POWERED-MEDICAL-KNOWLEDGE-CHATBOT-USING-RAG-AND-AWS/
│
├── Data/
│   └── The GALE Encyclopedia of Science, 3d edition, Vol2(792s).pdf
│
├── Documents/
│   ├── system_architectur_Advanced-RAG.png
│   ├── 01_vector_db.png
│   ├── 02_emebdeings_store_into_vdb.png
│   ├── 03_test_input_out_put_1.png
│   └── 03_test_input_out_put_2.png
│
├── src/
│   ├── __init__.py
│   ├── helper.py              # Document loading, chunking, and embedding functions
│   ├── prompt.py              # System prompts for LLM
│   └── store_index.py         # Pinecone index creation and management
│
├── static/
│   └── style.css              # Frontend styling
│
├── templates/
│   └── chat.html              # Chat interface
│
├── app.py                      # Flask application entry point
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables
└── README.md                   # Project documentation
```

## Installation

### Prerequisites

- Python 3.8 or higher
- Pinecone account with API key
- Ollama installed locally

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/LLM-Powered-Medical-Knowledge-Chatbot-using-RAG-and-AWS.git
cd LLM-Powered-Medical-Knowledge-Chatbot-using-RAG-and-AWS
```

### Step 2: Create Virtual Environment

```bash
python -m venv llmbot
source llmbot/bin/activate  # On Windows: llmbot\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Install and Configure Ollama

#### Windows
```powershell
winget install Ollama.Ollama
```

#### macOS
```bash
brew install ollama
```

#### Linux
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

Pull the Llama 3.2 model:
```bash
ollama pull llama3.2
```

### Step 5: Set Up Environment Variables

Create a `.env` file in the project root:

```env
PINECONE_API_KEY=your_pinecone_api_key_here
```

## Configuration

### Pinecone Setup

![Pinecone Vector Database](Documents/01_vector_db.png)

The system uses Pinecone serverless infrastructure for vector storage:

- **Index Name**: `llm-med-bot`
- **Cloud Provider**: AWS
- **Region**: `us-east-1`
- **Dimension**: 1024 (matches `intfloat/e5-large-v2` embedding size)
- **Metric**: Cosine similarity
- **Capacity Mode**: On-demand serverless

### Embedding Model Configuration

The system uses HuggingFace's `intfloat/e5-large-v2` model:
- **Dimension**: 1024
- **Maximum Sequence Length**: 512 tokens
- **Performance**: High quality semantic embeddings for medical text

### LLM Configuration

Ollama Llama 3.2 parameters:
- **Temperature**: 0.4 (balanced between creativity and consistency)
- **Model**: `llama3.2`
- **Deployment**: Local inference (no API costs)

## Usage

### Initialize Vector Database

Before running the application for the first time, create and populate the Pinecone index:

```bash
python src/store_index.py
```

![Embeddings Storage](Documents/02_emebdeings_store_into_vdb.png)

This process:
1. Loads PDF documents from the `Data/` directory
2. Splits text into chunks
3. Generates embeddings using HuggingFace model
4. Stores embeddings in Pinecone vector database
5. Creates a searchable index with 2,000 records

The index creation is idempotent - if the index already exists, the system will connect to it without re-embedding documents.

### Run the Application

Start the Flask server:

```bash
python app.py
```

The application will be available at `http://localhost:8080`

### Interact with the Chatbot

1. Open your web browser and navigate to `http://localhost:8080`
2. Type your medical question in the input field
3. Press Enter or click the send button
4. The chatbot will retrieve relevant information and generate a response

## Results

### Test Case 1: Acne Query

![Test Input/Output 1](Documents/03_test_input_out_put_1.png)

**User Query**: "what is acne and solution to prevent"

**System Response**: 
The chatbot successfully retrieved relevant information from the medical encyclopedia and provided a comprehensive answer explaining that acne is a skin condition characterized by comedones, papules, pustules, and cysts. The response included prevention strategies such as reducing sugar consumption, maintaining good hygiene, using non-comedogenic products, and avoiding picking or popping pimples.

**Performance Metrics**:
- Response time: ~0.4 seconds
- Retrieved documents: 3 relevant chunks
- Answer quality: Accurate and contextually appropriate

### Test Case 2: Out-of-Context Query

![Test Input/Output 2](Documents/03_test_input_out_put_2.png)

**User Query**: "what is Gen-AI?"

**System Response**: 
"I don't know. The provided context doesn't mention job searching or employment. It appears to be related to biology and science topics."

**Analysis**:
The system correctly identified that the query was outside the scope of the medical knowledge base. Rather than hallucinating or providing incorrect information, it honestly acknowledged the limitation and described the nature of its knowledge domain. This demonstrates:
- Proper grounding in source documents
- Absence of hallucination
- Transparent communication of limitations

## API Endpoints

### GET /
- **Description**: Serves the chat interface
- **Response**: HTML chat page

### POST /get
- **Description**: Processes user queries and returns chatbot responses
- **Request Body**: 
  ```json
  {
    "msg": "user question here"
  }
  ```
- **Response**: Plain text answer from the chatbot

## Performance Optimization

### Vector Search Optimization
- **Top-K Retrieval**: Limited to 3 most relevant documents to balance context quality and token usage
- **Cosine Similarity**: Efficient metric for semantic similarity in high-dimensional space

### Chunking Strategy
- **Chunk Size**: 500 characters (optimal balance between context and granularity)
- **Overlap**: 20 characters (preserves context across chunk boundaries)

### Embedding Efficiency
- **Model**: `intfloat/e5-large-v2` (1024 dimensions)
- **Batch Processing**: Documents processed in batches during indexing

## Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'langchain.chains'`
- **Solution**: Ensure all dependencies are installed. The project uses `langchain-core`, `langchain-community`, and `langchain-pinecone` packages.

**Issue**: Ollama connection error
- **Solution**: Verify Ollama service is running: `ollama --version`. Restart the Ollama service if needed.

**Issue**: Pinecone API authentication error
- **Solution**: Check that `PINECONE_API_KEY` is correctly set in the `.env` file.

**Issue**: Embeddings dimension mismatch
- **Solution**: Ensure the Pinecone index dimension (1024) matches the embedding model output dimension.

## Future Enhancements

- **Multi-language Support**: Extend to support medical queries in multiple languages
- **Advanced Retrieval**: Implement hybrid search combining dense and sparse retrieval
- **Citation Generation**: Add source citations to responses for transparency
- **User Authentication**: Implement user accounts and conversation history
- **Cloud Deployment**: Deploy to AWS/GCP for production use
- **Model Fine-tuning**: Fine-tune embedding model on medical domain data
- **Evaluation Framework**: Implement automated testing for response quality

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **LangChain**: For the comprehensive LLM application framework
- **Pinecone**: For scalable vector database infrastructure
- **HuggingFace**: For the high-quality embedding model
- **Ollama**: For enabling local LLM deployment
- **GALE Encyclopedia of Science**: For providing the medical knowledge base


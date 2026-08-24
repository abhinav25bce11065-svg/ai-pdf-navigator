# AI PDF Assistant

An AI-powered PDF Question Answering application built with **Streamlit**, **Google Gemini**, **LangChain**, and **ChromaDB**. Upload any PDF document and ask questions about its contents in natural language.

## Features

* 📄 Upload PDF documents
* 🔍 Extract and process PDF text
* ✂️ Intelligent text chunking
* 🧠 Vector embeddings using Google Gemini
* 📚 Semantic document retrieval with ChromaDB
* 🤖 Question answering using Gemini AI
* 🎨 Simple and interactive Streamlit interface

## Tech Stack

* Streamlit
* LangChain
* LangChain Community
* LangChain Google GenAI
* ChromaDB
* PyPDF
* Google Gemini API

## Project Workflow

1. User uploads a PDF file.
2. The PDF is loaded using `PyPDFLoader`.
3. The document is split into smaller chunks using `RecursiveCharacterTextSplitter`.
4. Gemini Embeddings convert text chunks into vector representations.
5. ChromaDB stores and indexes the vectors.
6. User asks a question.
7. Relevant document chunks are retrieved from ChromaDB.
8. Gemini generates an answer based on the retrieved context.

## Installation

### Clone the Repository

```bash
git clone <repository-url>
cd ai-pdf-assistant
```

### Create a Virtual Environment

```bash
python -m venv venv
```

### Activate the Virtual Environment

**Windows**

```bash
venv\Scripts\activate
```

**Linux/macOS**

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Running the Application

```bash
streamlit run app.py
```

The application will open in your browser automatically.

## Usage

1. Launch the application.
2. Enter your Google Gemini API Key in the sidebar.
3. Upload a PDF document.
4. Wait for the document to be processed and indexed.
5. Ask questions related to the document.
6. Receive AI-generated answers based on the document content.

## Example Questions

* What are the main topics covered in this document?
* Summarize the introduction section.
* Explain the key concepts discussed.
* What are the important formulas mentioned?
* List the major headings in the document.

## Project Structure

```text
ai-pdf-assistant/
│
├── app.py
├── requirements.txt
├── README.md
└── venv/
```

## Requirements

* Python 3.10+
* Google Gemini API Key
* Internet Connection

## Future Enhancements

* Chat history support
* Multi-PDF querying
* PDF summarization mode
* Source citation display
* Downloadable chat transcripts
* Study notes generation
* Quiz generation from uploaded PDFs

## License

This project is intended for educational and learning purposes.

## Author

Developed as an AI-powered PDF Assistant using Streamlit, LangChain, ChromaDB, and Google Gemini.

# RAG App

This repository contains a **Retrieval-Augmented Generation (RAG) App** built using Python and Streamlit. The app allows users to upload a PDF file, process its content, and ask questions about the document using advanced language models.

## Features

- **PDF Upload**: Upload a PDF file for processing.
- **Dynamic Model Selection**: Choose from a list of pre-configured language models.
- **Qdrant Integration**: Store and retrieve document embeddings using Qdrant.
- **Chat Interface**: Ask questions about the uploaded PDF or general queries.
- **Contextual Responses**: Generate answers based on the content of the uploaded PDF.

## Requirements

- Python 3.8 or higher
- Dependencies listed in `requirements.txt`

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-repo/rag-app.git
   cd rag-app
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Start the Qdrant server:
   ```bash
   docker run -p 6333:6333 qdrant/qdrant
   ```

## Usage

1. Run the Streamlit app:

   ```bash
   streamlit run app.py
   ```

2. Open the app in your browser at `http://localhost:8501`.

3. Upload a PDF file and interact with the app:
   - Select a language model from the dropdown.
   - Ask questions about the uploaded PDF or general queries.

## Key Components

- **Language Models**: Uses models like `llama3.2`, `deepseek-r1`, and `qwen2.5` for generating responses.
- **Qdrant**: Manages vector embeddings for efficient document retrieval.
- **PDF Processing**: Extracts text from PDF files using `PyPDF2`.
- **Text Splitting**: Splits large text into smaller chunks using `RecursiveCharacterTextSplitter`.

## Example Workflow

1. Upload a PDF file.
2. The app processes the file and creates a collection in Qdrant.
3. Ask a question about the PDF content.
4. The app retrieves relevant context and generates a response using the selected model.

## Dependencies

- `langchain`
- `streamlit`
- `qdrant-client`
- `PyPDF2`

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [LangChain](https://github.com/hwchase17/langchain)
- [Streamlit](https://streamlit.io/)
- [Qdrant](https://qdrant.tech/)

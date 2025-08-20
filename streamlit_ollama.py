from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_qdrant import QdrantVectorStore as Qdrant
from qdrant_client import QdrantClient
import streamlit as st
from PyPDF2 import PdfReader
from langchain_ollama.llms import OllamaLLM
from langchain.prompts import ChatPromptTemplate

separators = [
    "\n\n\n",
    "\n\n",
    "```",
    "---",
    "####",
    "###",
    "##",
    "#",
    ". ",
    "! ",
    "? ",
    "\n",
    " ",
    "",
]


def main() -> None:
    st.set_page_config(
        page_title="RAG App",
        page_icon=":guardsman:",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.sidebar.title("Welcome to the RAG App")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # options = ["llama3.2:1b", "deepseek-r1:1.5b"]
    options = ["deepseek-r1:14b", "mistral-nemo:latest", "llama3.1:8b"]

    model_select = st.sidebar.selectbox("Choose an option:", options)

    st.write(f"You selected: {model_select}")

    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    model = OllamaLLM(
        model=model_select,
        max_new_tokens=128,
        temperature=0.5,
        streaming=True,
        keep_alive="0",
        reasoning=False,
    )
    client = QdrantClient(url="http://localhost:6333")

    pdf = st.file_uploader(
        "Upload a PDF file", type="pdf", label_visibility="collapsed"
    )

    st.write("#### Chat History")
    if st.session_state.chat_history:
        for idx, (question, model_name, answer) in enumerate(
            st.session_state.chat_history, 1
        ):
            with st.expander(f"Question {idx}: {question}"):
                st.markdown(f"**Model:** {model_name}")
                st.markdown(f"**Answer:** {answer}")
    else:
        st.markdown("No chat history yet.")

    if pdf is not None:
        pdf_reader = PdfReader(pdf)

        if pdf_reader.metadata.title is not None:
            COLLECTION_NAME = "".join(
                "".join(
                    "_".join(pdf_reader.metadata.title.split(" ")).split(":")
                ).split(",")
            )
        else:
            COLLECTION_NAME = "rag_collection"

        raw_metadata = pdf_reader.metadata or {}
        metadata_dict = {k.strip("/").lower(): v for k, v in raw_metadata.items()}

        if client.collection_exists(COLLECTION_NAME):
            st.write("✅ Using existing collection from Qdrant.")
            vector_store = Qdrant(
                client=client,
                collection_name=COLLECTION_NAME,
                embedding=embeddings,
            )
        else:
            st.write("📚 Creating new collection and processing PDF...")
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1024, chunk_overlap=100, separators=separators
            )
            docs = splitter.split_text(text)

            documents = [
                Document(page_content=doc, metadata=metadata_dict) for doc in docs
            ]

            vector_store = Qdrant.from_documents(
                documents=documents,
                embedding=embeddings,
                collection_name=COLLECTION_NAME,
                url="http://localhost:6333",
            )
            st.write("✅ File processed and collection created!")

        retriever = vector_store.as_retriever()

        question = st.chat_input("Ask a question about the PDF file:")
        context = ""
        if question:
            st.markdown("#### " + question)
            relevant_docs = retriever.invoke(question)

            for doc in relevant_docs:
                meta = doc.metadata
                meta_str = "\n".join([f"{k}: {v}" for k, v in meta.items()])
                context += f"Metadata:\n{meta_str}\n\nContent:\n{doc.page_content}\n\n"

            prompt_template = ChatPromptTemplate.from_template(
                "Given the following extracted document context, answer the user's question accurately. If the answer is not found in the context, respond with 'Not enough information.'\n\nContext:\n{context}\n\nQuestion:\n{question}"
            )

            prompt = prompt_template.format(context=context, question=question)
            response = model.stream(prompt)
            st.write("#### " + model.model)

            output_placeholder = st.empty()

            final_output = ""

            for chunk in response:
                final_output += chunk
                output_placeholder.markdown(final_output)  # or .write() for plain text

            st.session_state.chat_history.append((question, model.model, final_output))
        else:
            st.markdown("No question asked yet in the PDF.")

    else:
        question = st.chat_input("Ask a question generally:")
        if question:
            st.markdown("#### " + question)
            prompt_template = ChatPromptTemplate.from_template(
                "Answer the following question in the best possible way:\n\nQuestion:\n {question}"
            )

            prompt = prompt_template.format(question=question)
            response = model.stream(prompt)
            st.write("#### " + model.model)

            output_placeholder = st.empty()

            final_output = ""

            for chunk in response:
                final_output += chunk
                output_placeholder.markdown(final_output)  # or .write() for plain text

            st.session_state.chat_history.append((question, model.model, final_output))
        else:
            st.markdown("No question asked yet generally.")


if __name__ == "__main__":
    main()

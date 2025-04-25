import streamlit as st
import fitz  # PyMuPDF
from markdownify import markdownify as md


def pdf_to_markdown(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    full_md = ""

    for page in doc:
        html = page.get_text("html")
        markdown_text = md(html)
        full_md += markdown_text + "\n\n"

    return full_md


# --- Streamlit UI ---
st.set_page_config(page_title="PDF to Markdown Converter", layout="centered")
st.title("📄 PDF to Markdown Converter")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    st.success("PDF uploaded successfully!")

    with st.spinner("Converting to Markdown..."):
        md_output = pdf_to_markdown(uploaded_file)

    st.subheader("📝 Markdown Output")
    st.code(md_output, language='markdown')

    st.download_button(
        label="💾 Download Markdown",
        data=md_output,
        file_name="converted.md",
        mime="text/markdown"
    )

import streamlit as st
from PyPDF2 import PdfReader
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_text_splitters import CharacterTextSplitter


def get_plain_text_from_pdf(pdf_files):
    text = ""
    for pdf_file in pdf_files:
        pdf_reader = PdfReader(pdf_file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def convert_text_to_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def create_vectorstore(txt_chunks):
    # embeddings = OpenAIEmbeddings()
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    faiss_vector_store = FAISS.from_texts(texts=txt_chunks, embedding=embeddings)
    return faiss_vector_store


# Create the base page
st.set_page_config(page_title="Upload PDFs and chat with their content",
                   page_icon=":books:")
st.header("Upload PDFs and chat with their content :books:")
st.subheader("Type your question below")

# Create sidebar with file upload option
with st.sidebar:
    st.subheader("Data collection center")
    uploaded_pdf_files = st.file_uploader(
        "Upload your PDF files here", accept_multiple_files=True)
    if st.button("Upload"):
        with st.spinner("Reading the files..."):
            # Get plain text from the PDF files
            raw_text = get_plain_text_from_pdf(uploaded_pdf_files)
            # st.write(raw_text)
            # Convert the raw text to chunks
            text_chunks = convert_text_to_chunks(raw_text)
            # Create vector store
            vectorstore = create_vectorstore(text_chunks)
            st.write("Done")
            # create conversation chain
            # st.session_state.conversation = get_conversation_chain(
            #     vectorstore)
st.text_input(label="Chat", placeholder="Start chatting")
st.caption('Response:')

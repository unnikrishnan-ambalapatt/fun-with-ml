import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.llms.huggingface_hub import HuggingFaceHub
from langchain_community.vectorstores.faiss import FAISS
from langchain_text_splitters import CharacterTextSplitter

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://cdn.pixabay.com/photo/2023/05/08/08/41/ai-7977960_1280.jpg" style="max-height: 78px; max-width: 78px; border-radius: 50%; object-fit: cover;">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://cdn.pixabay.com/photo/2014/04/03/11/55/robot-312566_1280.png" style="max-height: 78px; max-width: 78px; border-radius: 50%; object-fit: cover;">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''


def get_plain_text_from_pdf(pdf_files):
    text = ""
    for pdf_file in pdf_files:
        pdf_reader = PdfReader(pdf_file)
        # for page in pdf_reader.pages:
        #     text += page.extract_text()
        text += pdf_reader.pages[5].extract_text()
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


def get_conversational_chain(vector_store):
    llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature": 0.5, "max_length": 512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


load_dotenv()
# Create the base page
st.set_page_config(page_title="Upload PDFs and chat with their content",
                   page_icon=":books:")
st.header("Upload PDFs and chat with their content :books:")

st.subheader("Type your question below")
user_question = st.text_input("Ask a question about your documents:")
if user_question:
    handle_userinput(user_question)


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
            st.session_state.conversation = get_conversational_chain(
                vectorstore)
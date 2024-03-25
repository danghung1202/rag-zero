
import os
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

from load_config import getOpenAIKey

OPENAI_API_KEY = getOpenAIKey()
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Sidebar contents
with st.sidebar:
    st.title('LLM Chat App')
    st.markdown('''
    # About
    This app is an LLM-powered chat bot using:
    - [Streamlit](https://streamlit.io/)
                ''')
    
    add_vertical_space(5)
    st.write('Made with love AI')

def create_new_file(file_name, file_content):
    with open(file_name, 'w') as f:
        f.write(file_content)

def create_file_name(pdf_name):
    return str(abs(hash(pdf_name)))

# Load and convert the pdf file to text
def load_and_convert_pdf_to_text(pdf):
    if pdf is not None:
        pdf_reader = PdfReader(pdf)

        text=""
        for page in pdf_reader.pages:
            text += page.extract_text()

        return text
    return None

def split_text_into_chunks(text, chunk_size, chunk_overlap):
    """Because of limitation of context window,
     need to splits text into smaller chunks for processing."""
    text_spliter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap
    )
    return text_spliter.split_text(text)

def create_embeddings(api_key):
    """Creates embeddings from text."""
    return OpenAIEmbeddings(openai_api_key=api_key)

def load_vector_database(store_name, embeddings):
    vectordb = FAISS.load_local(store_name, embeddings,allow_dangerous_deserialization=True)
    return vectordb
 
def save_text_embedding_to_db(chunks, embeddings, store_name):
    """Sets up a vector database for storing embeddings."""
    vectordb = FAISS.from_texts(chunks, embedding=embeddings)
    vectordb.save_local(store_name)
    return vectordb

def create_retriever(vectordb):
    retriever = vectordb.as_retriever()
    return retriever

def create_prompt_template():
    template = """Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Use three sentences maximum and keep the answer as concise as possible.
    Always say "thanks for asking!" at the end of the answer.

    {context}

    Question: {question}

    Helpful Answer:"""
    prompt = ChatPromptTemplate.from_template(template)
    return prompt

def create_model(api_key):
    model = ChatOpenAI(api_key=api_key, temperature=0)
    return model

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def main():
    st.header("Chat with PDF")
    # upload a PDF file
    pdf = st.file_uploader("Upload your PDF", type='pdf')
    if pdf is not None:
        pdf_name = pdf.name[:-4]
        store_name = f"faiss_stores/{create_file_name(pdf_name)}"
        embeddings = create_embeddings(OPENAI_API_KEY)

        if os.path.exists(f"{store_name}/index.pkl"):
            vectordb = load_vector_database(store_name, embeddings)
        else:
            with st.spinner("Loading and processing the pdf..."):
                # Load and convert the pdf file to text
                text = load_and_convert_pdf_to_text(pdf)
                if text is not None:

                    chunks = split_text_into_chunks(text, CHUNK_SIZE, CHUNK_OVERLAP)
                    vectordb = save_text_embedding_to_db(chunks, embeddings, store_name)

                    # create the text file contains the pdf content
                    create_new_file(f"{store_name}/{pdf_name}.txt", f"{text.encode('utf-8')}")
                    st.write(f"The pdf is embedded to {store_name}.pkl successfully")
                else:
                    st.warning("Load and convert pdf to text failed")    

        question = st.text_input("Enter your question about your pdf:")
        

        if (vectordb is not None) and (question.strip()):
            docs = vectordb.similarity_search(question)
            st.write(f"Context: {format_docs(docs)}")

            retriever = create_retriever(vectordb)
            prompt = create_prompt_template()
            model = create_model(OPENAI_API_KEY)
            output_parser = StrOutputParser()
            # create the chain 
            chain =(
                # run paral
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt 
                | model 
                | output_parser
            )
            response = chain.invoke(question)
            # Another explicit way to put the relevant result to context

            # chain = prompt | model | output_parser
            # response = chain.invoke({"question": question, "context": docs})

            st.write(f"Q: {question}")
            st.write(f"A: {response}")

if __name__ == '__main__':
    main()
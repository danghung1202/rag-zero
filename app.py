
import os

import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import langchain_rag as lrag
import file_utils as fu

from load_config import getOpenAIKey

OPENAI_API_KEY = getOpenAIKey()
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Sidebar contents
with st.sidebar:
    openai_api_key = st.text_input(
    "OpenAI API Key", key="chatbot_api_key", type="password")
    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
    "[View the source code](https://github.com/streamlit/llm-examples/blob/main/Chatbot.py)"
    "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/llm-examples?quickstart=1)"

    add_vertical_space(5)
    st.write('Made with love AI')

def main():
    st.header("Chat with PDF")
    # upload a PDF file
    pdf = st.file_uploader("Upload your PDF", type='pdf')
    if pdf is not None:
        pdf_name = pdf.name[:-4]
        store_name = f"faiss_stores/{fu.create_file_name(pdf_name)}"
        embeddings = lrag.create_embeddings(OPENAI_API_KEY)

        if os.path.exists(f"{store_name}/index.pkl"):
            vectordb = lrag.load_vector_database(store_name, embeddings)
        else:
            with st.spinner("Loading and processing the pdf..."):
                # Load and convert the pdf file to text
                text = lrag.load_and_convert_pdf_to_text(pdf)
                if text is not None:

                    chunks = lrag.split_text_into_chunks(text, CHUNK_SIZE, CHUNK_OVERLAP)
                    vectordb = lrag.save_text_embedding_to_db(chunks, embeddings, store_name)

                    # create the text file contains the pdf content
                    fu.create_new_file(f"{store_name}/{pdf_name}.txt", f"{text.encode('utf-8')}")
                    st.write(f"The pdf is embedded to {store_name}.pkl successfully")
                else:
                    st.warning("Load and convert pdf to text failed")

        question = st.text_input("Enter your question about your pdf:")

        if (vectordb is not None) and (question.strip()):
            docs = lrag.vectordb.similarity_search(question)
            st.write(f"Context: {lrag.format_docs(docs)}")

            retriever = lrag.create_retriever(vectordb)
            prompt = lrag.create_prompt_template()
            model = lrag.create_model(OPENAI_API_KEY)
            output_parser = StrOutputParser()
            # create the chain
            chain = (
                # run RunnableParallel in short
                {"context": retriever | lrag.format_docs, "question": RunnablePassthrough()}
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

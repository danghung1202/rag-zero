
import os

import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import WebBaseLoader
import langchain_rag as lrag
import file_utils as fu
import url_utils as uu
from load_config import getOpenAIKey
from operator import itemgetter
from langchain_core.runnables import RunnableBranch
from langchain_core.runnables import RunnableLambda
from streamlit_chat import message as chat_ui
from langchain_core.messages import HumanMessage, AIMessage

OPENAI_API_KEY = getOpenAIKey()
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


# Sidebar contents
with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
    "[View the source code](https://github.com/streamlit/llm-examples/blob/main/Chatbot.py)"

    add_vertical_space(5)
    st.write('Made with love AI')


def route(input):
    if input.get("chat_history"):
        return None

def convert_multi_urls_to_string(urls):
    url_to_names = []
    for url in urls:
        url_to_names.append(uu.convert_url_to_file_name(url))
    
    urls_to_string = '_'.join(url_to_names)
    return urls_to_string

def create_embedding_from_list_urls(urls):

    urls_name = convert_multi_urls_to_string(urls)
    store_name = f"faiss_stores/{fu.create_file_name(urls_name)}"
    embeddings = lrag.create_embeddings(OPENAI_API_KEY)

    if os.path.exists(f"{store_name}/index.pkl"):
        vectordb = lrag.load_vector_database(store_name, embeddings)
        return vectordb;
    else:
        # Load and convert the pdf file to text
        docs = []
        for url in urls:
            loader = WebBaseLoader(url)
            docs += loader.load()

        if docs is not None:

            chunks = lrag.split_documents_into_chunks(docs, CHUNK_SIZE, CHUNK_OVERLAP)
            vectordb = lrag.save_document_embedding_to_db(chunks, embeddings, store_name)

            # create the text file contains the web content
            
            fu.create_new_file(f"{store_name}/{urls_name}.txt", f"{lrag.format_docs(docs).encode('utf-8')}")
            st.write(f"The website content is embedded to {store_name}.pkl successfully")
            return vectordb;
        else:
            st.warning("Load and convert website content to text failed")

def reset_chat_history():
    if 'history' in st.session_state:
        st.session_state['history'] = []

    if 'past' in st.session_state:
        st.session_state['past'] = ["Hey ! 👋"]

    if 'generated' in st.session_state:
        st.session_state['generated'] = ["Hello ! Ask me about " + " 🤗"]

def main():
    if 'vectordb' not in st.session_state:
        st.session_state['vectordb'] = None

    st.header("Chat with your website")
    with st.form("input_urls_form"):
        text = st.text_area("Enter text:", "")
        submitted = st.form_submit_button("Embedding")
        if submitted:
            urls = text.split("\n")
            st.session_state['vectordb'] = create_embedding_from_list_urls(urls)
            reset_chat_history()

    vectordb = st.session_state['vectordb']
    if (vectordb is not None):
        # docs = vectordb.similarity_search(question)
        # st.write(f"Context: {lrag.format_docs(docs)}")
        retriever = lrag.create_retriever(vectordb)
        model = lrag.create_model(OPENAI_API_KEY)

        reformulate_prompt = lrag.create_contextualize_q_prompt()
        qa_prompt = lrag.create_qa_prompt()
        output_parser = StrOutputParser()

        # if the input contains the history chat -> need to reformulate the last question
        reformulate_chain = (
            {
                "question": itemgetter("question"),
                "chat_history": itemgetter("chat_history")
            }
            | reformulate_prompt
            | model
            | output_parser)
        # reformulate_chain.invoke({"question": "abc"})
        # if the input is just the question -> as simple Q&A
        # qa_chain = (
        #     {
        #         "context": itemgetter("question") | retriever | lrag.format_docs,
        #         "question": itemgetter("question"),
        #         "chat_history": itemgetter("chat_history")
        #     }
        #     | qa_prompt
        #     | model
        #     | output_parser)

        def route(input):
            # check if input dict contains the key 'chat_history' and the value of this key is not empty
            # in case input.get("chat_history") is array, it will also check the len(array) > 0
            if input.get("chat_history"):
                return reformulate_chain
            else:
                return input.get("question")

        branch = RunnableBranch(
            (lambda x: x.get("chat_history"),  reformulate_chain),
            itemgetter("question") | retriever | lrag.format_docs,
        )

        full_chain = (
            {
                "context": RunnablePassthrough() | RunnableLambda(route) | retriever | lrag.format_docs,
                "question": itemgetter("question"),
                "chat_history": itemgetter("chat_history")
            }
            | qa_prompt
            | model
            | output_parser)
        # create the chain

        if 'history' not in st.session_state:
            st.session_state['history'] = []

        if 'past' not in st.session_state:
            st.session_state['past'] = ["Hey ! 👋"]

        if 'generated' not in st.session_state:
            st.session_state['generated'] = ["Hello ! Ask me about " + " 🤗"]

        def convert_tuples_to_chat_format(input):
            (human_message, ai_message) = input
            return (HumanMessage(content=human_message), AIMessage(content=ai_message))

        # Create containers for chat history and user input
        response_container = st.container()
        container = st.container()

        # User input form
        with container:
            with st.form(key='chat_form', clear_on_submit=True):
                user_input = st.text_input("", placeholder="Ask me something 👉 (:", key='input')
                submit_button = st.form_submit_button(label='Send')

            if submit_button and user_input:
                response = full_chain.invoke({"question": user_input, "chat_history": st.session_state['history']})
                st.session_state['history'].extend(convert_tuples_to_chat_format((user_input, response)))
                st.session_state['past'].append(user_input)
                st.session_state['generated'].append(response)

        # Display chat history
        if st.session_state['generated']:
            with response_container:
                for i in range(len(st.session_state['generated'])):
                    chat_ui(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
                    chat_ui(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")


if __name__ == '__main__':
    main()

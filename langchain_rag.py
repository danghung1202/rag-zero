from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_text_splitters import Language

# Constants and API Keys
OPENAI_API_KEY = "your_openai_api_key"  # Replace with your actual API key
GPT_MODEL_NAME = 'gpt-4'

# Load and convert the pdf file to text


def load_and_convert_pdf_to_text(pdf):
    """Load and extract the text from pdf using the PdfReader"""
    if pdf is not None:
        pdf_reader = PdfReader(pdf)

        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        return text
    return None


def split_text_into_chunks(text, chunk_size, chunk_overlap):
    """Because of limitation of context window,
     need to splits text into smaller chunks for processing."""
    text_spliter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return text_spliter.split_text(text)


def split_documents_into_chunks(docs, chunk_size, chunk_overlap):
    """Because of limitation of context window,
     need to splits text into smaller chunks for processing."""
    text_spliter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return text_spliter.split_documents(docs)

def split_code_into_chunks(docs, language, chunk_size, chunk_overlap):
    """Because of limitation of context window,
     need to splits text into smaller chunks for processing."""
    text_spliter = RecursiveCharacterTextSplitter.from_language(
        language,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return text_spliter.split_documents(docs)

def create_embeddings(api_key):
    """Creates embeddings from text."""
    # Set chunk_site = 300 to take over the issue rate limit when the token larger than context window
    # https://github.com/langchain-ai/langchain/discussions/9560
    return OpenAIEmbeddings(openai_api_key=api_key, chunk_size=300)


def load_vector_database(store_name, embeddings):
    vectordb = FAISS.load_local(
        store_name, embeddings, allow_dangerous_deserialization=True)
    return vectordb


def save_text_embedding_to_db(chunks_of_text, embeddings, store_name):
    """Sets up a vector database for storing embeddings."""
    vectordb = FAISS.from_texts(chunks_of_text, embedding=embeddings)
    vectordb.save_local(store_name)
    return vectordb


def save_document_embedding_to_db(chunks_of_documents, embeddings, store_name):
    """Sets up a vector database for storing embeddings."""
    vectordb = FAISS.from_documents(chunks_of_documents, embedding=embeddings)
    vectordb.save_local(store_name)
    return vectordb


def create_retriever(vectordb):
    retriever = vectordb.as_retriever()
    return retriever


def create_prompt_template(template=None):
    if template is None:
        template = """You are an assistant for question-answering tasks. \
        Use the following pieces of retrieved context to answer the question. \
        If you don't know the answer, just say that you don't know. \
        Use three sentences maximum and keep the answer concise.\

        {context}

        Question: {question}

        Helpful Answer:"""

    prompt = ChatPromptTemplate.from_template(template)
    return prompt

def create_qa_prompt():
    qa_system_prompt = """You are an assistant for question-answering tasks. \
    Use the following pieces of retrieved context to answer the question. \
    If you don't know the answer, just say that you don't know. \
    Use three sentences maximum and keep the answer concise.\

    {context}"""
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{question}"),
        ]
    )
    return qa_prompt

# https://python.langchain.com/docs/use_cases/question_answering/chat_history
def create_contextualize_q_prompt():
    '''
    Contextualizing questions: Add a sub-chain that takes the latest user question and reformulates it in the context of the chat history. 
    This is needed in case the latest question references some context from past messages. 
    For example, if a user asks a follow-up question like “Can you elaborate on the second point?”, 
    this cannot be understood without the context of the previous message. 
    Therefore we can’t effectively perform retrieval with a question like this.
    '''
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""


    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{question}"),
        ]
    )
    return contextualize_q_prompt


def create_model(api_key):
    model = ChatOpenAI(api_key=api_key, temperature=0.5)
    return model


def format_docs(docs):
    """Extract the `page_content` field from the `doc` object and join the list the docs to one string"""
    return "\n\n".join(doc.page_content for doc in docs)

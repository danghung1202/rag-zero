from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain

# Constants and API Keys
OPENAI_API_KEY = "your_openai_api_key"  # Replace with your actual API key
GPT_MODEL_NAME = 'gpt-4'



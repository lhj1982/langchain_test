from langchain_core.messages.ai import AIMessage
from langchain_core.messages.human import HumanMessage
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains import create_history_aware_retriever
from langchain.chains import create_retrieval_chain
from langchain_core.documents import Document
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain.prompts.chat import ChatPromptTemplate

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(openai_api_key=api_key)

loader = WebBaseLoader("https://docs.smith.langchain.com/user_guide")

docs = loader.load()

embeddings = OpenAIEmbeddings(show_progress_bar=True)


text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)

vector = FAISS.from_documents(documents, embeddings)

retriever = vector.as_retriever()
prompt_retriever = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
])
retriever_chain = create_history_aware_retriever(
    llm, retriever, prompt_retriever)

# retriever_chain = create_history_aware_retriever(llm, retriever, prompt)


# chat_history = [HumanMessage(
#     content="Can LangSmith help test my LLM applications?"), AIMessage(content="Yes!")]
# response = retriever_chain.invoke({
#     "chat_history": chat_history,
#     "input": "Tell me how"
# })

prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Answer the user's questions based on the below context:\n\n{context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
])
document_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)


chat_history = [HumanMessage(
    content="Can LangSmith help test my LLM applications?"), AIMessage(content="Yes!")]
response = retrieval_chain.invoke({
    "chat_history": chat_history,
    "input": "Tell me how"
})

print(response['answer'])
# LangSmith offers several features that can help with testing:...

# Install required packages

import os
import openai
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import OpenAI
import getpass


# Set OpenAI API key
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass()
#Examples
documents = [
    {"text": "LangChain is a framework for developing applications powered by language models."},
    {"text": "It offers a suite of tools to work with language models in various ways."},
    {"text": "One can build applications like chatbots, QA systems, and more using LangChain."},
]

# Initialize the embeddings and vector store
embeddings = OpenAIEmbeddings()
vector_store = FAISS.from_documents(documents, embeddings)

# Set up the conversational retrieval chain
llm = OpenAI(model="text-davinci-003")
qa_chain = ConversationalRetrievalChain(
    llm=llm,
    retriever=vector_store.as_retriever()
)

# Example queries
query1 = "What is LangChain?"
query2 = "What can you build with it?"

# Run the QA system
print(qa_chain({"query": query1}))
print(qa_chain({"query": query2}))

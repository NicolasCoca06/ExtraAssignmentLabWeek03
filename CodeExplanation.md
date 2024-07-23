Harold Nicolás Coca Peña - 202111176

# QA Chat History with LangChain and RAG
FOR THE CODE I MADE MY OWN EXAMPLE AS I WAS DOING THE TUTORIAL. THE PROCESS AND THE FUNCTIONING IS EXACTLY THE SAME AS THE ONE SHOWN IN THE TUTORIAL.


This guide shows hoow to create a QA system withe chat history using LangChain. The system uses Retrieval Augmented Generation (RAG) to give responses by retrieving relevant documents from a knowledge base.

## Code Explanation

### Environment Setup

First, we install the necessary packages and import the required modules:

```python
!pip install langchain openai faiss-cpu

import os
import openai
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import OpenAI
```

### API Key Configuration
After installing the neccesary packages, we set the OpenAI API key to authenticate requiests.
```python
os.environ["API_KEY"] = "API_KEY"
```


### Data instances for preparation
We need some examples for testing purposes, so we create some texts to test the code.
```python
documents = [
    {"text": "LangChain is a framework for developing applications powered by language models."},
    {"text": "It offers a suite of tools to work with language models in various ways."},
    {"text": "One can build applications like chatbots, QA systems, and more using LangChain."},
]
```

### Embeddings and Vector
We create embeddings for our documents and store them in a vector store.
```python
embeddings = OpenAIEmbeddings()
vector_store = FAISS.from_documents(documents, embeddings)
```
### Conversational Retrieval Chain
We set up the QA chain using the OpenAI model and the vector store retriever.
```python
llm = OpenAI(model="text-davinci-003")
qa_chain = ConversationalRetrievalChain(
    llm=llm,
    retriever=vector_store.as_retriever()
)
```

### Running the QA System
The last step is to run the QA system with some example queries.

```python
query1 = "What is LangChain?"
query2 = "What can you build with it?"

# Run the QA system
print(qa_chain({"query": query1}))
print(qa_chain({"query": query2}))

```

## Explanation of how the code works:

Chat Model and Embeddings:
The OpenAI chat model and OpenAIEmbeddings are used to create embeddings for the documents. Embeddings are numerical representations of the documents that capture their semantic meaning.

Vector Store (FAISS):
FAISS (Facebook AI Similarity Search) is used to store these embeddings and perform efficient similarity searches. When a question is asked, the system converts the question into an embedding and searches for similar document embeddings in the vector store.

Retriever:
The VectorStoreRetriever uses the vector store to find relevant documents based on the question embedding. This helps in narrowing down the documents that might contain the answer.

Chat Chain:
The ChatChain ties everything together. It uses the chat model to generate responses and the retriever to find relevant documents based on the chat history. The verbose mode allows you to see the steps taken by the chat chain to generate the response.

## How this relates to RAG (Retrieval augmentes generation?
Retrieval Augmented Generation (RAG) is a technique that enhances language model outputs by retrieving relevant information from a knowledge base. In this example, the QA system uses RAG to retrieve relevant documents from the vector store and incorporate them into the response generation process. This approach improves the accuracy and relevance of the generated responses by grounding them in factual information.






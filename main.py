import langchain
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import (
    Language,
    RecursiveCharacterTextSplitter,
)
from langchain.chains import RetrievalQA
from unstructured.embedding.transformers import CodeGenEmbedding

# Step 1: Load Codebase
print("Step 1")
loader = DirectoryLoader("./../humble_tiers/", glob="**/*.py")
docs = loader.load()

# Step 2: Split Code into Manageable Chunks
print("Step 2")
text_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON, chunk_size=50, chunk_overlap=0
)
documents = text_splitter.split_documents(docs)

# Step 3: Set up Chroma
print("Step 3")
from chromadb import Client

embedding_model = CodeGenEmbedding()
db = Chroma.from_documents(documents, embedding_func=embedding_model)

# Step 4: Create Memory and QA Chain
print("Step 4")
memory = VectorDBQA.from_documents(documents, db)

# Step 5: OLLAMA Integration
print("Step 5")
llm = Ollama(model="mixtral")
qa_chain = RetrievalQA.from_llm_and_memory(llm, memory)


# Step 6: Querying and Response Generation
print("Step 6")


def ask_about_code(question):
    response = qa_chain.run(question)
    return response


# Example Usage
question = "Can you provide an example of a function that calculates the factorial of a number?"
answer = ask_about_code(question)
print(answer)

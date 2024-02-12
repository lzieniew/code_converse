import langchain
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import (
    Language,
    RecursiveCharacterTextSplitter,
)
from langchain.chains import RetrievalQA

# Step 1: Load Codebase
loader = DirectoryLoader("./../humble_tiers/", glob="**/*.py")
docs = loader.load()

# Step 2: Split Code into Manageable Chunks
text_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON, chunk_size=50, chunk_overlap=0
)
documents = text_splitter.split_documents(docs)

# Step 3: Set up Chroma
from chromadb import Client

client = Client()  # In-memory, or use Client(path='/path/to/store') for persistence
vectorstore = Chroma(client)
vectorstore.add_documents(documents)  # Index code in Chroma

# Step 4: Create Memory and QA Chain
memory = VectorDBQA.from_documents(documents, vectorstore)

# Step 5: OLLAMA Integration
llm = Ollama(model="mixtral")
qa_chain = RetrievalQA.from_llm_and_memory(llm, memory)


# Step 6: Querying and Response Generation
def ask_about_code(question):
    response = qa_chain.run(question)
    return response


# Example Usage
question = "Can you provide an example of a function that calculates the factorial of a number?"
answer = ask_about_code(question)
print(answer)

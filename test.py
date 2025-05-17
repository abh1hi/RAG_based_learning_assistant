# Cell 1: Import Dependencies
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma  # Changed this line
from langchain import hub
from langgraph.graph import StateGraph, START
from typing_extensions import TypedDict, List
from IPython.display import Image, display
from langchain.schema import Document
import os

# Cell 2: Configuration
class Config:
    PDF_PATH = "jesc101.pdf"
    MODEL_NAME = "deepseek-r1:1.5b"
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    PERSIST_DIR = "chroma_db"
    COLLECTION_NAME = "jeff_docs"

    # Cell 3: Load and Process Document
def load_document(pdf_path):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    print(f"Loaded {len(docs)} pages")
    return docs

docs = load_document(Config.PDF_PATH)

# Cell 4: Text Splitting
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=Config.CHUNK_SIZE, 
    chunk_overlap=Config.CHUNK_OVERLAP
)
all_splits = text_splitter.split_documents(docs)


# New Cell: Vector Store Utilities
def check_vector_store_details():
    print(f"Vector store location: {os.path.abspath(Config.PERSIST_DIR)}")
    try:
        # Get collection stats
        collection = vector_store._collection
        count = collection.count()
        print(f"Number of documents: {count}")
        
        # Check if embeddings exist
        if count > 0:
            print("\nFirst few document IDs:")
            for doc_id in list(collection._collection.get())[:3]:
                print(f"- {doc_id}")
        else:
            print("\nNo documents found. Recreating vector store...")
            # Recreate vector store
            vector_store = Chroma.from_documents(
                documents=all_splits,
                embedding=embeddings,
                persist_directory=Config.PERSIST_DIR,
                collection_name=Config.COLLECTION_NAME
            )
            vector_store.persist()
            print(f"Vector store recreated with {len(all_splits)} documents")
            
    except Exception as e:
        print(f"Error checking vector store: {str(e)}")

# Cell 5: Vector Store Setup
embeddings = OllamaEmbeddings(model=Config.MODEL_NAME)

try:
    # Try to load existing vector store
    if os.path.exists(Config.PERSIST_DIR):
        print("Loading existing vector store...")
        vector_store = Chroma(
            persist_directory=Config.PERSIST_DIR,
            embedding_function=embeddings,
            collection_name=Config.COLLECTION_NAME
        )
        
        # Verify if store has documents
        if vector_store._collection.count() == 0:
            print("Vector store exists but is empty. Recreating...")
            vector_store = Chroma.from_documents(
                documents=all_splits,
                embedding=embeddings,
                persist_directory=Config.PERSIST_DIR,
                collection_name=Config.COLLECTION_NAME
            )
            vector_store.persist()
    else:
        print("Creating new vector store...")
        vector_store = Chroma.from_documents(
            documents=all_splits,
            embedding=embeddings,
            persist_directory=Config.PERSIST_DIR,
            collection_name=Config.COLLECTION_NAME
        )
        vector_store.persist()
        
except Exception as e:
    print(f"Error setting up vector store: {str(e)}")

# Add this as a new cell
def check_vector_store_details():
    details = {
        "Vector store location": os.path.abspath(Config.PERSIST_DIR),
        "Documents": []
    }
    
    try:
        collection = vector_store._collection
        count = collection.count()
        details["Number of documents"] = count
        
        if count > 0:
            doc_ids = list(collection._collection.get())[:3]
            details["Sample document IDs"] = doc_ids
        
        return details
        
    except Exception as e:
        return {"error": str(e)}

from langchain_core.prompts import PromptTemplate

template = """You are a friendly and knowledgeable science tutor helping 10th-grade students understand science concepts.

- Provide a clear and concise answer to the question.
- Use simple language and examples that a 10th grader can understand.
- If the answer is not in the text, say: "I’m not sure about that from this text."
- If the answer is in the text, provide a direct quote from the text to support your answer.
- Use the following format for your answer:
- Provide a short answer first, followed by a detailed explanation.
- Use bullet points for clarity and organization.
- answer the question using following exaples:
- If the question is "What is the chemical formula for water?" then answer should be like this:
- The chemical formula for water is H2O.
- Water is made up of two hydrogen atoms and one oxygen atom.
- If the question is "help me understand the process of photosynthesis" then answer should be like this:
- Photosynthesis is the process by which green plants use sunlight to convert carbon dioxide and water into glucose and oxygen.
- It occurs in the chloroplasts of plant cells, where chlorophyll captures sunlight.
- The overall equation for photosynthesis is: 6CO2 + 6H2O + light energy -> C6H12O6 + 6O2.
- If the question is "What is the role of chlorophyll in photosynthesis?" then answer should be like this:
- Chlorophyll is a green pigment found in the chloroplasts of plants.
- It plays a crucial role in photosynthesis by absorbing light energy, which is used to convert carbon dioxide and water into glucose and oxygen.
- If the question is "help me with activity 4.1" then answer should be like this:
- Activity 4.1 involves burning a magnesium ribbon in air to observe the reaction and the formation of magnesium oxide.
- The magnesium ribbon should be cleaned before burning to remove any oxide layer that may prevent it from burning properly.
- The burning of magnesium produces a bright white flame and white ash of magnesium oxide.
- If the question is "What is the significance of the activity?" then answer should be like this:
- The significance of the activity is to demonstrate the reaction of magnesium with oxygen in the air, leading to the formation of magnesium oxide.
- This helps students understand the concept of combustion and the properties of metals when they react with oxygen.
- If the question is "What is the chemical equation for the reaction of magnesium with oxygen?" then answer should be like this:
- The chemical equation for the reaction of magnesium with oxygen is: 2Mg + O2 -> 2MgO.
- This equation shows that two magnesium atoms react with one oxygen molecule to form two molecules of magnesium oxide.
- If the question is "What are the products of the reaction between magnesium and oxygen?" then answer should be like this:
- if user asks a question use context to answer it for ex if question is  Why should a magnesium ribbon be cleaned before burning in air?
 then answer should be like this:
- A magnesium ribbon should be cleaned before burning in air to remove any oxide layer that may prevent it from burning properly.
- If question ends with a question mark '?', then use related text and info and make answer.
Question: {question}  
Context: {context}  
Answer:

"""
custom_rag_prompt = PromptTemplate.from_template(template)

# Cell 7: RAG Components Setup
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}

def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    prompt = custom_rag_prompt
    llm = OllamaLLM(model=Config.MODEL_NAME)
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response}

# Cell 8: Graph Construction
def build_graph():
    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    return graph_builder.compile()

graph = build_graph()

# Cell 10: Test RAG System
def query_rag(question: str):
    result = graph.invoke({"question": question})
    print(f"Question: {question}")
    print(f"Answer: {result['answer']}")
    return result["answer"]
    
# Test the RAG system with a question
text = query_rag("What is rancidity?")

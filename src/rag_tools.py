import os
import uuid
from smolagents import Tool, OpenAIServerModel, CodeAgent

# LangChain Imports for RAG components
from langchain_chroma import Chroma
# Import for Hugging Face Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Directory to store ChromaDB persistent data
CHROMA_DB_PATH = "./chroma_knowledge_base"
# Collection name within ChromaDB
CHROMA_COLLECTION_NAME = "specialized_knowledge"

# Initialize the embedding model using a Hugging Face model
# We'll use 'all-MiniLM-L6-v2' as a common, lightweight, and effective embedding model.
# This model is loaded from the Hugging Face Hub.
embeddings_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5", # A popular BGE model
    model_kwargs={'device': 'cpu'} # Use 'cuda' if you have a GPU, otherwise 'cpu'
)

# Initialize the text splitter for chunking documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ".", " ", ""] # Common separators for robust splitting
)

# Initialize ChromaDB persistent client and collection
# This ensures data persists across runs if the path is the same
vectorstore = Chroma(
    persist_directory=CHROMA_DB_PATH,
    embedding_function=embeddings_model,
    collection_name=CHROMA_COLLECTION_NAME
)

# --- Smolagents Tools for RAG ---

class KnowledgeBaseReadTool(Tool):
    """
    A smolagents tool to read/search the specialized knowledge base.
    It performs a similarity search using the provided query and returns
    relevant chunks of information.

    Specialized knowledge for this tool are events in Pokemon GO.
    Each event should have an associated start date and time and an end date and time.
    Each event should have an associated title and description.
    """
    name = "knowledge_base_read"
    description = (
        "Reads and searches the specialized knowledge base for information. "
        "Provide a clear 'query' to find relevant data."
    )
    inputs = {
        "query": {"type": "string", "description": "The natural language query to search the knowledge base."}
    }
    output_type = "string"

    def forward(self, query: str) -> str:
        """
        Performs a similarity search on the ChromaDB knowledge base.

        Args:
            query (str): The search query from the agent.

        Returns:
            str: A formatted string of retrieved documents or a message if nothing is found.
        """
        try:
            # Perform similarity search to get relevant documents
            # k=5 means retrieve the top 5 most similar documents
            retrieved_docs = vectorstore.similarity_search(query, k=5)

            if not retrieved_docs:
                return "No relevant information found in the knowledge base for your query."

            # Format the retrieved documents into a readable string for the agent
            formatted_results = []
            for i, doc in enumerate(retrieved_docs):
                # Include page_content and potentially source from metadata if available
                source_info = doc.metadata.get("source", "N/A")
                doc_id = doc.metadata.get("doc_id", "N/A")
                formatted_results.append(
                    f"--- Document Chunk {i+1} (ID: {doc_id}, Source: {source_info}) ---\n"
                    f"{doc.page_content}\n"
                )
            return "Retrieved knowledge:\n" + "\n".join(formatted_results)

        except Exception as e:
            return f"An error occurred while reading the knowledge base: {e}"


class KnowledgeBaseWriteTool(Tool):
    """
    A smolagents tool to write/update specialized knowledge base data.
    It processes new text content, splits it into chunks, generates embeddings,
    and adds/updates them in the ChromaDB knowledge base.

    Specialized knowledge for this tool are events in Pokemon GO.
    Each event should have an associated start date and time and an end date and time.
    Each event should have an associated title and description.
    """
    name = "knowledge_base_write"
    description = (
        "Writes new information or updates existing information in the specialized knowledge base. "
        "Provide 'text_content' and optionally a 'document_id' for updates. "
        "You can also include 'source' and 'url' for provenance."
    )
    inputs = {
        "text_content": {"type": "string", "description": "The full text content to add or update."},
        "document_id": {"type": "string", "optional": True, "description": "Optional: A unique ID for the document. If provided, existing document with this ID will be updated. If not, a new ID will be generated.", "nullable": True},
        "source": {"type": "string", "optional": True, "description": "Optional: The source of the information (e.g., 'Wikipedia', 'Internal Report').", "nullable": True},
        "url": {"type": "string", "optional": True, "description": "Optional: The URL if the information was mined from the internet.", "nullable": True}
    }
    output_type = "string"

    def forward(self, text_content: str, document_id: str = None, source: str = None, url: str = None) -> str:
        """
        Adds or updates documents in the ChromaDB knowledge base.

        Args:
            text_content (str): The content to be added/updated.
            document_id (str, optional): The ID of the document. If None, a new one is generated.
            source (str, optional): The source of the content.
            url (str, optional): The URL of the content.

        Returns:
            str: A confirmation message.
        """
        try:
            # Generate a document_id if not provided
            if not document_id:
                document_id = str(uuid.uuid4())
                action_type = "added"
            else:
                action_type = "updated"

            # Split the text content into smaller chunks
            chunks = text_splitter.split_text(text_content)

            # Create LangChain Document objects for each chunk
            # Include original document_id and other metadata
            documents_to_add = []
            for i, chunk_text in enumerate(chunks):
                metadata = {
                    "doc_id": document_id,
                    "chunk_idx": i,
                    "source": source,
                    "url": url
                }
                documents_to_add.append(Document(page_content=chunk_text, metadata=metadata))

            # Add/upsert documents to ChromaDB. Chroma's add_documents handles upserting
            # if IDs are provided. Here, we're effectively adding new chunks or replacing
            # based on a new document_id for the original content.
            # For true 'update' of *existing* chunks based on a prior document_id
            # you might need to manage IDs more granularly or delete then add.
            # For simplicity, we'll treat adding content with an existing doc_id as a new version.
            vectorstore.add_documents(documents_to_add)
            vectorstore.persist() # Persist changes to disk

            return (
                f"Successfully {action_type} knowledge base document with ID: {document_id}. "
                f"Total {len(chunks)} chunks processed."
            )

        except Exception as e:
            return f"An error occurred while writing to the knowledge base: {e}"


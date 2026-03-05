# LLAMAINDEX RAG 
# https://docs.llamaindex.ai/en/stable/examples/vector_stores/PineconeIndexDemo/
# https://docs.llamaindex.ai/en/stable/examples/vector_stores/existing_data/pinecone_existing_data/

# PINECONE VECTORSTORE:
# https://docs.pinecone.io/guides/get-started/quickstart 
# https://docs.pinecone.io/guides/data/upsert-data

import logging 
import os
import dotenv
import openai
from pinecone import Pinecone, ServerlessSpec
from llama_index.core import StorageContext, VectorStoreIndex, Document
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from Data.Database_management import retrieve_record_by_name

logging.basicConfig(level=logging.INFO)

# Load environment variables
dotenv.load_dotenv()
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_region = os.getenv("PINECONE_REGION")
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI embeddings and LLM
embedding_model = OpenAIEmbedding(model="text-embedding-3-small", api_key=openai_api_key)
llm = OpenAI(
    model="gpt-4o-mini-2024-07-18",
    api_key=openai_api_key,
    max_new_tokens=500,
    temperature=0.0,
)

openai.api_request_timeout = 30  # Set timeout to 60 seconds

def createOrGetPinecone(index_name: str):
    pinecone = Pinecone(api_key=pinecone_api_key)
    if index_name not in pinecone.list_indexes().names():
        pinecone.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(
                cloud='aws',
                region=pinecone_region
            )
        )
    return pinecone.Index(index_name)

def check_document_exists(index_name: str, file_hash: str, text_type: str) -> bool:
    """
    Check if a document with the given file_hash and text_type already exists in Pinecone.
    
    Returns:
        True if document exists, False otherwise
    """
    try:
        pinecone_index = createOrGetPinecone(index_name)
        vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
        index = VectorStoreIndex.from_vector_store(vector_store)
        
        # Query with metadata filters to check if document exists
        metadata_filters = MetadataFilters(
            filters=[
                ExactMatchFilter(key="file_hash", value=file_hash),
                ExactMatchFilter(key="text_type", value=text_type)
            ]
        )
        
        # Use retrieve to check if any documents match the filters
        retriever = index.as_retriever(similarity_top_k=1, filters=metadata_filters)
        # Use a dummy query to check if any documents match
        results = retriever.retrieve("dummy")
        
        # If we get any results, the document exists
        return len(results) > 0
    except Exception as e:
        # If there's an error (e.g., index doesn't exist yet, or no documents), assume document doesn't exist
        logging.debug(f"Error checking if document exists in Pinecone: {e}")
        return False

def loadDataPinecone(index_name: str, text: str, file_name: str, file_hash: str, text_type: str, skip_if_exists: bool = True):
    """
    Load data into Pinecone.
    
    Args:
        index_name: Name of the Pinecone index
        text: Text content to index
        file_name: Name of the file
        file_hash: Hash of the file
        text_type: Type of text (e.g., 'text_with_pii', 'text_pii_deleted')
        skip_if_exists: If True, skip insertion if document already exists
    """
    if skip_if_exists:
        if check_document_exists(index_name, file_hash, text_type):
            logging.info(f"Document already exists in Pinecone for {file_name} ({file_hash}, {text_type}), skipping insert")
            return
    
    pinecone_index = createOrGetPinecone(index_name)
    document = Document(
        text=text,
        metadata={"file_name": file_name, "file_hash": file_hash, "text_type": text_type}
    )
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(
        documents=[document],
        storage_context=storage_context,
        embed_model=embedding_model,
        show_progress=True, 
        chunk_size=2048
    )

def getResponse(index_name: str, question: str, filters: list) -> str:
    """
    Query Pinecone with (file_hash, text_type) filters, then LLM. If no vectors exist
    for that text_type (e.g. dp_prompt not yet loaded), retrieval returns 0 nodes and
    the LLM often returns 'Empty response'. Load that text_type via load_tab_text2_to_pinecone first.
    """
    pinecone_index = createOrGetPinecone(index_name)
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    index = VectorStoreIndex.from_vector_store(vector_store)
    file_hash, text_type = filters[0], filters[1]
    metadata_filters = MetadataFilters(
        filters=[
            ExactMatchFilter(key="file_hash", value=file_hash),
            ExactMatchFilter(key="text_type", value=text_type)
        ]
    )
    similarity_top_k = 4
    query_engine = index.as_query_engine(llm=llm, streaming=False, similarity_top_k=similarity_top_k, filters=metadata_filters)
    response = query_engine.query(question)

    nodes_text = []
    for i in range( 0, min(similarity_top_k, len(response.source_nodes)) ):
        node_text = response.source_nodes[i].get_text()
        nodes_text.append(node_text)

    # No context in Pinecone for this (file_hash, text_type) → LLM often returns "Empty response"
    if len(response.source_nodes) == 0:
        logging.warning(
            "No documents in Pinecone for file_hash=%s text_type=%s. Load this text_type with load_tab_text2_to_pinecone.",
            file_hash[:16] if file_hash else None, text_type
        )

    evaluation_result = {}
    return (response, nodes_text, evaluation_result)

def load_all(table_name, index_name, file_name_pattern, start, last):
        text_types = ["text_with_pii", "text_pii_deleted", "text_pii_labeled", "text_pii_synthetic", "text_pii_dp_diffractor1", "text_pii_dp_diffractor2", "text_pii_dp_diffractor3", "text_pii_dp_dp_prompt1", "text_pii_dp_dp_prompt2", "text_pii_dp_dp_prompt3", "text_pii_dp_dpmlm1", "text_pii_dp_dpmlm2", "text_pii_dp_dpmlm3"]
        for i in range(start, last+1): # FOR EACH DATABASE FILE
            if table_name == "enron_text2" and i == 61: continue # MISSING TEXT IN ENRON
            file_name = file_name_pattern.format(i)
            database_file = retrieve_record_by_name(table_name, file_name)
            for text_type in text_types:
                loadDataPinecone(
                    index_name=index_name,
                    text=database_file[text_type],
                    file_name=file_name,
                    file_hash=database_file['file_hash'],
                    text_type=text_type
                )
                print(f"Loaded embeddings for: {file_name} {text_type}")

if __name__ == "__main__":
    # load_all(table_name="enron_text2", index_name="enron2", file_name_pattern="Enron_{}", start=1, last=99)
    load_all(table_name="bbc_text2", index_name="bbc2", file_name_pattern="BBC_{}", start=1, last=200)
# LIVE, SAFE, PRIVACY-AWARE RAG
# -> in-memory vector store (SimpleVectorStore) without index persistance
# -> local, not cloud-based embedding model (HuggingFace SentenceTransformers all-MiniLM-L6-v2)

import os
import dotenv
from llama_index.llms.openai import OpenAI
from llama_index.core import VectorStoreIndex, Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from llama_index.embeddings.langchain import LangchainEmbedding

dotenv.load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
llm = OpenAI(
    model="gpt-4o-mini-2024-07-18",
    api_key=openai_api_key,
    max_new_tokens=500,
    temperature=0.0,
)

def get_offline_RAG_response(question, text):

    embed_model = LangchainEmbedding(
        HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    )
    doc = Document(text=text)

    index = VectorStoreIndex.from_documents([doc], embed_model=embed_model)     # in-memory index
    query_engine = index.as_query_engine(llm=llm, streaming=False, similarity_top_k=10)

    response = query_engine.query(question)
    return response.response # for the text only
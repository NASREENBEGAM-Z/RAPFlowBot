import os
import torch
import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
import uuid
import time
import gc

# Load LLM (quantized model for GPU, Mistral-7B is used as default)
@st.cache_resource(show_spinner=False)
def load_llm():
    tokenizer = AutoTokenizer.from_pretrained("TheBloke/Mistral-7B-Instruct-v0.1-GPTQ")
    model = AutoModelForCausalLM.from_pretrained(
        "TheBloke/Mistral-7B-Instruct-v0.1-GPTQ",
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    return pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)

# Load SentenceTransformer for embeddings
@st.cache_resource(show_spinner=False)
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# Chunk PDF documents into small passages
def chunk_pdf(pdf_path, chunk_size=200, overlap=50):
    chunks = []
    reader = PdfReader(pdf_path)
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            words = text.split()
            for i in range(0, len(words), chunk_size - overlap):
                chunk_words = words[i:i + chunk_size]
                chunk_text = ' '.join(chunk_words)
                chunks.append({
                    'content': chunk_text,
                    'metadata': {
                        'filename': os.path.basename(pdf_path),
                        'page_number': page_num + 1,
                        'chunk_id': str(uuid.uuid4())
                    }
                })
    return chunks

# Index documents to Qdrant
def index_documents(embedding_model, qdrant_client, collection_name, chunks):
    if not qdrant_client.collection_exists(collection_name):
        qdrant_client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=embedding_model.get_sentence_embedding_dimension(), distance=Distance.COSINE)
        )
    points = []
    for chunk in chunks:
        vector = embedding_model.encode(chunk['content']).tolist()
        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={
                'content': chunk['content'],
                **chunk['metadata']
            }
        ))
    qdrant_client.upsert(collection_name=collection_name, points=points)

# Query Qdrant for a single best-matching chunk
def search_query(embedding_model, qdrant_client, collection_name, query):
    vector = embedding_model.encode(query).tolist()
    search_result = qdrant_client.search(
        collection_name=collection_name,
        query_vector=vector,
        limit=1
    )
    return search_result[0] if search_result else None

# Format the source reference display
def format_reference(metadata):
    return f"File: {metadata['filename']} | Page: {metadata['page_number']} | Chunk ID: {metadata['chunk_id']}"

# Generate answer with LLM
@torch.inference_mode()
def generate_answer(llm, context, query):
    prompt = f"You are a helpful assistant. Use the below context to answer the question.\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    response = llm(prompt, max_new_tokens=300, do_sample=True, top_p=0.95, temperature=0.7)[0]['generated_text']
    return response.split("Answer:")[-1].strip()

# STREAMLIT UI
def main():
    st.title("🧠 Conversational RAG with Qdrant (Offline)")

    folder_path = st.text_input("Enter the folder path containing PDFs:", value="./docs")
    query = st.text_input("Enter your question:")

    if st.button("Run Query") and query:
        start_time = time.time()

        with st.spinner("Processing..."):
            embedding_model = load_embedding_model()
            llm = load_llm()
            client = QdrantClient(path="./qdrant_storage")

            # Index documents from the given folder
            collection_name = "rag_collection"
            all_chunks = []
            for filename in os.listdir(folder_path):
                if filename.endswith(".pdf"):
                    chunks = chunk_pdf(os.path.join(folder_path, filename))
                    all_chunks.extend(chunks)
            index_documents(embedding_model, client, collection_name, all_chunks)

            # Search and generate response
            result = search_query(embedding_model, client, collection_name, query)
            if result and 'content' in result.payload:
                source_text = result.payload['content']
                metadata = {
                    'filename': result.payload['filename'],
                    'page_number': result.payload['page_number'],
                    'chunk_id': result.payload['chunk_id']
                }
                answer = generate_answer(llm, source_text, query)
                st.success(answer)
                st.caption("📎 Source: " + format_reference(metadata))
            else:
                st.warning("❌ Answer not found in the documents.")

        end_time = time.time()
        st.text(f"⏱️ Time taken: {end_time - start_time:.2f} seconds")

        gc.collect()
        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()

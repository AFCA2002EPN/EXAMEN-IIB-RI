import streamlit as st
import pandas as pd
import numpy as np
import faiss
import os
from sentence_transformers import SentenceTransformer, CrossEncoder
import google.generativeai as genai

st.set_page_config(page_title="arXiv RAG Assistant", layout="wide")

@st.cache_resource
def load_models():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    return model, reranker

@st.cache_resource
def load_data_and_index():
    df = pd.read_csv('arxiv_data.csv')
    df['text_to_embed'] = df['titles'].fillna('') + ' ' + df['summaries'].fillna('')
    model, _ = load_models()
    embeddings = model.encode(df['text_to_embed'].tolist())
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.ascontiguousarray(embeddings.astype('float32')))
    return df, index

model, reranker_model = load_models()
df, index = load_data_and_index()

def semantic_search(query, top_k=5):
    query_embedding = model.encode([query])
    distances, indices = index.search(np.ascontiguousarray(query_embedding.astype('float32')), top_k)
    results = df.iloc[indices[0]].copy()
    results['distance'] = distances[0]
    return results

def rerank_documents(query, retrieved_df):
    pairs = [[query, row['summaries']] for _, row in retrieved_df.iterrows()]
    scores = reranker_model.predict(pairs)
    retrieved_df['rerank_score'] = scores
    return retrieved_df.sort_values(by='rerank_score', ascending=False)

st.title("🔍 Asistente de Investigación arXiv (RAG)")
query = st.text_input("¿Qué deseas investigar hoy?")

if query:
    results = semantic_search(query, top_k=5)
    reranked = rerank_documents(query, results)
    best_score = reranked['rerank_score'].max()
    
    if best_score < -5.0:
        st.warning("⚠️ No he encontrado información relevante.")
    else:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("Respuesta del Asistente")
            api_key = os.getenv('GOOGLE_API_KEY')
            if api_key:
                genai.configure(api_key=api_key)
                llm = genai.GenerativeModel('gemini-pro')
                contexto = "\n".join(reranked['summaries'].tolist()[:3])
                response = llm.generate_content(f"Contexto: {contexto}\n\nPregunta: {query}")
                st.markdown(response.text)
            else:
                st.info("Configure la API Key en Settings.")
        with col2:
            st.subheader("Fuentes")
            for _, row in reranked.iterrows():
                with st.expander(f"📄 {row['titles'][:50]}..."):
                    st.write(row['summaries'])
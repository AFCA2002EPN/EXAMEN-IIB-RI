import streamlit as st
import pandas as pd
import numpy as np
import faiss
import os
from sentence_transformers import SentenceTransformer, CrossEncoder
import google.generativeai as genai

# 1. Configuración de página
st.set_page_config(page_title="arXiv RAG Assistant", layout="wide")

# 2. Carga de Modelos
@st.cache_resource
def load_models():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    return model, reranker

# 3. Carga de Datos e Índice Faiss
@st.cache_resource
def load_data_and_index():
    # Leemos el dataset
    df = pd.read_csv("arxiv_data.csv", engine='python', on_bad_lines='skip', encoding='utf-8')
    # Cargamos directamente tu índice guardado para máxima velocidad y ahorro de RAM
    index = faiss.read_index("faiss_index.faiss")
    return df, index

# Inicializamos
model, reranker_model = load_models()
df, index = load_data_and_index()

# 4. Funciones de Búsqueda
def semantic_search(query, top_k=5):
    query_embedding = model.encode([query], device='cpu')
    distances, indices = index.search(np.ascontiguousarray(query_embedding.astype('float32')), top_k)
    results = df.iloc[indices[0]].copy()
    results['distance'] = distances[0]
    return results

def rerank_documents(query, retrieved_df):
    pairs = [[query, row['summaries']] for _, row in retrieved_df.iterrows()]
    scores = reranker_model.predict(pairs, device='cpu')
    retrieved_df = retrieved_df.copy()
    retrieved_df['rerank_score'] = scores
    return retrieved_df.sort_values(by='rerank_score', ascending=False)

# 5. Interfaz de Usuario
st.title("🔍 Asistente de Investigación arXiv (RAG)")
query = st.text_input("¿Qué deseas investigar hoy?")

if query:
    with st.spinner("Buscando en el corpus de arXiv..."):
        results = semantic_search(query, top_k=5)
        reranked = rerank_documents(query, results)
        best_score = reranked['rerank_score'].max()

        # Filtro estricto para evitar respuestas fuera de contexto
        if best_score < 3.0:
            st.warning("⚠️ No he encontrado información relevante sobre este tema en el corpus.")
        else:
            col1, col2 = st.columns([2, 1])
            with col1:
                st.subheader("Respuesta del Asistente")
                api_key = os.getenv('GOOGLE_API_KEY')
                
                if api_key:
                    try:
                        genai.configure(api_key=api_key)
                        llm = genai.GenerativeModel('gemini-2.0-flash')
                        contexto = "\n".join(reranked['summaries'].tolist()[:3])
                        prompt = f"Contexto: {contexto}\n\nPregunta: {query}\n\nRespuesta:"
                        response = llm.generate_content(prompt)
                        st.markdown(response.text)
                    except Exception as e:
                        st.error(f"Error al generar respuesta con Gemini: {str(e)}")
                else:
                    st.info("Configure la variable de entorno GOOGLE_API_KEY en los Secrets de Streamlit.")
            
            with col2:
                st.subheader("Fuentes")
                for _, row in reranked.iterrows():
                    with st.expander(f"📄 {row['titles'][:50]}..."):
                        st.write(f"**Puntaje:** `{row['rerank_score']:.4f}`")
                        st.write(row['summaries'])

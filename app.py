import streamlit as st
import pandas as pd
import numpy as np
import faiss
import os
from sentence_transformers import SentenceTransformer, CrossEncoder
import google.generativeai as genai

# 1. Configuración inicial de la página
st.set_page_config(page_title="arXiv RAG Assistant", layout="wide")

# 2. Carga de modelos con caché para no saturar la memoria
@st.cache_resource
def load_models():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    return model, reranker

# 3. Carga de datos e índice CON LÍMITE DE 200 PARA EVITAR COLAPSO
@st.cache_resource
def load_data_and_index():
    # El .head(200) salva la memoria RAM del servidor
    df = pd.read_csv('arxiv_data.csv', engine='python', on_bad_lines='skip', encoding='utf-8').head(200)
    df['text_to_embed'] = df['titles'].fillna('') + ' ' + df['summaries'].fillna('')
    
    model, _ = load_models()
    embeddings = model.encode(df['text_to_embed'].tolist())
    dimension = embeddings.shape[1]
    
    index = faiss.IndexFlatL2(dimension)
    index.add(np.ascontiguousarray(embeddings.astype('float32')))
    return df, index

# 4. Inicializar modelos y datos
model, reranker_model = load_models()
df, index = load_data_and_index()

# 5. Funciones de búsqueda RAG
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

# 6. Interfaz Visual de Usuario
st.title("🔍 Asistente de Investigación arXiv (RAG)")
query = st.text_input("¿Qué deseas investigar hoy?")

if query:
    with st.spinner("Procesando consulta y buscando documentos..."):
        results = semantic_search(query, top_k=5)
        reranked = rerank_documents(query, results)
        best_score = reranked['rerank_score'].max()
        
        if best_score < -5.0:
            st.warning("⚠️ No he encontrado información relevante en el corpus de 200 documentos cargados.")
        else:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("Respuesta del Asistente")
                # Intenta leer la API Key desde los secrets de Streamlit o variables de entorno
                api_key = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))
                
                if api_key:
                    try:
                        genai.configure(api_key=api_key)
                        llm = genai.GenerativeModel('gemini-1.5-flash')
                        contexto = "\n".join(reranked['summaries'].tolist()[:3])
                        prompt = f"Contexto: {contexto}\n\nPregunta: {query}\n\nResponde detalladamente basándote en el contexto."
                        response = llm.generate_content(prompt)
                        st.markdown(response.text)
                    except Exception as e:
                        st.error(f"Error al generar respuesta con Gemini: {str(e)}")
                else:
                    st.info("Por favor, configure su GOOGLE_API_KEY en los Secrets de la aplicación.")
            
            with col2:
                st.subheader("Fuentes Recuperadas")
                for _, row in reranked.iterrows():
                    with st.expander(f"📄 {row['titles'][:50]}..."):
                        st.markdown(f"**Relevancia:** `{row['rerank_score']:.2f}`")
                        st.write(row['summaries'])

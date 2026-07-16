import streamlit as st
import os
import pandas as pd
import numpy as np
import faiss
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

# --- Carga de datos y modelos ---
# Asegúrate de que estos archivos estén en el mismo directorio que streamlit_app.py al desplegar
# El archivo 'arxiv_data.csv' debe estar disponible en la raíz del proyecto.
# Puedes usar st.cache_data para cargar estos datos una sola vez y mejorar el rendimiento

@st.cache_resource
def load_data_and_models():
    df = pd.read_csv("arxiv_data.csv", engine='python', on_bad_lines='skip', encoding='utf-8')

    document_embeddings = np.load("document_embeddings.npy")

    dimension = document_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.read_index("faiss_index.faiss")

    model = SentenceTransformer('all-MiniLM-L6-v2')
    return df, document_embeddings, index, model

df, document_embeddings, index, model = load_data_and_models()

# --- Configuración de la API de Gemini ---
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    st.error("La variable de entorno GOOGLE_API_KEY no está configurada. \nPor favor, configúrala en tu plataforma de despliegue.")
    st.stop()
genai.configure(api_key=GOOGLE_API_KEY)

# --- Funciones del Sistema RAG (estas son las mismas que en app.py) ---
def semantic_search(query, top_k=10):
    query_embedding = model.encode([query])
    distances, indices = index.search(np.ascontiguousarray(query_embedding.astype('float32')), top_k)
    results = df.iloc[indices[0]].copy()
    results['distance'] = distances[0]
    return results

def recuperar_contexto(query, top_k=5):
    retrieved_docs = semantic_search(query, top_k=top_k)
    if retrieved_docs.empty:
        return "", pd.DataFrame()
    contexto_str = "\n\n".join(retrieved_docs['summaries'].tolist())
    return contexto_str, retrieved_docs

def generar_respuesta_rag(query, contexto):
    if not contexto or contexto.strip() == "":
        return "Lo siento, el corpus actual no contiene información suficiente para responder a esta consulta."

    system_prompt = '''Eres un asistente académico experto en análisis de artículos científicos. \n    Tu tarea es responder a la pregunta del usuario utilizando ÚNICAMENTE la información provista en el "Contexto extraído de los artículos". \n    Si la respuesta no se encuentra en el contexto, debes indicar explícitamente que el corpus no tiene información suficiente para responder. \n    No utilices tu conocimiento previo ni inventes información.'''

    llm_model = genai.GenerativeModel(
        model_name='gemini-pro-latest',
        system_instruction=system_prompt,
        generation_config=genai.GenerationConfig(
            temperature=0.2,
            max_output_tokens=1024
        )
    )

    user_message = f"Contexto extraído de los artículos:\n{contexto}\n\nPregunta del usuario: {query}"

    try:
        response = llm_model.generate_content(user_message, request_options={'timeout': 180})
        return response.text
    except Exception as e:
        return f"Error al generar la respuesta con Gemini: {str(e)}"


# --- Interfaz de Usuario con Streamlit ---
st.set_page_config(page_title="Sistema RAG - arXiv", layout="wide")
st.title("📚 Asistente de Recuperación Científica (RAG) con Streamlit")
st.markdown("Ingresa tu consulta sobre inteligencia artificial, machine learning o ciencias de la computación. El sistema buscará en el corpus de **arXiv** y generará una respuesta basada en las evidencias encontradas.")

user_query = st.text_area(
    "Escribe tu consulta aquí:",
    placeholder="Ej. What are the main applications of Graph Neural Networks?",
    height=100
)

if st.button("Enviar Consulta"):
    if user_query:
        with st.spinner("Buscando documentos y generando respuesta..."):
            contexto_recuperado, evidencias_df = recuperar_contexto(user_query)
            respuesta_llm = generar_respuesta_rag(user_query, contexto_recuperado)

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Respuesta Generada")
                st.write(respuesta_llm)

            with col2:
                st.subheader("Evidencias")
                if evidencias_df.empty:
                    st.info("No se encontraron documentos relevantes en el corpus para esta consulta.")
                else:
                    for i, row in evidencias_df.iterrows():
                        st.markdown(f"**Documento {i+1}: {row['titles']}**")
                        st.markdown(f"*Similitud (Distancia L2): {row['distance']:.4f}*")
                        fragmento = row['summaries'][:300] + "..." if len(row['summaries']) > 300 else row['summaries']
                        st.markdown(f"> {fragmento}")
                        st.markdown("---")
    else:
        st.warning("Por favor, introduce una consulta.")

import streamlit as st
import os
import google.generativeai as genai
from qdrant_client import QdrantClient
from qdrant_client.models import SparseVector, Prefetch, FusionQuery, Fusion
from fastembed import SparseTextEmbedding
from dotenv import load_dotenv
from embedding_model import load_embedding_model, get_embeddings

# Setup
load_dotenv()
st.set_page_config(
    layout="wide", 
    page_title="Hybrid RAG System", 
    page_icon="🔮",
)

# --- STATO DEL VIDEO ---
if 'video_active' not in st.session_state:
    st.session_state.video_active = True

def skip_video():
    st.session_state.video_active = False

# Configurazione API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

# --- CONFIGURAZIONE ---
COLLECTION_NAME = "hybrid_rag_collection"
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# Cache dei modelli
@st.cache_resource
def get_sparse_model():
    return SparseTextEmbedding(model_name="Qdrant/bm25")

@st.cache_resource
def get_dense_model():
    """Carica il modello ONNX per dense embeddings"""
    tokenizer, model = load_embedding_model()
    if tokenizer is None or model is None:
        st.error("❌ Impossibile caricare il modello di embedding")
        st.stop()
    return tokenizer, model

sparse_model = get_sparse_model()
dense_tokenizer, dense_model = get_dense_model()

# --- CSS PERSONALIZZATO ---
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0 1rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
    }
    
    .gen-stat-card {
        background-color: rgba(102, 126, 234, 0.1);
        border: 1px solid rgba(102, 126, 234, 0.3);
        height: 45px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 8px;
        font-weight: 500;
        margin-bottom: 0px;
    }

    div[data-testid="stButton"] button {
        height: 45px !important;
    }

    .doc-card {
        background-color: rgba(128, 128, 128, 0.08);
        border-left: 4px solid #667eea;
        padding: 1rem;
        margin-top: 0.5rem;
        border-radius: 5px;
    }
    
    .score-badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        background: #667eea;
        color: white;
        border-radius: 12px;
        font-size: 0.75rem;
    }
    
    .answer-section {
        border: 1px solid #667eea;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0 2rem 0;
        background-color: rgba(102, 126, 234, 0.03);
    }
    
    .preview-text {
        padding: 0.8rem;
        background-color: rgba(255, 255, 255, 0.03);
        border-radius: 5px;
        border: 1px solid rgba(128, 128, 128, 0.1);
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# --- FUNZIONI ---
def hybrid_search(query, top_k=30):
    """Esegue ricerca ibrida usando ONNX per dense embeddings"""
    # Dense embedding con ONNX
    dense_vector = get_embeddings(query, dense_tokenizer, dense_model)[0].tolist()
    
    # Sparse embedding con BM25
    sparse_embeddings = list(sparse_model.embed([query]))
    sparse_embedding = sparse_embeddings[0]
    sparse_vector = SparseVector(
        indices=sparse_embedding.indices.tolist(),
        values=sparse_embedding.values.tolist()
    )
    
    return qdrant.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            Prefetch(query=dense_vector, using="dense", limit=top_k * 2),
            Prefetch(query=sparse_vector, using="sparse", limit=top_k * 2)
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=top_k,
        with_payload=True
    ).points

def generate_answer(query, selected_docs):
    model = genai.GenerativeModel(GEMINI_MODEL)
    context = "\n\n---\n\n".join([f"[Doc {i+1}]: {d.payload['text']}" for i, d in enumerate(selected_docs)])
    prompt = f"Rispondi alla domanda usando solo i documenti forniti.\n\nCONTESTO:\n{context}\n\nDOMANDA: {query}"
    return model.generate_content(prompt).text

# --- INTERFACCIA ---

# 1. VIDEO OVERLAY (Se attivo)
if st.session_state.video_active:
    empty_l, content_col, empty_r = st.columns([0.05, 0.9, 0.05])
    
    with content_col:
        col_header, col_btn = st.columns([0.7, 0.2])
        with col_header:
            st.markdown("<h3 style='margin: 0;'>🔮 OHmyRAGS! ✨</h3>", unsafe_allow_html=True)
        with col_btn:
            st.button("⏩ Salta Video", on_click=skip_video, use_container_width=True)
        
        st.markdown("<div style='margin-bottom: 15px;'></div>", unsafe_allow_html=True)
        
        if os.path.exists("video.mp4"):
            st.video("video.mp4", autoplay=True)
        else:
            st.warning("File 'video.mp4' non trovato. Passando all'app...")
            skip_video()
            
    st.stop()

# 2. APP PRINCIPALE
st.markdown("""<div class="main-header"><h1>🔮 OHmyRAGS! ✨</h1></div>""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 🤗 Open-source 👾​ Hybrid &emsp;&emsp; 🧠​ RAG System")
    with st.expander("⚙️ Configurazione", expanded=True):
        st.markdown(f"**Parsing:** `Docling` \n\n**Chunking:** `Docling`\n\n **Dense:** `Gemma 0.3B ONNX`\n\n**Sparse:** `BM25`\n\n**Database:** `Qdrant`\n\n**Ranking:** `RRF`\n\n**LLM:** `{GEMINI_MODEL}`")
    st.markdown("---")
    top_k = st.slider("Quanti documenti recuperare? 🤔", 5, 100, 30)
    st.markdown("---")
    auto_select_top = st.number_input("✅ Documenti da pre-selezionare: ", 0, 10, 5)
    st.markdown("---")
    show_full_text = st.checkbox("🧐​ Mostra tutto il testo 📖", value=False)

st.markdown("### 💬 Inserisci la tua domanda")
col_q1, col_q2 = st.columns([4, 1])
with col_q1:
    query = st.text_input("Domanda", placeholder="Spiegami in maniera semplice il fuorigioco", label_visibility="collapsed")
with col_q2:
    search_btn = st.button("🔎 Cerca", type="primary", use_container_width=True)

if 'results' not in st.session_state: st.session_state.results = None
if 'query' not in st.session_state: st.session_state.query = ""
if 'generated_answer' not in st.session_state: st.session_state.generated_answer = None

if search_btn and query:
    with st.spinner("🔍 Ricerca in corso..."):
        st.session_state.results = hybrid_search(query, top_k=top_k)
        st.session_state.query = query
        st.session_state.generated_answer = None

if st.session_state.results:
    st.markdown("---")
    st.markdown("### ✨ Genera Risposta Contestualizzata")
    
    selected_indices = [i for i in range(len(st.session_state.results)) if st.session_state.get(f"select_{i}", i < auto_select_top)]
    selected_docs = [st.session_state.results[i] for i in selected_indices]
    sources_count = len(set(d.payload.get('source', 'unknown') for d in selected_docs))

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        st.markdown(f'<div class="gen-stat-card">📌 {len(selected_docs)} documenti selezionati</div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="gen-stat-card">📚 fonti diverse: {sources_count}</div>', unsafe_allow_html=True)
    with c3:
        generate_btn = st.button("🪄 Genera Risposta", type="primary", use_container_width=True)

    if generate_btn and selected_docs:import streamlit as st
import os
import google.generativeai as genai
from qdrant_client import QdrantClient
from qdrant_client.models import SparseVector, Prefetch, FusionQuery, Fusion
from fastembed import SparseTextEmbedding
from dotenv import load_dotenv
from embedding_model import load_embedding_model, get_embeddings

# Setup
load_dotenv()
st.set_page_config(
    layout="wide", 
    page_title="Hybrid RAG System", 
    page_icon="🔮",
)

# --- STATO DEL VIDEO ---
if 'video_active' not in st.session_state:
    st.session_state.video_active = True

def skip_video():
    st.session_state.video_active = False

# Configurazione API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

# --- CONFIGURAZIONE ---
COLLECTION_NAME = "hybrid_rag_collection"
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# Cache dei modelli
@st.cache_resource
def get_sparse_model():
    return SparseTextEmbedding(model_name="Qdrant/bm25")

@st.cache_resource
def get_dense_model():
    """Carica il modello ONNX per dense embeddings"""
    tokenizer, model = load_embedding_model()
    if tokenizer is None or model is None:
        st.error("❌ Impossibile caricare il modello di embedding")
        st.stop()
    return tokenizer, model

sparse_model = get_sparse_model()
dense_tokenizer, dense_model = get_dense_model()

# --- CSS PERSONALIZZATO ---
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0 1rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
    }
    
    .gen-stat-card {
        background-color: rgba(102, 126, 234, 0.1);
        border: 1px solid rgba(102, 126, 234, 0.3);
        height: 45px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 8px;
        font-weight: 500;
        margin-bottom: 0px;
    }

    div[data-testid="stButton"] button {
        height: 45px !important;
    }

    .doc-card {
        background-color: rgba(128, 128, 128, 0.08);
        border-left: 4px solid #667eea;
        padding: 1rem;
        margin-top: 0.5rem;
        border-radius: 5px;
    }
    
    .score-badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        background: #667eea;
        color: white;
        border-radius: 12px;
        font-size: 0.75rem;
    }
    
    .answer-section {
        border: 1px solid #667eea;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0 2rem 0;
        background-color: rgba(102, 126, 234, 0.03);
    }
    
    .preview-text {
        padding: 0.8rem;
        background-color: rgba(255, 255, 255, 0.03);
        border-radius: 5px;
        border: 1px solid rgba(128, 128, 128, 0.1);
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# --- FUNZIONI ---
def hybrid_search(query, top_k=30):
    """Esegue ricerca ibrida usando ONNX per dense embeddings"""
    # Dense embedding con ONNX
    dense_vector = get_embeddings(query, dense_tokenizer, dense_model)[0].tolist()
    
    # Sparse embedding con BM25
    sparse_embeddings = list(sparse_model.embed([query]))
    sparse_embedding = sparse_embeddings[0]
    sparse_vector = SparseVector(
        indices=sparse_embedding.indices.tolist(),
        values=sparse_embedding.values.tolist()
    )
    
    return qdrant.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            Prefetch(query=dense_vector, using="dense", limit=top_k * 2),
            Prefetch(query=sparse_vector, using="sparse", limit=top_k * 2)
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=top_k,
        with_payload=True
    ).points

def generate_answer(query, selected_docs):
    model = genai.GenerativeModel(GEMINI_MODEL)
    context = "\n\n---\n\n".join([f"[Doc {i+1}]: {d.payload['text']}" for i, d in enumerate(selected_docs)])
    prompt = f"Rispondi alla domanda usando solo i documenti forniti.\n\nCONTESTO:\n{context}\n\nDOMANDA: {query}"
    return model.generate_content(prompt).text

# --- INTERFACCIA ---

# 1. VIDEO OVERLAY (Se attivo)
if st.session_state.video_active:
    empty_l, content_col, empty_r = st.columns([0.05, 0.9, 0.05])
    
    with content_col:
        col_header, col_btn = st.columns([0.7, 0.2])
        with col_header:
            st.markdown("<h3 style='margin: 0;'>🔮 OHmyRAGS! ✨</h3>", unsafe_allow_html=True)
        with col_btn:
            st.button("⏩ Salta Video", on_click=skip_video, use_container_width=True)
        
        st.markdown("<div style='margin-bottom: 15px;'></div>", unsafe_allow_html=True)
        
        if os.path.exists("video.mp4"):
            st.video("video.mp4", autoplay=True)
        else:
            st.warning("File 'video.mp4' non trovato. Passando all'app...")
            skip_video()
            
    st.stop()

# 2. APP PRINCIPALE
st.markdown("""<div class="main-header"><h1>🔮 OHmyRAGS! ✨</h1></div>""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 🤗 Open-source 👾​ Hybrid &emsp;&emsp; 🧠​ RAG System")
    with st.expander("⚙️ Configurazione", expanded=True):
        st.markdown(f"**Parsing:** `Docling` \n\n**Chunking:** `Docling`\n\n **Dense:** `Gemma 0.3B ONNX`\n\n**Sparse:** `BM25`\n\n**Database:** `Qdrant`\n\n**Ranking:** `RRF`\n\n**LLM:** `{GEMINI_MODEL}`")
    st.markdown("---")
    top_k = st.slider("Quanti documenti recuperare? 🤔", 5, 100, 30)
    st.markdown("---")
    auto_select_top = st.number_input("✅ Documenti da pre-selezionare: ", 0, 10, 5)
    st.markdown("---")
    show_full_text = st.checkbox("🧐​ Mostra tutto il testo 📖", value=False)

st.markdown("### 💬 Inserisci la tua domanda")
col_q1, col_q2 = st.columns([4, 1])
with col_q1:
    query = st.text_input("Domanda", placeholder="Spiegami in maniera semplice il fuorigioco", label_visibility="collapsed")
with col_q2:
    search_btn = st.button("🔎 Cerca", type="primary", use_container_width=True)

if 'results' not in st.session_state: st.session_state.results = None
if 'query' not in st.session_state: st.session_state.query = ""
if 'generated_answer' not in st.session_state: st.session_state.generated_answer = None

if search_btn and query:
    with st.spinner("🔍 Ricerca in corso..."):
        st.session_state.results = hybrid_search(query, top_k=top_k)
        st.session_state.query = query
        st.session_state.generated_answer = None

if st.session_state.results:
    st.markdown("---")
    st.markdown("### ✨ Genera Risposta Contestualizzata")
    
    selected_indices = [i for i in range(len(st.session_state.results)) if st.session_state.get(f"select_{i}", i < auto_select_top)]
    selected_docs = [st.session_state.results[i] for i in selected_indices]
    sources_count = len(set(d.payload.get('source', 'unknown') for d in selected_docs))

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        st.markdown(f'<div class="gen-stat-card">📌 {len(selected_docs)} documenti selezionati</div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="gen-stat-card">📚 fonti diverse: {sources_count}</div>', unsafe_allow_html=True)
    with c3:
        generate_btn = st.button("🪄 Genera Risposta", type="primary", use_container_width=True)

    if generate_btn and selected_docs:
        with st.spinner("🗿 Elaborazione..."):
            st.session_state.generated_answer = generate_answer(st.session_state.query, selected_docs)

    if st.session_state.generated_answer:
        st.markdown(f'<div class="answer-section"><h3 style="color: #667eea; margin-top:0;">💡 Risposta Generata</h3>{st.session_state.generated_answer}</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 📚 Risultati della Ricerca")
    
    for i, doc in enumerate(st.session_state.results):
        col_check, col_content = st.columns([0.06, 0.94])
        with col_check:
            st.checkbox(f"#{i+1}", value=(i < auto_select_top), key=f"select_{i}")
        with col_content:
            source = doc.payload.get('source', 'unknown')
            is_table = doc.payload.get('is_table', False)
            doc_type = doc.payload.get('type', 'text')
            icon = "📊" if is_table else "📄"
            type_label = "TABELLA" if is_table else "TESTO"
            
            st.markdown(f'<div class="doc-card"><div style="display: flex; justify-content: space-between; align-items: center;"><span><strong>{icon} {type_label} {i+1}</strong> <span style="color: #888; margin-left:10px;">({source})</span></span><span class="score-badge">RRF: {doc.score:.4f}</span></div></div>', unsafe_allow_html=True)
            text = doc.payload['text']
            if show_full_text:
                with st.expander("📖 Espandi testo"): st.markdown(text)
            else:
                preview = text[:350] + "..." if len(text) > 350 else text
                st.markdown(f'<div class="preview-text">{preview}</div>', unsafe_allow_html=True)
        st.markdown("<div style='margin-bottom: 10px;'></div>", unsafe_allow_html=True)
        with st.spinner("🗿 Elaborazione..."):
            st.session_state.generated_answer = generate_answer(st.session_state.query, selected_docs)

    if st.session_state.generated_answer:
        st.markdown(f'<div class="answer-section"><h3 style="color: #667eea; margin-top:0;">💡 Risposta Generata</h3>{st.session_state.generated_answer}</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 📚 Risultati della Ricerca")
    
    for i, doc in enumerate(st.session_state.results):
        col_check, col_content = st.columns([0.06, 0.94])
        with col_check:
            st.checkbox(f"#{i+1}", value=(i < auto_select_top), key=f"select_{i}")
        with col_content:
            source = doc.payload.get('source', 'unknown')
            is_table = doc.payload.get('is_table', False)
            doc_type = doc.payload.get('type', 'text')
            icon = "📊" if is_table else "📄"
            type_label = "TABELLA" if is_table else "TESTO"
            
            st.markdown(f'<div class="doc-card"><div style="display: flex; justify-content: space-between; align-items: center;"><span><strong>{icon} {type_label} {i+1}</strong> <span style="color: #888; margin-left:10px;">({source})</span></span><span class="score-badge">RRF: {doc.score:.4f}</span></div></div>', unsafe_allow_html=True)
            text = doc.payload['text']
            if show_full_text:
                with st.expander("📖 Espandi testo"): st.markdown(text)
            else:
                preview = text[:350] + "..." if len(text) > 350 else text
                st.markdown(f'<div class="preview-text">{preview}</div>', unsafe_allow_html=True)
        st.markdown("<div style='margin-bottom: 10px;'></div>", unsafe_allow_html=True)
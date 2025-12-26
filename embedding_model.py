"""
Modulo per gestire embeddings con ONNX Embedding Gemma
Ottimizzato per Streamlit Cloud (free tier)
"""
import streamlit as st
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForFeatureExtraction
import torch
import numpy as np

MODEL_NAME = "onnx-community/embeddinggemma-300m-ONNX"

@st.cache_resource
def load_embedding_model():
    """
    Carica il modello ONNX con cache per evitare reload
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = ORTModelForFeatureExtraction.from_pretrained(
            MODEL_NAME,
            file_name="model_quantized.onnx"  # Versione quantizzata per risparmiare memoria
        )
        return tokenizer, model
    except Exception as e:
        st.error(f"Errore nel caricamento del modello: {e}")
        return None, None

def get_embeddings(texts, tokenizer, model, batch_size=8):
    """
    Genera embeddings per una lista di testi
    Args:
        texts: lista di stringhe o stringa singola
        tokenizer: tokenizer del modello
        model: modello ONNX
        batch_size: dimensione batch per elaborazione
    Returns:
        numpy array con embeddings
    """
    if isinstance(texts, str):
        texts = [texts]
    
    embeddings = []
    
    # Processa in batch per gestire memoria limitata
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        
        # Tokenizza con limite corretto
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=2048,  # ✅ 2048 token supportati dal modello
            return_tensors="pt"
        )
        
        # Genera embeddings
        with torch.no_grad():
            outputs = model(**inputs)
            # Mean pooling
            attention_mask = inputs['attention_mask']
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            batch_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            
        embeddings.append(batch_embeddings.cpu().numpy())
    
    return np.vstack(embeddings) if len(embeddings) > 1 else embeddings[0]

def get_embedding_dimension():
    """
    Ritorna la dimensione degli embeddings del modello
    """
    return 768  # Embedding Gemma 300M produce vettori di 768 dimensioni
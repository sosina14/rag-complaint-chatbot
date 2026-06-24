# app.py — CrediTrust RAG Complaint Chatbot
# Gradio interface for the RAG pipeline
# Author: Sosina Ayele

import gradio as gr
import numpy as np
import pandas as pd
import faiss
import os
import sys
sys.path.insert(0, 'src')
from sentence_transformers import SentenceTransformer
from transformers import pipeline as hf_pipeline
import warnings
warnings.filterwarnings('ignore')

# ── Configuration ──────────────────────────────────────────────
PARQUET_PATHS = [
    'data/raw/complaint_embeddings.parquet',
    r'c:\KAIM\rag-complaint-chatbot\data\raw\complaint_embeddings.parquet',
]
PRODUCTS = ["All Products", "Credit Card", "Personal Loan", "Savings Account", "Money Transfer"]

PROMPT_TEMPLATE = """You are a financial analyst assistant for CrediTrust Financial.
Your task is to answer questions about customer complaints using ONLY the retrieved complaint excerpts below.
If the context does not contain enough information, say: "I don't have enough information in the retrieved complaints to answer this."
Be concise, specific, and actionable. Your audience is a Product Manager.

Retrieved Complaint Excerpts:
{context}

Question: {question}

Answer:"""

# ── Load Resources ─────────────────────────────────────────────
print("Loading embedding model...")
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

print("Loading complaint embeddings...")
df_emb = None
for p in PARQUET_PATHS:
    if os.path.exists(p):
        df_emb = pd.read_parquet(p)
        print(f"Loaded: {df_emb.shape} from {p}")
        break

if df_emb is None:
    raise FileNotFoundError("complaint_embeddings.parquet not found!")

# Auto-detect columns
emb_col = None
text_col = None
product_col = None

for col in df_emb.columns:
    val = df_emb[col].iloc[0]
    if isinstance(val, (list, np.ndarray)) and len(val) > 10:
        emb_col = col
    elif isinstance(val, str) and len(val) > 30 and text_col is None:
        text_col = col
    if 'product' in col.lower():
        product_col = col

print(f"Columns detected — emb: {emb_col}, text: {text_col}, product: {product_col}")

# Build FAISS index
print("Building FAISS index...")
embeddings = np.array(df_emb[emb_col].tolist()).astype('float32')
norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
norms = np.where(norms == 0, 1, norms)
embeddings_norm = embeddings / norms

dim = embeddings_norm.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(embeddings_norm)

chunks = df_emb[text_col].tolist()
metadata = df_emb.drop(columns=[emb_col]).to_dict('records')
print(f"Index ready! {index.ntotal:,} vectors")

# Load LLM
print("Loading LLM...")
generator = None
try:
    generator = hf_pipeline(
        'text2text-generation',
        model='google/flan-t5-small',
        max_new_tokens=250,
    )
    print("flan-t5-small loaded!")
except Exception as e:
    print(f"flan-t5 failed: {e}, trying opt-125m...")
    try:
        generator = hf_pipeline(
            'text-generation',
            model='facebook/opt-125m',
            max_new_tokens=250,
            do_sample=False,
        )
        print("opt-125m loaded!")
    except Exception as e2:
        print(f"LLM load failed: {e2}. Using extractive fallback.")

# ── Core Functions ─────────────────────────────────────────────
def retrieve(query, k=5, product_filter=None):
    q_emb = embed_model.encode([query], convert_to_numpy=True).astype('float32')
    q_emb = q_emb / np.linalg.norm(q_emb)
    scores, indices = index.search(q_emb, k * 5)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(chunks):
            continue
        meta = metadata[idx]
        if product_filter and product_filter != "All Products":
            prod = str(meta.get(product_col, '')).lower()
            if product_filter.lower() not in prod:
                continue
        results.append({
            'text': chunks[idx],
            'score': float(score),
            'metadata': meta
        })
        if len(results) >= k:
            break
    return results


def generate_answer(question, context):
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    if generator:
        try:
            out = generator(prompt)[0]
            # Handle both text-generation and text2text-generation
            if 'generated_text' in out:
                text = out['generated_text']
                # For text-generation, skip the prompt
                if text.startswith(prompt):
                    text = text[len(prompt):].strip()
            else:
                text = str(out)
            return text.strip() if text.strip() else "I could not generate a response. Please try rephrasing your question."
        except Exception as e:
            return f"Generation error: {e}"
    else:
        # Extractive fallback
        lines = [l.strip() for l in context.split('.') if len(l.strip()) > 40]
        q_words = set(question.lower().split())
        scored = [(sum(w in l.lower() for w in q_words), l) for l in lines]
        scored.sort(reverse=True)
        top = '. '.join([s[1] for s in scored[:3]])
        return f"Based on retrieved complaints: {top}"


def format_sources(results):
    if not results:
        return "No sources found."
    parts = []
    for i, r in enumerate(results):
        meta = r['metadata']
        prod = meta.get(product_col, 'Unknown')
        score = r['score']
        text = r['text'][:300] + ('...' if len(r['text']) > 300 else '')
        parts.append(
            f"**Source {i+1}** | Product: {prod} | Relevance: {score:.3f}\n\n{text}"
        )
    return "\n\n---\n\n".join(parts)


def chat(question, product_filter, history):
    if not question.strip():
        return history, "", "Please enter a question."

    # Retrieve
    results = retrieve(question, k=5, product_filter=product_filter)
    if not results:
        answer = "No relevant complaints found for your query. Try rephrasing or selecting a different product filter."
        sources_text = "No sources retrieved."
    else:
        context = "\n\n".join([
            f"[Excerpt {i+1} | Product: {r['metadata'].get(product_col, 'Unknown')}]\n{r['text']}"
            for i, r in enumerate(results)
        ])
        answer = generate_answer(question, context)
        sources_text = format_sources(results)

    # Update chat history
    history = history or []
    history.append((question, answer))
    return history, "", sources_text


def clear_all():
    return [], "", "", "All Products"

# ── Gradio UI ──────────────────────────────────────────────────
CSS = """
.gradio-container { max-width: 1100px; margin: auto; }
.source-box { background: #f8fafc; border-left: 4px solid #177E89; padding: 12px; }
"""

with gr.Blocks(title="CrediTrust Complaint Chatbot", css=CSS, theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🏦 CrediTrust Financial — Complaint Analysis Chatbot
    **Ask questions about customer complaints across our 4 product lines.**
    Powered by RAG (Retrieval-Augmented Generation) with semantic search over CFPB complaint data.
    """)

    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                label="Conversation",
                height=420,
                bubble_full_width=False,
            )
            with gr.Row():
                question_box = gr.Textbox(
                    placeholder="e.g. Why are customers unhappy with credit cards?",
                    label="Your Question",
                    lines=2,
                    scale=4
                )
                product_filter = gr.Dropdown(
                    choices=PRODUCTS,
                    value="All Products",
                    label="Filter by Product",
                    scale=1
                )
            with gr.Row():
                submit_btn = gr.Button("🔍 Ask", variant="primary", scale=3)
                clear_btn = gr.Button("🗑️ Clear", scale=1)

        with gr.Column(scale=1):
            gr.Markdown("### 📄 Retrieved Sources")
            sources_box = gr.Markdown(
                value="Sources will appear here after you ask a question.",
                label="Sources"
            )

    gr.Markdown("""
    ### 💡 Example Questions
    - Why are customers unhappy with credit cards?
    - What are the most common money transfer fraud complaints?
    - What problems do customers face when closing savings accounts?
    - Why do personal loan customers struggle with payments?
    - What billing disputes are reported for credit cards?
    """)

    # Wire up events
    submit_btn.click(
        fn=chat,
        inputs=[question_box, product_filter, chatbot],
        outputs=[chatbot, question_box, sources_box]
    )
    question_box.submit(
        fn=chat,
        inputs=[question_box, product_filter, chatbot],
        outputs=[chatbot, question_box, sources_box]
    )
    clear_btn.click(
        fn=clear_all,
        outputs=[chatbot, question_box, sources_box, product_filter]
    )

if __name__ == "__main__":
    demo.launch(share=False, server_port=7860)

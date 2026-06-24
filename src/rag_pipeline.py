
# src/rag_pipeline.py
import numpy as np
import pandas as pd
import faiss
import os
from sentence_transformers import SentenceTransformer
from transformers import pipeline as hf_pipeline

PROMPT_TEMPLATE = """
You are a financial analyst assistant for CrediTrust Financial.
Answer questions about customer complaints using ONLY the retrieved excerpts.
If context is insufficient, say so. Be concise and actionable.

Retrieved Complaint Excerpts:
{context}

Question: {question}

Answer:"""

class RAGPipeline:
    def __init__(self, parquet_path: str, emb_col: str, text_col: str, product_col: str):
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        df = pd.read_parquet(parquet_path)
        self.chunks = df[text_col].tolist()
        self.metadata = df.drop(columns=[emb_col]).to_dict("records")
        self.product_col = product_col
        embs = np.array(df[emb_col].tolist()).astype("float32")
        embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)
        self.index = faiss.IndexFlatIP(embs.shape[1])
        self.index.add(embs)
        try:
            self.generator = hf_pipeline("text-generation", model="facebook/opt-125m",
                max_new_tokens=300, do_sample=False)
        except:
            self.generator = None

    def retrieve(self, query, k=5, product_filter=None):
        q = self.embed_model.encode([query], convert_to_numpy=True).astype("float32")
        q = q / np.linalg.norm(q)
        scores, indices = self.index.search(q, k * 5)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0: continue
            meta = self.metadata[idx]
            if product_filter and product_filter.lower() not in str(meta.get(self.product_col,"")).lower():
                continue
            results.append({"text": self.chunks[idx], "score": float(score), "metadata": meta})
            if len(results) >= k: break
        return results

    def query(self, question, k=5, product_filter=None):
        results = self.retrieve(question, k=k, product_filter=product_filter)
        context = "\n\n".join([f"[Excerpt {i+1}]\n{r['text']}" for i, r in enumerate(results)])
        prompt = PROMPT_TEMPLATE.format(context=context, question=question)
        if self.generator:
            out = self.generator(prompt)[0]["generated_text"][len(prompt):].strip()
        else:
            out = "LLM unavailable. Top complaint: " + (results[0]["text"][:300] if results else "No results found.")
        return {"answer": out, "sources": results}

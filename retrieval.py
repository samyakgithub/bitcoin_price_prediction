import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer, util

class TextRetriever:
    def __init__(self, embedding_model_name='all-MiniLM-L6-v2', corpus_path='data/external_texts/news_headlines.json'):
        self.model = SentenceTransformer(embedding_model_name)
        self.corpus_path = corpus_path
        self.corpus = self.load_corpus()
        self.corpus_embeddings = self.embed_corpus()

    def load_corpus(self):
        if not os.path.exists(self.corpus_path):
            raise FileNotFoundError(f"Corpus file not found: {self.corpus_path}")
        with open(self.corpus_path, 'r', encoding='utf-8') as f:
            corpus = json.load(f)
        return corpus

    def embed_corpus(self):
        embeddings = self.model.encode(self.corpus, convert_to_tensor=True, show_progress_bar=True)
        return embeddings

    def retrieve(self, query, top_k=5):
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        cosine_scores = util.cos_sim(query_embedding, self.corpus_embeddings)[0]
        top_results = np.argpartition(-cosine_scores.cpu(), range(top_k))[:top_k]
        results = [(self.corpus[idx], float(cosine_scores[idx])) for idx in top_results]
        results.sort(key=lambda x: x[1], reverse=True)
        return results

if __name__ == "__main__":
    retriever = TextRetriever()
    sample_query = "Bitcoin price surge"
    results = retriever.retrieve(sample_query, top_k=3)
    print("Top relevant news:")
    for text, score in results:
        print(f"{score:.4f} - {text}")

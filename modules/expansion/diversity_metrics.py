# diversity_metrics.py
"""Metrics for measuring diversity between reasoning chains."""
from typing import List, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import re

class DiversityMetrics:
    """Compute diversity metrics between chains."""
    
    def __init__(self, use_embeddings: bool = True):
        self.use_embeddings = use_embeddings
        
        if use_embeddings:
            try:
                from sentence_transformers import SentenceTransformer
                self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
            except ImportError:
                print("sentence-transformers not installed. Using n-gram only.")
                self.use_embeddings = False
    
    def compute_pairwise_diversity(
        self,
        chain_texts: List[str]
    ) -> np.ndarray:
        """
        Compute pairwise diversity matrix.
        
        Args:
            chain_texts: List of chain texts
            
        Returns:
            NxN matrix where entry (i,j) is diversity between chains i and j
        """
        n = len(chain_texts)
        diversity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i+1, n):
                div = self.compute_diversity(chain_texts[i], chain_texts[j])
                diversity_matrix[i, j] = div
                diversity_matrix[j, i] = div
        
        return diversity_matrix
    
    def compute_diversity(self, text1: str, text2: str) -> float:
        """
        Compute diversity between two texts.
        Returns value in [0, 1] where higher = more diverse.
        """
        scores = []
        
        # Embedding-based similarity
        if self.use_embeddings:
            emb_sim = self._embedding_similarity(text1, text2)
            emb_div = 1.0 - emb_sim
            scores.append(emb_div)
        
        # N-gram diversity
        ngram_div = self._ngram_diversity(text1, text2)
        scores.append(ngram_div)
        
        # Structural diversity
        struct_div = self._structural_diversity(text1, text2)
        scores.append(struct_div)
        
        return np.mean(scores)
    
    def _embedding_similarity(self, text1: str, text2: str) -> float:
        """Compute embedding-based similarity."""
        try:
            emb1 = self.encoder.encode([text1])[0]
            emb2 = self.encoder.encode([text2])[0]
            sim = cosine_similarity([emb1], [emb2])[0][0]
            return float(sim)
        except:
            return 0.5
    
    def _ngram_diversity(self, text1: str, text2: str, n: int = 3) -> float:
        """Compute n-gram based diversity."""
        def get_ngrams(text: str, n: int) -> List[str]:
            words = re.findall(r'\w+', text.lower())
            return [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]
        
        ngrams1 = set(get_ngrams(text1, n))
        ngrams2 = set(get_ngrams(text2, n))
        
        if not ngrams1 or not ngrams2:
            return 0.5
        
        intersection = len(ngrams1 & ngrams2)
        union = len(ngrams1 | ngrams2)
        
        jaccard = intersection / union if union > 0 else 0.0
        return 1.0 - jaccard  # Convert similarity to diversity
    
    def _structural_diversity(self, text1: str, text2: str) -> float:
        """Compute diversity based on text structure."""
        # Count sentences, words, etc.
        def get_structure(text: str) -> Tuple[int, int, float]:
            sentences = text.split('.')
            words = re.findall(r'\w+', text)
            avg_sent_len = len(words) / max(len(sentences), 1)
            return len(sentences), len(words), avg_sent_len
        
        s1_sents, s1_words, s1_avg = get_structure(text1)
        s2_sents, s2_words, s2_avg = get_structure(text2)
        
        # Normalize differences
        sent_diff = abs(s1_sents - s2_sents) / max(s1_sents, s2_sents, 1)
        word_diff = abs(s1_words - s2_words) / max(s1_words, s2_words, 1)
        avg_diff = abs(s1_avg - s2_avg) / max(s1_avg, s2_avg, 1)
        
        return np.mean([sent_diff, word_diff, avg_diff])
    
    def compute_novelty(
        self,
        new_chain: str,
        existing_chains: List[str]
    ) -> float:
        """
        Compute how novel a chain is compared to existing ones.
        
        Returns value in [0, 1] where higher = more novel.
        """
        if not existing_chains:
            return 1.0
        
        diversities = [
            self.compute_diversity(new_chain, existing)
            for existing in existing_chains
        ]
        
        return np.mean(diversities)
import logging
import numpy as np
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import hdbscan
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import AutoTokenizer, AutoModel
import nltk
from nltk.tokenize import sent_tokenize
from scipy.cluster.hierarchy import linkage, fcluster
from collections import defaultdict

logger = logging.getLogger(__name__)

class EnhancedClusterAnalyzer:
    def __init__(self):
        self.encoder = None
        self.tokenizer = None
        
    def load_model(self):
        """Load the DeBERTa model for better semantic understanding"""
        if self.encoder is None:
            model_name = "microsoft/deberta-v3-large"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.encoder = AutoModel.from_pretrained(model_name)
            if torch.cuda.is_available():
                self.encoder = self.encoder.cuda()
            self.encoder.eval()

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings using DeBERTa with better pooling"""
        self.load_model()
        embeddings = []
        
        with torch.no_grad():
            for text in texts:
                # Tokenize and encode
                inputs = self.tokenizer(text, padding=True, truncation=True, 
                                     max_length=512, return_tensors="pt")
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                
                # Get model output
                outputs = self.encoder(**inputs)
                
                # Use attention-weighted pooling
                attention_mask = inputs['attention_mask']
                token_embeddings = outputs[0]
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                embedding = (sum_embeddings / sum_mask).cpu().numpy()[0]
                embeddings.append(embedding)
                
        return np.array(embeddings)

    def temporal_similarity(self, t1: float, t2: float, max_time_diff: float = 300) -> float:
        """Calculate temporal similarity between timestamps"""
        time_diff = abs(t1 - t2)
        return max(0, 1 - (time_diff / max_time_diff))

    def cluster_segments(self, segments: List[Dict[str, Any]], min_cluster_size: int = 2) -> List[List[Dict[str, Any]]]:
        """Enhanced clustering with temporal awareness and hierarchical structure"""
        if not segments:
            return []

        # Extract text and timestamps
        texts = [seg.get('text', '') for seg in segments]
        timestamps = np.array([seg.get('start', 0) for seg in segments])
        
        # Get semantic embeddings
        embeddings = self.get_embeddings(texts)
        
        # Normalize timestamps
        scaler = StandardScaler()
        normalized_timestamps = scaler.fit_transform(timestamps.reshape(-1, 1))
        
        # Combine semantic and temporal features
        temporal_weight = 0.3  # Adjust this weight based on importance of temporal proximity
        combined_features = np.hstack([
            embeddings,
            normalized_timestamps * temporal_weight
        ])
        
        # Primary clustering using HDBSCAN
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=1,
            metric='euclidean',
            cluster_selection_method='eom'
        )
        
        cluster_labels = clusterer.fit_predict(combined_features)
        
        # Group segments by cluster
        clusters = defaultdict(list)
        for idx, segment in enumerate(segments):
            clusters[cluster_labels[idx]].append(segment)
            
        # Filter and sort clusters
        valid_clusters = []
        for label, cluster in clusters.items():
            if label != -1 and len(cluster) >= min_cluster_size:
                valid_clusters.append(cluster)
                
        # Sort clusters by average timestamp
        valid_clusters.sort(key=lambda x: np.mean([s.get('start', 0) for s in x]))
        
        # Hierarchical subclustering for large clusters
        refined_clusters = []
        for cluster in valid_clusters:
            if len(cluster) > 10:  # Only subcluster large clusters
                subclusters = self._create_subclusters(cluster)
                refined_clusters.extend(subclusters)
            else:
                refined_clusters.append(cluster)
        
        return refined_clusters

    def _create_subclusters(self, cluster: List[Dict[str, Any]], max_subcluster_size: int = 5) -> List[List[Dict[str, Any]]]:
        """Create hierarchical subclusters for large clusters"""
        texts = [seg.get('text', '') for seg in cluster]
        embeddings = self.get_embeddings(texts)
        
        # Calculate linkage matrix
        linkage_matrix = linkage(embeddings, method='ward')
        
        # Determine number of subclusters
        n_subclusters = max(2, len(cluster) // max_subcluster_size)
        labels = fcluster(linkage_matrix, n_subclusters, criterion='maxclust')
        
        # Group into subclusters
        subclusters = defaultdict(list)
        for idx, label in enumerate(labels):
            subclusters[label].append(cluster[idx])
            
        return list(subclusters.values())

    def analyze_cluster_coherence(self, cluster: List[Dict[str, Any]]) -> float:
        """Analyze the semantic coherence of a cluster"""
        texts = [seg.get('text', '') for seg in cluster]
        embeddings = self.get_embeddings(texts)
        
        # Calculate pairwise similarities
        similarities = cosine_similarity(embeddings)
        
        # Get average similarity excluding self-similarity
        n = len(similarities)
        total_sim = 0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                total_sim += similarities[i][j]
                count += 1
                
        return total_sim / count if count > 0 else 0.0

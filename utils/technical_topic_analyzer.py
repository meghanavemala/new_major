import logging
import json
import os
from typing import List, Dict, Any, Optional, Tuple
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import spacy
import re

logger = logging.getLogger(__name__)

class TechnicalTopicAnalyzer:
    def __init__(self):
        self.nlp = None
        self.encoder = None
        self.tokenizer = None
        self.technical_terms_cache = {}
        
    def load_models(self):
        """Load required models"""
        if self.nlp is None:
            # Load SpaCy model for technical term extraction
            try:
                self.nlp = spacy.load('en_core_web_sm')
            except:
                spacy.cli.download('en_core_web_sm')
                self.nlp = spacy.load('en_core_web_sm')
                
        if self.encoder is None:
            # Load domain-adapted BERT model
            model_name = "microsoft/deberta-v3-large"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.encoder = AutoModel.from_pretrained(model_name)
            if torch.cuda.is_available():
                self.encoder = self.encoder.cuda()
            self.encoder.eval()

    def extract_technical_terms(self, text: str) -> List[str]:
        """Extract technical terms and concepts from text"""
        self.load_models()
        
        # Check cache
        if text in self.technical_terms_cache:
            return self.technical_terms_cache[text]
            
        # Process with SpaCy
        doc = self.nlp(text)
        
        # Technical term patterns
        technical_patterns = [
            # Sorting algorithms
            r'\b(?:bubble|quick|merge|heap|insertion|selection|radix|counting|bucket|shell|tim)\s*sort(?:ing)?\b',
            # Complexity terms
            r'\b(?:O\([^)]+\)|time|space)\s*complex(?:ity)?\b',
            # Data structures
            r'\b(?:array|list|tree|heap|stack|queue|graph|hash|table)\b',
            # Algorithm characteristics
            r'\b(?:in-place|stable|recursive|iterative|adaptive|comparison|non-comparison)\b',
            # Performance terms
            r'\b(?:best|worst|average)\s*case\b',
            # Technical metrics
            r'\b(?:log[arithm]*|linear|quadratic|constant|exponential)\b'
        ]
        
        terms = set()
        
        # Extract terms using patterns
        for pattern in technical_patterns:
            matches = re.finditer(pattern, text.lower())
            for match in matches:
                terms.add(match.group())
                
        # Extract noun phrases that might be technical terms
        for chunk in doc.noun_chunks:
            # Filter for likely technical terms
            if (any(word.pos_ in ['NOUN', 'PROPN'] for word in chunk) and
                not any(word.is_stop for word in chunk) and
                len(chunk.text) > 2):
                terms.add(chunk.text.lower())
        
        # Cache results
        self.technical_terms_cache[text] = list(terms)
        return list(terms)

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings using domain-adapted model"""
        self.load_models()
        embeddings = []
        
        with torch.no_grad():
            for text in texts:
                inputs = self.tokenizer(text, padding=True, truncation=True, 
                                     max_length=512, return_tensors="pt")
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                
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

    def analyze_segments(self, 
                        segments: List[Dict[str, Any]], 
                        output_dir: str) -> Tuple[List[Dict[str, Any]], str, str]:
        """
        Analyze segments to identify and group technical topics
        
        Args:
            segments: List of transcript segments with text and timestamps
            output_dir: Directory to save JSON files
            
        Returns:
            Tuple of (topics list, timestamps file path, summaries file path)
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract text and technical terms from each segment
        segment_data = []
        for segment in segments:
            text = segment.get('text', '').strip()
            if text:
                technical_terms = self.extract_technical_terms(text)
                segment_data.append({
                    'text': text,
                    'terms': technical_terms,
                    'start': segment.get('start', 0),
                    'end': segment.get('end', 0)
                })

        # Get embeddings for segments
        texts = [s['text'] for s in segment_data]
        embeddings = self.get_embeddings(texts)
        
        # Calculate similarity matrix
        similarities = cosine_similarity(embeddings)
        
        # Group related segments into topics
        topics = []
        used_segments = set()
        
        for i, segment in enumerate(segment_data):
            if i in used_segments:
                continue
                
            # Find related segments based on similarity and shared terms
            related_indices = []
            for j, other_segment in enumerate(segment_data):
                if j not in used_segments:
                    # Check both semantic similarity and shared technical terms
                    similarity = similarities[i][j]
                    shared_terms = set(segment['terms']) & set(other_segment['terms'])
                    
                    if similarity > 0.6 or len(shared_terms) >= 2:
                        related_indices.append(j)
                        used_segments.add(j)
            
            if related_indices:
                # Create topic group
                topic_segments = [segment_data[idx] for idx in related_indices]
                
                # Extract common technical terms
                all_terms = []
                for seg in topic_segments:
                    all_terms.extend(seg['terms'])
                term_counts = defaultdict(int)
                for term in all_terms:
                    term_counts[term] += 1
                
                # Get most common terms for topic name
                top_terms = sorted(term_counts.items(), key=lambda x: x[1], reverse=True)[:3]
                topic_name = ' + '.join(term for term, _ in top_terms)
                
                topic = {
                    'id': len(topics),
                    'name': topic_name,
                    'segments': topic_segments,
                    'start_time': min(seg['start'] for seg in topic_segments),
                    'end_time': max(seg['end'] for seg in topic_segments),
                    'technical_terms': list(term_counts.keys())
                }
                topics.append(topic)
        
        # Sort topics by start time
        topics.sort(key=lambda x: x['start_time'])
        
        # Save timestamps to JSON
        timestamps_file = os.path.join(output_dir, 'topic_timestamps.json')
        timestamps_data = [{
            'topic_id': topic['id'],
            'name': topic['name'],
            'start_time': topic['start_time'],
            'end_time': topic['end_time'],
            'technical_terms': topic['technical_terms']
        } for topic in topics]
        
        with open(timestamps_file, 'w', encoding='utf-8') as f:
            json.dump(timestamps_data, f, indent=2)
            
        # Save detailed segments to JSON
        segments_file = os.path.join(output_dir, 'topic_segments.json')
        with open(segments_file, 'w', encoding='utf-8') as f:
            json.dump(topics, f, indent=2)
            
        return topics, timestamps_file, segments_file

import logging
import json
from typing import List, Dict, Any, Optional
from transformers import AutoTokenizer, BartForConditionalGeneration, pipeline
import torch
import re
from collections import defaultdict
import numpy as np
from torch.cuda.amp import autocast
import gc
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)

class TechnicalSummarizer:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Prefer GPU
        self.summarizer = None
        self.fallback_mode = False
        self.model_name = "facebook/bart-large-cnn"  # Start with full model
        
    def summarize_topics(self, topics: List[Dict[str, Any]], output_dir: str) -> str:
        """
        Generate summaries for each topic and save to file
        
        Args:
            topics: List of topic dictionaries
            output_dir: Directory to save summaries
            
        Returns:
            Path to the saved summaries file
        """
        summaries = []
        for topic in topics:
            try:
                # Generate structured summary
                summary = self.generate_topic_summary(topic)
                
                # Create summary entry
                summary_entry = {
                    'topic_id': topic['id'],
                    'name': topic.get('name', ''),
                    'start_time': topic.get('start_time', 0),
                    'end_time': topic.get('end_time', 0),
                    'summary': summary
                }
                summaries.append(summary_entry)
                
            except Exception as e:
                logger.error(f"Failed to summarize topic {topic.get('id')}: {e}")
                # Add basic entry with error handling
                summary_entry = {
                    'topic_id': topic['id'],
                    'name': topic.get('name', ''),
                    'start_time': topic.get('start_time', 0),
                    'end_time': topic.get('end_time', 0),
                    'summary': {
                        'overview': 'Failed to generate summary',
                        'technical_terms': [],
                        'key_concepts': [],
                        'examples': []
                    }
                }
                summaries.append(summary_entry)
        
        # Save summaries to file
        os.makedirs(output_dir, exist_ok=True)
        summaries_file = os.path.join(output_dir, 'topic_summaries.json')
        with open(summaries_file, 'w', encoding='utf-8') as f:
            json.dump(summaries, f, ensure_ascii=False, indent=2)
            
        return summaries_file
        
    def try_load_lighter_model(self):
        """Attempt to load a lighter model if main model fails"""
        try:
            # First try bart-base
            self.model_name = "facebook/bart-base"
            self.model = BartForConditionalGeneration.from_pretrained(
                self.model_name,
                cache_dir='model_cache',
                low_cpu_mem_usage=True
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir='model_cache'
            )
            self.model.eval()
            
            # Try to keep on GPU if available
            if torch.cuda.is_available():
                try:
                    self.model.to('cuda')
                    self.device = 'cuda'
                except RuntimeError:
                    self.device = 'cpu'
                    self.model.to('cpu')
            return True
        except Exception as e:
            logger.error(f"Failed to load lighter model: {e}")
            return False
        
    def load_model(self):
        """Load the summarization model prioritizing GPU usage"""
        if self.model is None:
            try:
                # Clear GPU memory first
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
                
                # Try loading full model first
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir='model_cache')
                self.model = BartForConditionalGeneration.from_pretrained(
                    self.model_name,
                    cache_dir='model_cache',
                    low_cpu_mem_usage=True
                )
                self.model.eval()
                
                # Try GPU first
                if torch.cuda.is_available():
                    try:
                        self.model.to('cuda')
                        self.device = 'cuda'
                        # Test GPU memory
                        dummy_input = self.tokenizer("test", return_tensors="pt").to('cuda')
                        with torch.no_grad():
                            _ = self.model(**dummy_input)
                    except RuntimeError as e:
                        # If GPU memory error, try lighter model on GPU
                        logger.warning(f"GPU memory error with full model: {e}")
                        self.model.to('cpu')
                        success = self.try_load_lighter_model()
                        if not success:
                            # If lighter model fails, stay on CPU with full model
                            self.device = 'cpu'
                            logger.info("Falling back to CPU with full model")
                else:
                    self.device = 'cpu'
                    self.model.to('cpu')
                    
            except Exception as e:
                logger.warning(f"Failed to load main model: {e}. Using fallback mode.")
                self.fallback_mode = True
                # Try lighter model one last time
                if not self.try_load_lighter_model():
                    logger.error("All model loading attempts failed")

    def extractive_summarize(self, text: str, max_sentences: int = 3) -> str:
        """Fallback extractive summarization using TF-IDF"""
        sentences = sent_tokenize(text)
        if len(sentences) <= max_sentences:
            return text
            
        vectorizer = TfidfVectorizer(stop_words='english')
        try:
            tfidf_matrix = vectorizer.fit_transform(sentences)
            sentence_scores = np.sum(tfidf_matrix.toarray(), axis=1)
            top_idx = sentence_scores.argsort()[-max_sentences:]
            return ' '.join([sentences[i] for i in sorted(top_idx)])
        except Exception as e:
            logger.warning(f"TF-IDF summarization failed: {e}")
            return ' '.join(sentences[:max_sentences])
    
    def analyze_technical_content(self, text: str) -> Dict[str, Any]:
        """Extract technical terms and key concepts from text"""
        # Simple technical term extraction based on common patterns
        technical_patterns = [
            r'\b[A-Z][a-z]*(?:[A-Z][a-z]*)+\b',  # CamelCase words
            r'\b[A-Z]{2,}\b',  # Acronyms
            r'\b\w+(?:[-_]\w+)+\b',  # Hyphenated terms
            r'\b(?:function|class|method|algorithm|data structure|complexity|runtime)\s+\w+\b',  # Programming concepts
        ]
        
        technical_terms = set()
        for pattern in technical_patterns:
            terms = re.findall(pattern, text)
            technical_terms.update(terms)
        
        # Extract key concepts using TF-IDF
        vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            max_features=10
        )
        try:
            tfidf_matrix = vectorizer.fit_transform([text])
            feature_names = vectorizer.get_feature_names_out()
            scores = tfidf_matrix.toarray()[0]
            key_concepts = [feature_names[i] for i in scores.argsort()[-5:][::-1]]
        except:
            key_concepts = []
            
        return {
            'technical_terms': list(technical_terms)[:5],
            'key_concepts': key_concepts
        }
    
    def generate_topic_summary(self, topic: Dict[str, Any], max_length: int = 150) -> Dict[str, Any]:
        """
        Generate a structured summary for a technical topic with memory efficiency
        
        Args:
            topic: Topic dictionary containing segments and metadata
            max_length: Maximum summary length
            
        Returns:
            Dictionary containing structured summary information
        """
        # Combine all text from segments
        all_text = " ".join(seg['text'] for seg in topic['segments'])
        
        # Analyze technical content
        tech_analysis = self.analyze_technical_content(all_text)
        
        # Use extractive summarization if in fallback mode or text is too long
        if self.fallback_mode or len(all_text.split()) > 500:
            return {
                'overview': self.extractive_summarize(all_text),
                'technical_terms': tech_analysis['technical_terms'],
                'key_concepts': tech_analysis['key_concepts'],
                'examples': []
            }
            
        try:
            self.load_model()
            
            # Process in smaller chunks if needed
            max_chunk_length = 200  # Process smaller chunks
            words = all_text.split()
            chunks = [' '.join(words[i:i + max_chunk_length]) 
                     for i in range(0, len(words), max_chunk_length)]
            
            summaries = []
            for chunk in chunks:
                # Create focused prompt
                technical_terms = tech_analysis['technical_terms']
                prompt = f"Summarize technically about {', '.join(technical_terms[:2])}:"
                input_text = f"{prompt}\n\n{chunk}"
                
                # Generate summary with memory efficiency
                inputs = self.tokenizer(input_text, 
                                      max_length=512,  # Reduced from 1024
                                      truncation=True,
                                      return_tensors="pt")
                
                # Move inputs to appropriate device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                try:
                    # Use automatic mixed precision if on GPU
                    with autocast(enabled=self.device=='cuda'):
                        outputs = self.model.generate(
                            **inputs,
                            max_length=max_length,
                            min_length=30,
                            num_beams=2,
                            length_penalty=1.0,
                            early_stopping=True
                        )
                    
                    # Decode and append summary
                    chunk_summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    summaries.append(chunk_summary)
                    
                    # Clear GPU memory if needed
                    if self.device == 'cuda':
                        del outputs
                        torch.cuda.empty_cache()
                        gc.collect()
                        
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        # Fall back to CPU if GPU runs out of memory
                        logger.warning("GPU OOM, falling back to CPU")
                        if self.device == 'cuda':
                            self.model.to('cpu')
                            self.device = 'cpu'
                            # Retry on CPU
                            inputs = {k: v.to('cpu') for k, v in inputs.items()}
                            outputs = self.model.generate(
                                **inputs,
                                max_length=max_length,
                                min_length=30,
                                num_beams=2
                            )
                            chunk_summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                            summaries.append(chunk_summary)
                    else:
                        # For other runtime errors, fall back to extractive
                        logger.error(f"Error in generation: {e}")
                        summaries.append(self.extractive_summarize(chunk))
            
            # Combine chunk summaries and structure the result
            overview = " ".join(summaries)
            
            # Try to extract examples if any are present
            examples = []
            example_patterns = [
                r'For example[^.]*\.',
                r'Example[^.]*\.',
                r'Such as[^.]*\.'
            ]
            for pattern in example_patterns:
                examples.extend(re.findall(pattern, all_text))
            
            return {
                'overview': overview,
                'technical_terms': tech_analysis['technical_terms'],
                'key_concepts': tech_analysis['key_concepts'],
                'examples': examples[:3]  # Limit to top 3 examples
            }
            
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            # Fall back to extractive summarization with structure
            return {
                'overview': self.extractive_summarize(all_text),
                'technical_terms': tech_analysis['technical_terms'],
                'key_concepts': tech_analysis['key_concepts'],
                'examples': []
            }
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
            
        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            min_length=50,
            num_beams=4,
            length_penalty=2.0,
            early_stopping=True
        )
        
        summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract key concepts and examples
        concepts = self.extract_key_concepts(all_text)
        examples = self.extract_examples(all_text)
        
        # Format structured summary
        structured_summary = {
            'overview': summary,
            'key_concepts': concepts,
            'examples': examples,
            'technical_terms': technical_terms
        }
        
        return structured_summary
        
    def extract_key_concepts(self, text: str) -> List[str]:
        """Extract key technical concepts from text"""
        concepts = []
        
        # Pattern for concept definitions
        definition_patterns = [
            r'(?:is|means|refers to|defined as)[^.]*\.',
            r'(?:called|known as)[^.]*\.',
            r'\b\w+(?:\s+\w+){0,5}\s+is\s+[^.]*\.'
        ]
        
        for pattern in definition_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                concept = match.group().strip()
                if len(concept) > 20:  # Filter out too short matches
                    concepts.append(concept)
                    
        return concepts[:5]  # Return top 5 concepts
        
    def extract_examples(self, text: str) -> List[str]:
        """Extract examples from text"""
        examples = []
        
        # Pattern for examples
        example_patterns = [
            r'(?:for example|e\.g\.|example)[^.]*\.',
            r'(?:such as|like)[^.]*\.',
            r'(?:instance)[^.]*\.'
        ]
        
        for pattern in example_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                example = match.group().strip()
                if len(example) > 20:
                    examples.append(example)
                    
        return examples[:3]  # Return top 3 examples
        
    def summarize_topics(self, 
                        topics: List[Dict[str, Any]], 
                        output_dir: str) -> str:
        """
        Generate structured summaries for all topics
        
        Args:
            topics: List of topic dictionaries
            output_dir: Directory to save summaries
            
        Returns:
            Path to the saved summaries file
        """
        summaries = []
        
        for topic in topics:
            summary = self.generate_topic_summary(topic)
            
            summary_entry = {
                'topic_id': topic['id'],
                'name': topic['name'],
                'start_time': topic['start_time'],
                'end_time': topic['end_time'],
                'summary': summary
            }
            summaries.append(summary_entry)
            
        # Save summaries to JSON
        summaries_file = f"{output_dir}/topic_summaries.json"
        with open(summaries_file, 'w', encoding='utf-8') as f:
            json.dump(summaries, f, indent=2)
            
        return summaries_file

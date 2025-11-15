"""
Output Content Safety Filter for Children's Language Models

This module provides comprehensive multi-layer filtering for language model outputs
to ensure child-appropriate content. Includes prohibited topics detection, toxicity
checking, readability scoring, and vocabulary screening.
"""

import re
from typing import Dict, List, Set, Tuple
from transformers import pipeline


class OutputFilter:
    """
    Comprehensive output filter with four layers of safety checks:
    1. Prohibited Topics Detection (violence, sexual content, drugs, etc.)
    2. Toxicity Detection (harmful language)
    3. Readability Score (Flesch-Kincaid Grade Level)
    4. Vocabulary Screening (Dale-Chall easy word list)
    """
    
    # Define prohibited topics for zero-shot classification
    PROHIBITED_TOPICS = [
        "sexual content",
        "violence",
        "drugs",
        "self-harm",
        "hate speech"
    ]
    
    def __init__(
        self, 
        toxic_threshold: float = 0.2,
        topic_threshold: float = 0.6,
        dale_chall_file: str = 'dale_chall_words.txt'
    ):
        """
        Initialize the output filter with all necessary models and resources.
        
        Args:
            toxic_threshold (float): Threshold for toxicity detection (default: 0.2)
            topic_threshold (float): Threshold for prohibited topics (default: 0.6)
            dale_chall_file (str): Path to Dale-Chall word list file
        """
        self.toxic_threshold = toxic_threshold
        self.topic_threshold = topic_threshold
        
        print("Loading models for output filtering...")
        
        # Load toxic-BERT for toxicity detection
        print("  - Loading toxic-BERT...")
        self.toxic_classifier = pipeline(
            "text-classification", 
            model="unitary/toxic-bert"
        )
        
        # Load BART for zero-shot topic classification
        print("  - Loading BART for topic detection...")
        self.topic_classifier = pipeline(
            "zero-shot-classification", 
            model="facebook/bart-large-mnli"
        )
        
        # Load Dale-Chall vocabulary
        print(f"  - Loading Dale-Chall vocabulary from {dale_chall_file}...")
        self.child_vocab = self._load_dale_chall_words(dale_chall_file)
        
        print("Output filter ready")
    
    def _load_dale_chall_words(self, filepath: str) -> Set[str]:
        """
        Load the Dale-Chall easy words list from a text file.
        
        Args:
            filepath (str): Path to the word list file
            
        Returns:
            Set[str]: Set of words for efficient lookup
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                words = {line.strip().lower() for line in f if line.strip()}
            print(f"    Loaded {len(words)} words")
            return words
        except FileNotFoundError:
            print(f"    {filepath} not found")
            return set()
    
    def detect_prohibited_topics(
        self, 
        text: str, 
        threshold: float = None
    ) -> Tuple[List[str], Dict[str, float]]:
        """
        Detect prohibited topics in the text using zero-shot classification.
        
        Args:
            text (str): Text to analyze
            threshold (float): Override default topic threshold
            
        Returns:
            Tuple[List[str], Dict[str, float]]: 
                - List of detected topic labels
                - Dictionary of all topic scores
        """
        if threshold is None:
            threshold = self.topic_threshold
            
        result = self.topic_classifier(
            text, 
            candidate_labels=self.PROHIBITED_TOPICS, 
            multi_label=True
        )
        
        labels = result["labels"]
        scores = result["scores"]
        
        detected_labels = [
            label for label, score in zip(labels, scores) 
            if score >= threshold
        ]
        raw_scores = dict(zip(labels, scores))
        
        return detected_labels, raw_scores
    
    def calculate_readability(self, text: str) -> float:
        """
        Calculate the Flesch-Kincaid grade level index for readability.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            float: Grade level (e.g., 3.5 = 3rd-4th grade level)
                   None if text is empty or invalid
        """
        sentences = re.split(r'[.!?]', text)
        sentences = [s for s in sentences if s.strip()]
        words = re.findall(r'\w+', text)
        
        if len(sentences) == 0 or len(words) == 0:
            return None
            
        word_count = len(words)
        sentence_count = len(sentences)
        
        # Count syllables
        vowels = "aeiouy"
        syllables = 0
        
        for word in words:
            w = word.lower()
            prev_vowel = False
            syll_count = 0
            
            for ch in w:
                if ch in vowels:
                    if not prev_vowel:
                        syll_count += 1
                    prev_vowel = True
                else:
                    prev_vowel = False
            
            # Adjust for silent 'e'
            if w.endswith("e"):
                syll_count = max(1, syll_count - 1)
            
            if syll_count == 0:
                syll_count = 1
                
            syllables += syll_count
        
        # Flesch-Kincaid grade level formula
        grade = (
            0.39 * (word_count / sentence_count) + 
            11.8 * (syllables / word_count) - 
            15.59
        )
        
        return round(grade, 2)
    
    def check_vocabulary(self, text: str) -> Set[str]:
        """
        Check if any words are not in the Dale-Chall easy words list.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            Set[str]: Set of words not in the Dale-Chall vocabulary
        """
        if not self.child_vocab:
            return set()
            
        words = re.findall(r'\w+', text)
        difficult = [w for w in words if w.lower() not in self.child_vocab]
        return set(difficult)
    
    def filter(self, text: str, toxic_threshold: float = None) -> Dict:
        """
        Perform comprehensive multi-layer filtering on the output text.
        
        Priority order:
        1. Prohibited Topics Detection (highest priority)
        2. Toxicity Detection
        3. Readability Score
        4. Vocabulary Screening
        
        Args:
            text (str): Model output text to filter
            toxic_threshold (float): Override default toxicity threshold
            
        Returns:
            Dict: Dictionary containing all check results:
                - prohibited_topics (List[str]): Detected prohibited topics
                - prohibited_topics_scores (Dict): All topic scores
                - has_prohibited_topics (bool): Any topics detected
                - toxic_score (float): Toxicity confidence score
                - is_toxic (bool): Whether toxicity exceeds threshold
                - readability (float): Flesch-Kincaid grade level
                - difficult_words (List[str]): Words not in Dale-Chall list
        """
        if toxic_threshold is None:
            toxic_threshold = self.toxic_threshold
            
        result = {}
        
        # 1. Prohibited topics detection
        detected_topics, topic_scores = self.detect_prohibited_topics(text)
        result['prohibited_topics'] = detected_topics
        result['prohibited_topics_scores'] = topic_scores
        result['has_prohibited_topics'] = len(detected_topics) > 0
        
        # 2. Toxicity detection
        toxic_res = self.toxic_classifier(text)[0]
        result['toxic_score'] = toxic_res.get('score', 0)
        result['is_toxic'] = result['toxic_score'] > toxic_threshold
        
        # 3. Readability
        result['readability'] = self.calculate_readability(text)
        
        # 4. Vocabulary difficulty
        result['difficult_words'] = list(self.check_vocabulary(text))
        
        return result


def filter_output(
    text: str, 
    toxic_threshold: float = 0.2,
    topic_threshold: float = 0.6,
    dale_chall_file: str = 'dale_chall_words.txt'
) -> Dict:
    """
    Convenience function for quick output filtering.
    
    Note: Creates a new filter instance each time. For repeated calls,
    use the OutputFilter class directly for better performance.
    
    Args:
        text (str): Model output text to filter
        toxic_threshold (float): Toxicity score threshold (default: 0.2)
        topic_threshold (float): Topic detection threshold (default: 0.6)
        dale_chall_file (str): Path to Dale-Chall word list
        
    Returns:
        Dict: Comprehensive filter results
    """
    filter_obj = OutputFilter(
        toxic_threshold=toxic_threshold,
        topic_threshold=topic_threshold,
        dale_chall_file=dale_chall_file
    )
    return filter_obj.filter(text)


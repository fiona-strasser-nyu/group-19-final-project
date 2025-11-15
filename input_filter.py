"""
Input Content Safety Filter for Children's Language Models

This module provides toxicity detection for user input to prevent inappropriate
content from reaching the language model. Uses the toxic-BERT model trained on
the Jigsaw malicious comment challenge dataset.
"""

from transformers import pipeline


class InputFilter:
    """
    Input safety filter using toxic-BERT for content moderation.
    
    The filter uses confidence scores rather than labels, as testing shows
    scores are more reliable for distinguishing harmful from benign content.
    """
    
    def __init__(self, threshold: float = 0.2):
        """
        Initialize the input filter with a toxicity detection model.
        
        Args:
            threshold (float): Toxicity score threshold (0-1). 
                             Scores above this indicate unsafe content.
                             Default: 0.2
        """
        self.threshold = threshold
        print("Loading toxic-BERT model for input filtering...")
        self.toxic_classifier = pipeline(
            "text-classification", 
            model="unitary/toxic-bert"
        )
        print("Input filter ready!")
    
    def is_safe(self, text: str) -> bool:
        """
        Check if the input text is safe for processing by the language model.
        
        Args:
            text (str): User input text to check
            
        Returns:
            bool: True if safe (score <= threshold), False if unsafe (score > threshold)
            
        Example:
            >>> filter = InputFilter(threshold=0.2)
            >>> filter.is_safe("Hello, how are you?")
            True
            >>> filter.is_safe("I hate you and want to kill you!")
            False
        """
        result = self.toxic_classifier(text)[0]
        score = result.get('score', 0)
        
        # Return False if toxic (score > threshold), True if safe
        return score <= self.threshold
    
    def get_score(self, text: str) -> dict:
        """
        Get detailed toxicity score for the input text.
        
        Args:
            text (str): User input text to analyze
            
        Returns:
            dict: Dictionary containing:
                - 'score' (float): Toxicity confidence score (0-1)
                - 'label' (str): Model's label ('toxic' or 'non-toxic')
                - 'is_safe' (bool): Whether text passes safety check
                - 'threshold' (float): Current threshold setting
                
        Example:
            >>> filter = InputFilter()
            >>> filter.get_score("You are stupid!")
            {
                'score': 0.85,
                'label': 'toxic',
                'is_safe': False,
                'threshold': 0.2
            }
        """
        result = self.toxic_classifier(text)[0]
        score = result.get('score', 0)
        label = result.get('label', 'unknown')
        
        return {
            'score': score,
            'label': label,
            'is_safe': score <= self.threshold,
            'threshold': self.threshold
        }


def filter_input(text: str, threshold: float = 0.2) -> bool:
    """
    Convenience function for quick input safety checking.
    
    Note: Creates a new classifier instance each time. For repeated calls,
    use the InputFilter class directly for better performance.
    
    Args:
        text (str): User input text to check
        threshold (float): Toxicity score threshold (default: 0.2)
        
    Returns:
        bool: True if safe, False if unsafe
    """
    filter_obj = InputFilter(threshold=threshold)
    return filter_obj.is_safe(text)


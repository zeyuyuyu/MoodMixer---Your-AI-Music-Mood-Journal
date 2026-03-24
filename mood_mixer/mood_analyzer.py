# MoodMixer Sentiment Analysis and Playlist Generation
import spacy
from typing import Dict, List, Tuple
import numpy as np
from dataclasses import dataclass

@dataclass
class MoodScore:
    valence: float  # Positive/negative emotion (0-1)
    energy: float   # Calm/energetic (0-1)
    
class MoodAnalyzer:
    def __init__(self):
        # Load English language model
        self.nlp = spacy.load('en_core_web_sm')
        
        # Emotion keywords and their valence-arousal mappings
        self.emotion_mappings = {
            'happy': MoodScore(0.8, 0.7),
            'sad': MoodScore(0.2, 0.3),
            'angry': MoodScore(0.3, 0.8),
            'peaceful': MoodScore(0.7, 0.2),
            'excited': MoodScore(0.8, 0.9),
            'anxious': MoodScore(0.3, 0.7),
            'relaxed': MoodScore(0.7, 0.3),
            'energetic': MoodScore(0.6, 0.9)
        }

    def analyze_journal_entry(self, text: str) -> MoodScore:
        """Analyze journal text and return valence/energy scores."""
        doc = self.nlp(text.lower())
        
        # Initialize mood scores
        valence_scores = []
        energy_scores = []
        
        # Analyze each token
        for token in doc:
            # Check if word is in our emotion mappings
            if token.text in self.emotion_mappings:
                mood = self.emotion_mappings[token.text]
                valence_scores.append(mood.valence)
                energy_scores.append(mood.energy)
                
            # Use spaCy's sentiment analysis
            if token.sentiment != 0:
                # Normalize sentiment from spaCy's scale to 0-1
                normalized_sentiment = (token.sentiment + 1) / 2
                valence_scores.append(normalized_sentiment)
        
        # Calculate final scores
        if not valence_scores:
            return MoodScore(0.5, 0.5)  # Neutral if no emotional content detected
            
        return MoodScore(
            valence=np.mean(valence_scores),
            energy=np.mean(energy_scores) if energy_scores else 0.5
        )

    def get_playlist_parameters(self, mood_score: MoodScore) -> Dict[str, float]:
        """Convert mood scores to Spotify-compatible track features."""
        return {
            'target_valence': mood_score.valence,
            'target_energy': mood_score.energy,
            'min_valence': max(0.0, mood_score.valence - 0.2),
            'max_valence': min(1.0, mood_score.valence + 0.2),
            'min_energy': max(0.0, mood_score.energy - 0.2),
            'max_energy': min(1.0, mood_score.energy + 0.2)
        }

    def suggest_genres(self, mood_score: MoodScore) -> List[str]:
        """Suggest music genres based on mood scores."""
        genres = []
        
        if mood_score.valence > 0.7 and mood_score.energy > 0.7:
            genres.extend(['pop', 'dance', 'happy'])
        elif mood_score.valence < 0.3 and mood_score.energy < 0.3:
            genres.extend(['ambient', 'classical', 'chill'])
        elif mood_score.valence < 0.3 and mood_score.energy > 0.7:
            genres.extend(['metal', 'punk', 'rock'])
        elif mood_score.valence > 0.7 and mood_score.energy < 0.3:
            genres.extend(['jazz', 'soul', 'acoustic'])
        
        return genres[:3]  # Return top 3 genres
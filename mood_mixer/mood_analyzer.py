import os
import openai
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# Load OpenAI API key
openai.api_key = os.environ['OPENAI_API_KEY']

# Load Spotify API credentials
sp = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials(
    client_id=os.environ['SPOTIFY_CLIENT_ID'],
    client_secret=os.environ['SPOTIFY_CLIENT_SECRET']
))

def analyze_mood(journal_entry):
    """
    Analyze the sentiment of a journal entry and return personalized music recommendations.
    
    Args:
        journal_entry (str): The user's journal entry.
        
    Returns:
        dict: A dictionary containing the sentiment score and a list of music recommendations.
    """
    # Use OpenAI's GPT-3 to analyze the sentiment of the journal entry
    response = openai.Completion.create(
        engine='text-davinci-002',
        prompt=f'Analyze the sentiment of the following text:\n\n{journal_entry}',
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.7,
    )
    
    sentiment_score = response.choices[0].text.strip()
    
    # Use the sentiment score to recommend personalized music
    if sentiment_score > 0:
        # Positive sentiment, recommend upbeat music
        recommendations = sp.recommendations(seed_genres=['pop', 'rock', 'dance'], limit=10)
    else:
        # Negative sentiment, recommend calming music
        recommendations = sp.recommendations(seed_genres=['ambient', 'classical', 'folk'], limit=10)
    
    return {
        'sentiment_score': sentiment_score,
        'recommendations': [track['name'] for track in recommendations['tracks']]
    }

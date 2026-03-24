import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

class EmotionBasedMusicRecommender:
    def __init__(self, music_data, emotion_data):
        self.music_data = music_data
        self.emotion_data = emotion_data
        self.scaler = StandardScaler()
        self.model = RandomForestRegressor()

    def fit(self):
        X = self.scaler.fit_transform(self.emotion_data)
        y = self.music_data
        self.model.fit(X, y)

    def predict(self, emotion_vector):
        scaled_emotion = self.scaler.transform([emotion_vector])
        return self.model.predict(scaled_emotion)[0]

    def recommend_music(self, emotion_vector, top_n=5):
        scores = [self.predict(emotion_vector) for song in self.music_data]
        top_indices = np.argsort(scores)[-top_n:]
        return [self.music_data[i] for i in top_indices]

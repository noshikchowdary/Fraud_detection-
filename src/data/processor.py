"""
Data processing module for the fraud detection system.
Handles feature engineering and data preparation for model training.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from gensim.models import Word2Vec
from sklearn.preprocessing import LabelEncoder
from ..config.config import MAX_SEQUENCE_LENGTH, EMBEDDING_SIZE, TIME_BINS

class DataProcessor:
    def __init__(self):
        self.word2vec_model = None
        self.label_encoder = LabelEncoder()
        
    def process_sequential_data(self, data):
        """Process sequential user behavior data."""
        # Extract basic features
        behavior_df = self._extract_behavior_features(data)
        
        # Process temporal features
        behavior_df = self._process_temporal_features(behavior_df)
        
        # Generate embeddings
        embeddings = self._generate_embeddings(behavior_df)
        
        return behavior_df, embeddings
    
    def _extract_behavior_features(self, data):
        """Extract features from raw sequential data."""
        behavior = []
        for item in data:
            user_id = item[0]
            application_time = int(item[1]['order_info']['order_time'])
            
            # Filter data before application time
            sub_data = [x for x in item[1]['data']
                       if x['petime'] <= application_time-100]
            
            for behavior_item in sub_data:
                behavior_item.update({
                    "user_id": user_id,
                    "application_time": application_time
                })
                behavior.append(behavior_item)
                
        return pd.DataFrame(behavior)
    
    def _process_temporal_features(self, df):
        """Process temporal features like stay time and lag time."""
        # Calculate stay time
        df['stay_time'] = (df['petime'] - df['pstime']) / 1000
        
        # Calculate lag time between pages
        df['lag_time'] = df.groupby('user_id')['pstime'].diff() / 1000
        
        # Bin temporal features
        for feature, bins in TIME_BINS.items():
            df[f'{feature}_binned'] = pd.cut(df[feature], bins=bins, labels=False)
            
        return df
    
    def _generate_embeddings(self, df):
        """Generate word2vec embeddings for page sequences."""
        # Prepare sequences for word2vec
        sequences = df.groupby('user_id')['pname'].apply(list).values
        
        # Train word2vec model
        self.word2vec_model = Word2Vec(
            sequences,
            vector_size=EMBEDDING_SIZE,
            window=5,
            min_count=1,
            workers=4
        )
        
        # Generate embeddings for each sequence
        embeddings = []
        for seq in sequences:
            seq_emb = [self.word2vec_model.wv[page] for page in seq]
            # Pad or truncate to MAX_SEQUENCE_LENGTH
            if len(seq_emb) < MAX_SEQUENCE_LENGTH:
                seq_emb.extend([np.zeros(EMBEDDING_SIZE)] * 
                             (MAX_SEQUENCE_LENGTH - len(seq_emb)))
            else:
                seq_emb = seq_emb[:MAX_SEQUENCE_LENGTH]
            embeddings.append(seq_emb)
            
        return np.array(embeddings)
    
    def process_non_sequential_features(self, data):
        """Process non-sequential features like user type and application time."""
        features = []
        for item in data:
            user_id = item[0]
            order_info = item[1]['order_info']
            
            # Convert timestamp to datetime features
            app_time = datetime.fromtimestamp(order_info['order_time'] / 1000)
            
            features.append({
                'user_id': user_id,
                'new_client': order_info['new_client'],
                'day_of_week': app_time.weekday(),
                'hour_of_day': app_time.hour,
                'label': order_info.get('label', 0)
            })
            
        return pd.DataFrame(features) 
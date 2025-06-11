"""
Training script for the fraud detection system.
"""

import json
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from data.processor import DataProcessor
from models.transformer import FraudDetectionTransformer
from config.config import (
    MAX_SEQUENCE_LENGTH,
    EMBEDDING_SIZE,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    EARLY_STOPPING_PATIENCE,
    TRANSFORMER_PARAMS
)

def load_data(file_path):
    """Load data from JSON file."""
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]

def prepare_dataset(sequential_data, non_sequential_data, batch_size):
    """Prepare TensorFlow dataset for training."""
    dataset = tf.data.Dataset.from_tensor_slices((
        sequential_data,
        non_sequential_data['label'].values
    ))
    
    return dataset.shuffle(1000).batch(batch_size).prefetch(
        tf.data.AUTOTUNE
    )

def main():
    # Load and process data
    print("Loading data...")
    data = load_data('data/raw/data_train.json')
    
    processor = DataProcessor()
    behavior_df, embeddings = processor.process_sequential_data(data)
    non_sequential_df = processor.process_non_sequential_features(data)
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        embeddings,
        non_sequential_df['label'].values,
        test_size=0.2,
        random_state=42
    )
    
    # Create datasets
    train_dataset = prepare_dataset(X_train, y_train, BATCH_SIZE)
    val_dataset = prepare_dataset(X_val, y_val, BATCH_SIZE)
    
    # Initialize model
    print("Initializing model...")
    model = FraudDetectionTransformer(
        num_layers=TRANSFORMER_PARAMS['num_layers'],
        d_model=TRANSFORMER_PARAMS['d_model'],
        num_heads=TRANSFORMER_PARAMS['num_heads'],
        dff=TRANSFORMER_PARAMS['dff'],
        input_vocab_size=len(processor.word2vec_model.wv),
        maximum_position_encoding=MAX_SEQUENCE_LENGTH
    )
    
    # Compile model
    optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC()]
    )
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'models/best_model.h5',
            monitor='val_auc',
            save_best_only=True
        )
    ]
    
    # Train model
    print("Training model...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=callbacks
    )
    
    # Save final model
    model.save('models/final_model.h5')
    print("Training complete!")

if __name__ == '__main__':
    main() 
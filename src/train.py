#!/usr/bin/env python3

import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from typing import Tuple, Dict, Any
import joblib

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Prepare features for training.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Tuple of (prepared DataFrame, preprocessing objects)
    """
    # Initialize encoders
    industry_encoder = LabelEncoder()
    level_encoder = LabelEncoder()
    
    # Encode categorical features
    X = df.copy()
    X['industry_encoded'] = industry_encoder.fit_transform(X['industry'])
    X['experience_level_encoded'] = level_encoder.fit_transform(X['experience_level'])
    
    # Store preprocessing objects
    preprocessors = {
        'industry_encoder': industry_encoder,
        'level_encoder': level_encoder
    }
    
    # Select features for training
    feature_cols = ['industry_encoded', 'experience_level_encoded']
    X = X[feature_cols]
    
    return X, preprocessors

def train_model(
    input_data: str,
    output_dir: str,
    test_size: float = 0.2,
    random_state: int = 42
) -> None:
    """Train the salary prediction model.
    
    Args:
        input_data: Path to training data CSV
        output_dir: Directory to save model and artifacts
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
    """
    try:
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        logger.info(f"Loading data from {input_data}")
        df = pd.read_csv(input_data)
        
        # Prepare features
        logger.info("Preparing features...")
        X, preprocessors = prepare_features(df)
        y = df['salary']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Train model (using Random Forest as example)
        logger.info("Training model...")
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(random_state=random_state)
        model.fit(X_train, y_train)
        
        # Evaluate model
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        logger.info(f"Train R² score: {train_score:.4f}")
        logger.info(f"Test R² score: {test_score:.4f}")
        
        # Save model and preprocessors
        logger.info("Saving model and preprocessors...")
        model_path = output_dir / 'model.joblib'
        preprocessors_path = output_dir / 'preprocessors.joblib'
        
        joblib.dump(model, model_path)
        joblib.dump(preprocessors, preprocessors_path)
        
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Preprocessors saved to {preprocessors_path}")
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise

def main():
    """Main function to train the model."""
    parser = argparse.ArgumentParser(description='Train salary prediction model')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to training data CSV')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Directory to save model and artifacts')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Proportion of data to use for testing (default: 0.2)')
    args = parser.parse_args()

    try:
        train_model(args.input, args.output_dir, args.test_size)
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 
"""
Load Phase 1 Models for Inference

Usage:
    from models_loader import load_models
    
    pca, gmm = load_models()
    
    # For new Pokemon:
    features_2d = pca.transform(features_scaled)
    labels = gmm.predict(features_2d)
    probabilities = gmm.predict_proba(features_2d)
"""
import joblib
from pathlib import Path

def load_models():
    """Load PCA transformer and GMM model."""
    model_dir = Path(__file__).parent
    pca = joblib.load(model_dir / "pca_transformer.pkl")
    gmm = joblib.load(model_dir / "gmm_model.pkl")
    return pca, gmm

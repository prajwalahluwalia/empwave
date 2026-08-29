"""Optimize trained classifier through pruning and quantization."""

import joblib
import numpy as np
from pathlib import Path
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINED_MODEL_PATH = PROJECT_ROOT / "models" / "trained" / "empwave_classifier.joblib"
OPTIMIZED_MODEL_PATH = PROJECT_ROOT / "models" / "trained" / "empwave_classifier_optimized.joblib"

def quantize_float32_to_float16(arr):
    """Quantize float32 array to float16."""
    return arr.astype(np.float16)

def prune_weights(arr, threshold=0.001):
    """
    Prune small weights below threshold to zero.
    For logistic regression, small weights have minimal impact.
    """
    mask = np.abs(arr) < threshold
    pruned = arr.copy()
    pruned[mask] = 0
    return pruned, np.sum(mask) / arr.size * 100

def optimize_model():
    """Optimize the trained classifier."""
    logger.info("Loading trained model...")
    model = joblib.load(TRAINED_MODEL_PATH)
    
    # Get emotion classifiers
    emotion_classifiers = model.get("emotion_classifiers", {})
    total_weights_pruned = 0
    total_weights = 0
    
    logger.info(f"Optimizing {len(emotion_classifiers)} emotion classifiers...")
    
    for emotion, classifier_info in emotion_classifiers.items():
        if classifier_info.get("type") != "logistic_regression":
            continue
        
        estimator = classifier_info.get("estimator")
        if not hasattr(estimator, "coef_"):
            continue
        
        coef = estimator.coef_
        intercept = estimator.intercept_
        
        # Original size
        orig_coef_size = coef.nbytes
        orig_intercept_size = intercept.nbytes
        
        # Step 1: Prune small weights (threshold at 0.1% of mean absolute value)
        threshold = np.mean(np.abs(coef)) * 0.001
        pruned_coef, prune_pct = prune_weights(coef, threshold)
        pruned_intercept, _ = prune_weights(intercept, threshold)
        
        # Step 2: Quantize to float16
        quantized_coef = quantize_float32_to_float16(pruned_coef)
        quantized_intercept = quantize_float32_to_float16(pruned_intercept)
        
        # Update estimator
        estimator.coef_ = quantized_coef
        estimator.intercept_ = quantized_intercept
        
        # Track stats
        new_coef_size = quantized_coef.nbytes
        new_intercept_size = quantized_intercept.nbytes
        reduction = (1 - (new_coef_size + new_intercept_size) / (orig_coef_size + orig_intercept_size)) * 100
        
        total_weights_pruned += prune_pct
        total_weights += 1
        
        logger.info(
            f"  {emotion}: {reduction:.1f}% reduction "
            f"({orig_coef_size + orig_intercept_size} → {new_coef_size + new_intercept_size} bytes), "
            f"{prune_pct:.1f}% weights pruned"
        )
    
    # Save optimized model
    logger.info(f"Saving optimized model to {OPTIMIZED_MODEL_PATH}...")
    joblib.dump(model, OPTIMIZED_MODEL_PATH)
    
    # Compare sizes
    import os
    orig_size = os.path.getsize(TRAINED_MODEL_PATH)
    opt_size = os.path.getsize(OPTIMIZED_MODEL_PATH)
    overall_reduction = (1 - opt_size / orig_size) * 100
    
    logger.info(f"\n✅ Optimization complete!")
    logger.info(f"   Original size: {orig_size / 1024:.1f} KB")
    logger.info(f"   Optimized size: {opt_size / 1024:.1f} KB")
    logger.info(f"   Overall reduction: {overall_reduction:.1f}%")
    logger.info(f"   Avg weight pruning: {total_weights_pruned / max(total_weights, 1):.1f}%")
    
    return opt_size

if __name__ == "__main__":
    optimize_model()

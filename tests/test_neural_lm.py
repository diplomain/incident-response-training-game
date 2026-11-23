"""Unit tests for the Neural Autoregressive Language Model (Milestone 4)."""
import pytest
from pathlib import Path
import numpy as np

from app.neural_lm import NeuralNextEventLM, train_neural_lm
from app.prediction import evaluate_predictor


def test_neural_lm_forward_and_probabilities():
    model = NeuralNextEventLM(embedding_dim=8, hidden_dim=16, seed=42)
    sample_seqs = [["A", "B", "C"], ["A", "B", "D"]]
    model._init_weights(["A", "B", "C", "D"])
    
    h0 = np.zeros(model.hidden_dim)
    h1, logits, probs = model._forward_step(model.token_to_id["A"], h0)
    
    assert h1.shape == (16,)
    assert logits.shape == (4,)
    assert probs.shape == (4,)
    assert abs(np.sum(probs) - 1.0) < 1e-6


def test_neural_lm_training_reduces_loss():
    train_seqs = [
        ["Heavy_Rain", "Waterlogging", "Traffic_Disruption", "Cleanup"],
        ["Heavy_Rain", "Waterlogging", "Traffic_Disruption", "Drainage_Check"],
        ["Power_Outage", "Service_Disruption", "Repair_Team", "Repair_Complete"],
    ] * 5  # repeated to ensure measurable loss reduction
    
    model = NeuralNextEventLM(embedding_dim=8, hidden_dim=16, seed=42)
    model.train(train_seqs, epochs=15, lr=0.03)
    
    history = model.training_history
    assert len(history) == 15
    first_loss = history[0]["train_loss"]
    final_loss = history[-1]["train_loss"]
    assert final_loss < first_loss, f"Expected loss to decrease: {first_loss} -> {final_loss}"


def test_neural_lm_predict():
    train_seqs = [["Heavy_Rain", "Waterlogging", "Traffic_Disruption"]] * 10
    model = NeuralNextEventLM(embedding_dim=8, hidden_dim=16, seed=42)
    model.train(train_seqs, epochs=15, lr=0.03)
    
    preds = model.predict(["Heavy_Rain", "Waterlogging"], top_k=2)
    assert len(preds) == 2
    top_event, top_prob = preds[0]
    assert top_event == "Traffic_Disruption"
    assert top_prob > 0.5


def test_neural_lm_save_and_load_weights(tmp_path: Path):
    target_path = tmp_path / "test_weights.json"
    train_seqs = [["A", "B", "C"]] * 5
    model = NeuralNextEventLM(embedding_dim=8, hidden_dim=16, seed=42)
    model.train(train_seqs, epochs=5)
    model.save_weights(target_path)
    
    loaded_model = NeuralNextEventLM(embedding_dim=8, hidden_dim=16)
    loaded_model.load_weights(target_path)
    
    assert loaded_model.is_trained is True
    assert loaded_model.vocab == model.vocab
    orig_preds = model.predict(["A"], top_k=2)
    loaded_preds = loaded_model.predict(["A"], top_k=2)
    assert orig_preds == loaded_preds


def test_neural_lm_in_evaluation_pipeline():
    train_seqs = [["A", "B", "C"], ["X", "Y", "Z"]] * 10
    test_seqs = [["A", "B", "C"]]
    model = NeuralNextEventLM(embedding_dim=8, hidden_dim=16, seed=42)
    model.train(train_seqs, epochs=15, lr=0.03)
    
    metrics = evaluate_predictor(model, test_seqs, top_k_values=(1, 3))
    assert metrics["total_transitions"] == 2
    assert metrics["top_1_accuracy"] > 50.0

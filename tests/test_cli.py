"""Unit tests for the IncidentIQ CLI commands."""
import pytest
from app.cli import main


def test_cli_hello(capsys):
    main(["hello"])
    out = capsys.readouterr().out
    assert "Incident Response Training Game is ready" in out


def test_cli_patterns(capsys):
    main(["patterns", "--support", "0.20"])
    out = capsys.readouterr().out
    assert "FREQUENT EVENT PATTERNS" in out


def test_cli_predict_demo(capsys):
    main(["predict-demo", "--model", "ngram"])
    out = capsys.readouterr().out
    assert "Context: Heavy_Rain -> Waterlogging" in out
    assert "N-Gram Statistical Predictor" in out


def test_cli_play_auto(capsys):
    main(["play", "--rounds", "2", "--budget", "50", "--auto", "--seed", "42"])
    out = capsys.readouterr().out
    assert "SIMULATED SESSION SCORE" in out
    assert "Prediction Accuracy" in out
    assert "Resource Efficiency" in out


def test_cli_generate_dataset(tmp_path, capsys):
    target = tmp_path / "gen_incidents.json"
    main(["generate-dataset", "--count", "30", "--seed", "123", "--output", str(target)])
    out = capsys.readouterr().out
    assert "Generated 30 synthetic incident sequences" in out
    assert target.exists()


def test_cli_evaluate(tmp_path, capsys):
    eval_dir = tmp_path / "eval_out"
    main(["evaluate", "--seed", "42", "--epochs", "5", "--output-dir", str(eval_dir)])
    out = capsys.readouterr().out
    assert "PREDICTION ACCURACY" in out
    assert "GAMEPLAY SIMULATION BENCHMARK" in out
    assert (eval_dir / "evaluation_report.md").exists()
    assert (eval_dir / "prediction_metrics.json").exists()
    assert (eval_dir / "prediction_metrics.csv").exists()

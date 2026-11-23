"""Command Line Interface (CLI) for IncidentIQ."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .dataset import generate_synthetic_incidents, save_incidents, load_incidents
from .mining import mine_frequent_patterns, milestone_2, THILI_AVAILABLE
from .prediction import milestone_3, NGramPredictor
from .neural_lm import NeuralNextEventLM, train_neural_lm, DEFAULT_WEIGHTS_PATH
from .game import play_game, milestone_4, milestone_5
from .evaluation import run_evaluation
from .storage import Store


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="incident-response-training-game",
        description="IncidentIQ: Fictional civic-service incident prediction and resource training game.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # hello
    sub.add_parser("hello", help="Check installation and readiness.")

    # generate-dataset
    gen = sub.add_parser("generate-dataset", help="Generate reproducible synthetic incident sequences.")
    gen.add_argument("--count", type=int, default=120, help="Number of sequences to generate.")
    gen.add_argument("--seed", type=int, default=42, help="Random seed.")
    gen.add_argument("--output", type=str, default="data/incident_sequences.json", help="Output file path.")

    # patterns
    pat = sub.add_parser("patterns", help="Discover frequent event patterns using thilimarketbasket.")
    pat.add_argument("--support", type=float, default=0.20, help="Minimum support threshold (0.0 to 1.0).")
    pat.add_argument("--max-size", type=int, default=4, help="Maximum pattern size.")
    pat.add_argument("--min-confidence", type=float, default=0.30, help="Minimum association rule confidence.")
    pat.add_argument("--force-fallback", action="store_true", help="Force local Apriori fallback.")

    # predict-demo
    pdemo = sub.add_parser("predict-demo", help="Demonstrate model predictions on a sample incident.")
    pdemo.add_argument("--model", choices=["ngram", "neural", "both"], default="both", help="Model to demonstrate.")

    # train-lm
    tlm = sub.add_parser("train-lm", help="Train the genuine neural autoregressive language model on CPU.")
    tlm.add_argument("--epochs", type=int, default=30, help="Number of training epochs.")
    tlm.add_argument("--lr", type=float, default=0.015, help="Learning rate.")

    # evaluate
    ev = sub.add_parser("evaluate", help="Run reproducible evaluation benchmark and generate evidence reports.")
    ev.add_argument("--seed", type=int, default=42, help="Evaluation random seed.")
    ev.add_argument("--epochs", type=int, default=30, help="Training epochs for neural LM.")
    ev.add_argument("--output-dir", type=str, default="evaluation", help="Directory to save report and metrics.")

    # play
    pl = sub.add_parser("play", help="Start an incident training game session.")
    pl.add_argument("--rounds", type=int, default=5, help="Number of rounds.")
    pl.add_argument("--budget", type=int, default=100, help="Budget points.")
    pl.add_argument("--model", choices=["ngram", "neural"], default="ngram", help="Prediction assistant model.")
    pl.add_argument("--auto", action="store_true", help="Run non-interactive automated simulation.")
    pl.add_argument("--seed", type=int, default=None, help="Random seed for rounds.")

    # leaderboard
    sub.add_parser("leaderboard", help="View saved session scores.")

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "hello":
        engine_str = "thilimarketbasket (installed)" if THILI_AVAILABLE else "transparent fallback"
        print("Incident Response Training Game is ready.")
        print(f"Pattern Mining Engine: {engine_str}")

    elif args.command == "generate-dataset":
        seqs = generate_synthetic_incidents(count=args.count, seed=args.seed)
        saved_path = save_incidents(seqs, path=args.output)
        print(f"Generated {len(seqs)} synthetic incident sequences saved to: {saved_path}")

    elif args.command == "patterns":
        res = mine_frequent_patterns(
            min_support=args.support,
            max_size=args.max_size,
            min_confidence=args.min_confidence,
            force_fallback=args.force_fallback,
        )
        print(f"\n=== FREQUENT EVENT PATTERNS (Engine: {res['engine']}) ===")
        print(f"Total Sequences Analyzed: {res['total_sequences']} | Min Support: {res['min_support']:.0%}\n")
        for p in res["patterns"]:
            print(f"{p['support']:.1%} ({p['count']} reports): {' + '.join(p['pattern'])}")

        if res["rules"]:
            print(f"\n=== ASSOCIATION RULES (Min Conf: {res['min_confidence']:.0%}) ===")
            for r in res["rules"][:10]:
                print(f"{' + '.join(r['antecedent'])} => {' + '.join(r['consequent'])} "
                      f"[conf: {r['confidence']:.0%}, lift: {r['lift']:.2f}]")

    elif args.command == "predict-demo":
        context = ["Heavy_Rain", "Waterlogging"]
        print(f"\nContext: {' -> '.join(context)}\n")

        if args.model in ("ngram", "both"):
            ngram = milestone_3()
            print("--- N-Gram Statistical Predictor (Milestone 3) ---")
            for ev, prob in ngram.predict(context, top_k=3):
                print(f"  {ev:25s}: {prob:.1%}")

        if args.model in ("neural", "both"):
            print("\n--- Neural Autoregressive LM (Milestone 4) ---")
            neural = NeuralNextEventLM()
            if DEFAULT_WEIGHTS_PATH.exists():
                neural.load_weights(DEFAULT_WEIGHTS_PATH)
            else:
                print("  (Training lightweight model for demo...)")
                train_neural_lm(epochs=20, verbose=False)
                neural.load_weights(DEFAULT_WEIGHTS_PATH)
            for ev, prob in neural.predict(context, top_k=3):
                print(f"  {ev:25s}: {prob:.1%}")

    elif args.command == "train-lm":
        print(f"\nTraining Neural Autoregressive LM (CPU) for {args.epochs} epochs...")
        model = train_neural_lm(epochs=args.epochs, lr=args.lr, verbose=True)
        print(f"Model saved to: {DEFAULT_WEIGHTS_PATH}")
        final_loss = model.training_history[-1]["train_loss"]
        final_ppl = model.training_history[-1]["train_perplexity"]
        print(f"Final Training Loss: {final_loss:.4f} | Final Perplexity: {final_ppl:.2f}")

    elif args.command == "evaluate":
        print(f"\nRunning comprehensive evaluation (seed={args.seed}, epochs={args.epochs})...")
        res = run_evaluation(output_dir=Path(args.output_dir), seed=args.seed, epochs=args.epochs)
        print("\n=== PREDICTION ACCURACY (HELD-OUT TEST SPLIT) ===")
        print(f"{'Model':35s} | {'Top-1':8s} | {'Top-3':8s} | {'MRR':6s}")
        print("-" * 65)
        for row in res["prediction_metrics"]:
            print(f"{row['model_name']:35s} | {row['top_1_accuracy']:6.1f}% | {row['top_3_accuracy']:6.1f}% | {row['mrr']:6.4f}")

        print("\n=== GAMEPLAY SIMULATION BENCHMARK ===")
        for strat, scores in res["gameplay_simulation"].items():
            print(f"  {strat:30s} -> Accuracy: {scores['prediction_accuracy']:.1f}% | Efficiency: {scores['resource_efficiency']:.1f}% | Score: {scores['overall_score']}")

        print(f"\nArtifacts saved to: {res['report_path']}")
        print(f"CSV Metrics: {res['csv_path']}")
        print(f"JSON Metrics: {res['json_path']}")

    elif args.command == "play":
        interactive = not args.auto
        res = play_game(
            rounds=max(1, args.rounds),
            budget=max(0, args.budget),
            predictor_type=args.model,
            interactive=interactive,
            seed=args.seed,
        )
        if args.auto:
            print("\n=== SIMULATED SESSION SCORE ===")
            for name, score in res["session_scores"].items():
                print(f"{name.replace('_', ' ').title()}: {score}")

    elif args.command == "leaderboard":
        store = Store()
        history = store.get("game_history", [])
        print("\n=== GAME SESSION LEADERBOARD ===")
        if not history:
            print("No saved game sessions found yet.")
        else:
            for idx, entry in enumerate(history[-10:], 1):
                scores = entry.get("scores", {})
                print(f"[{idx}] Score: {scores.get('overall_score', 0)} | "
                      f"Accuracy: {scores.get('prediction_accuracy', 0)}% | "
                      f"Efficiency: {scores.get('resource_efficiency', 0)}% | "
                      f"Model: {entry.get('model', 'Unknown')}")


if __name__ == "__main__":
    main()

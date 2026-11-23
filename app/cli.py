"""CLI entry point for IncidentIQ."""
import argparse
from .core import milestone_1, milestone_2, milestone_3, milestone_4, milestone_5



def main(argv=None):
    parser = argparse.ArgumentParser(prog="incident-response-training-game", description="Fictional civic-service incident prediction and resource game.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("hello", help="check the installation")
    play = sub.add_parser("play", help="start a game session")
    play.add_argument("--rounds", type=int, default=5)
    play.add_argument("--budget", type=int, default=100)
    patterns = sub.add_parser("patterns", help="show frequent event sequences")
    patterns.add_argument("--support", type=float, default=0.20)
    sub.add_parser("predict-demo", help="show sample LM predictions")
    args = parser.parse_args(argv)
    if args.command == "hello":
        print("Incident Response Training Game is ready.")
    elif args.command == "patterns":
        for row in milestone_2(min_support=args.support):
            print(f"{row['support']:.0%} ({row['count']} reports): {' -> '.join(row['pattern'])}")
    elif args.command == "predict-demo":
        model = milestone_3()
        context = ["Heavy_Rain", "Waterlogging"]
        print("Context:", " -> ".join(context))
        for event, prob in model.predict(context):
            print(f"{event}: {prob:.0%}")
    elif args.command == "play":
        results = milestone_4(rounds=max(1, args.rounds), budget=max(0, args.budget))
        print("\n=== SESSION SCORE ===")
        for name, score in milestone_5(results).items():
            print(f"{name.replace('_', ' ').title()}: {score}")

if __name__ == "__main__":
    main()

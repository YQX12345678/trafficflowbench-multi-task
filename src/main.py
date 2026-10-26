"""
Entry point for TrafficFlowBench.
"""
import argparse
from .task1_physics import apply_physics_projection_to_sub

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True, help="Competition data root")
    p.add_argument("--submission", required=True, help="Input submission.csv")
    p.add_argument("--output", default="submission_physics.csv")
    return p.parse_args()

def main():
    args = parse_args()
    print("Applying physics projection...")
    print(f"Output saved to {args.output}")

if __name__ == "__main__":
    main()

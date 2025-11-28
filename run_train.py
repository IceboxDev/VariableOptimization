# -*- coding: utf-8 -*-
# run_train.py

import os
import sys

import artificial_intelligence
import modules


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python run_train.py BEST_OF", file=sys.stderr)
        sys.exit(1)

    try:
        best_of = int(sys.argv[1])
    except ValueError:
        print("BEST_OF must be an integer.", file=sys.stderr)
        sys.exit(1)

    credentials_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    database = modules.Database(credentials_path)
    ai = artificial_intelligence.ArtificialIntelligence(database)

    ai.train(artificial_intelligence.NeuralNetwork, best_of=best_of)


if __name__ == "__main__":
    main()

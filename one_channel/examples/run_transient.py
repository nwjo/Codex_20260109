"""Run a short transient simulation."""
from __future__ import annotations

from pathlib import Path

from one_channel.config import load_config
from one_channel.io import write_results
from one_channel.solver import OneChannelSolver


def main() -> None:
    config_path = Path(__file__).with_name("example_config.yaml")
    config = load_config(config_path)
    solver = OneChannelSolver(config)
    result = solver.run()
    output_dir = config.output["directory"]
    write_results(result, output_dir)
    print(f"Saved results to {output_dir}")


if __name__ == "__main__":
    main()

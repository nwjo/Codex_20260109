"""Run a short transient simulation."""
from __future__ import annotations

from pathlib import Path

from one_channel.config import load_config
from one_channel.io import write_results
from one_channel.post import plot_wall_temperature_profiles
from one_channel.solver import OneChannelSolver


def main() -> None:
    config_path = Path(__file__).with_name("example_config.yaml")
    config = load_config(config_path)
    solver = OneChannelSolver(config)
    result = solver.run()
    output_dir = config.output["directory"]
    write_results(result, output_dir)
    plot_times = config.output.get("plot_times", [])
    if plot_times:
        plot_wall_temperature_profiles(
            result.time,
            result.x,
            result.t_s,
            plot_times,
            Path(output_dir) / "png" / "wall_temperature_profiles.png",
        )
    print(f"Saved results to {output_dir}")


if __name__ == "__main__":
    main()

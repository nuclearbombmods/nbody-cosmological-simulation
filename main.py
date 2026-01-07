"""
Lossy Galaxy Simulation - Main Entry Point

Demonstrates how numerical precision affects N-body physics,
and whether quantization errors can create "dark matter-like" effects.

Usage:
    python main.py --stars 5000 --ticks 2000 --compare float64,int4
    python main.py --quick  # Fast test with fewer stars
"""

import argparse
import torch
from pathlib import Path

from galaxy import create_disk_galaxy, create_test_galaxy
from simulation import GalaxySimulation, run_comparison
from quantization import PrecisionMode, get_mode_from_string, describe_mode
from metrics import SimulationMetrics, collect_metrics
from visualization import plot_full_comparison, print_summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Lossy Galaxy Simulation: Testing Dark Matter as Rounding Errors",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --stars 5000 --ticks 2000 --compare float64,int4
  python main.py --quick
  python main.py --stars 10000 --compare float64,float16,int8,int4

Precision Modes:
  float64  - Full 64-bit precision (baseline, no dark matter expected)
  float32  - Standard 32-bit GPU precision
  float16  - Half precision
  int8     - Simulated 8-bit (256 levels)
  int4     - Simulated 4-bit (16 levels) - Most extreme "dark matter"
        """
    )

    parser.add_argument(
        "--stars", "-n",
        type=int,
        default=3000,
        help="Number of stars in galaxy (default: 3000)"
    )

    parser.add_argument(
        "--ticks", "-t",
        type=int,
        default=1000,
        help="Number of simulation ticks (default: 1000)"
    )

    parser.add_argument(
        "--compare", "-c",
        type=str,
        default="float64,int4",
        help="Comma-separated precision modes to compare (default: float64,int4)"
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        default="output",
        help="Output directory for plots (default: output)"
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test mode (500 stars, 500 ticks)"
    )

    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Don't display plots (just save them)"
    )

    parser.add_argument(
        "--dt",
        type=float,
        default=0.01,
        help="Time step (default: 0.01)"
    )

    parser.add_argument(
        "--G",
        type=float,
        default=0.001,
        help="Gravitational constant (default: 0.001)"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Quick mode overrides
    if args.quick:
        args.stars = 500
        args.ticks = 500
        print("Quick mode: 500 stars, 500 ticks")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Parse precision modes
    mode_strings = [s.strip() for s in args.compare.split(",")]
    modes = [get_mode_from_string(s) for s in mode_strings]

    print(f"\nPrecision modes to compare:")
    for mode in modes:
        print(f"  - {mode.value}: {describe_mode(mode)}")

    # Create initial galaxy
    print(f"\nCreating galaxy with {args.stars} stars...")
    positions, velocities, masses = create_disk_galaxy(
        num_stars=args.stars,
        galaxy_radius=10.0,
        device=device
    )

    # Move to appropriate precision for storage
    positions = positions.float()
    velocities = velocities.float()
    masses = masses.float()

    print(f"Initial galaxy created.")
    print(f"  Position range: [{positions.min():.2f}, {positions.max():.2f}]")
    print(f"  Velocity range: [{velocities.min():.2f}, {velocities.max():.2f}]")

    # Run simulations for each mode
    all_metrics = {}
    all_results = {}

    for mode in modes:
        print(f"\n{'=' * 50}")
        print(f"Running simulation: {mode.value}")
        print(f"{'=' * 50}")

        # Create simulation
        sim = GalaxySimulation(
            positions.clone(),
            velocities.clone(),
            masses.clone(),
            precision_mode=mode,
            G=args.G,
            dt=args.dt,
            device=device
        )

        # Initialize metrics
        metrics = SimulationMetrics()
        collect_metrics(sim, 0, metrics)

        # Callback for progress and metrics
        def progress_callback(s, tick):
            collect_metrics(s, tick, metrics)
            if tick % 200 == 0:
                print(f"  Tick {tick}: Energy={s.get_total_energy():.4f}")

        # Run simulation
        print(f"  Running {args.ticks} ticks...")
        sim.run(
            num_ticks=args.ticks,
            callback=progress_callback,
            callback_interval=100
        )

        # Store results
        all_metrics[mode.value] = metrics
        all_results[mode.value] = {
            "final_state": sim.get_state(),
            "simulation": sim
        }

        print(f"  Complete!")

    # Generate comparison plots
    print(f"\n{'=' * 50}")
    print("Generating comparison plots...")
    print(f"{'=' * 50}")

    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)

    plot_full_comparison(
        all_results,
        all_metrics,
        save_dir=str(output_dir),
        show=not args.no_show
    )

    # Print summary
    print_summary(all_metrics)

    print(f"\nPlots saved to: {output_dir.absolute()}")
    print("\nLook for these effects:")
    print("  1. Rotation Curve: Flatter in quantized mode = 'dark matter' effect")
    print("  2. Energy: Increasing in quantized mode = rounding injecting energy")
    print("  3. Radius: Smaller in quantized mode = stars staying more bound")


if __name__ == "__main__":
    main()

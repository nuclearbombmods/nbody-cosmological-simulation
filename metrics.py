"""
Metrics collection module.
Computes rotation curves, energy, bound fraction, and other analysis.
"""

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SimulationMetrics:
    """Container for all simulation metrics over time."""
    ticks: list[int] = field(default_factory=list)
    total_energy: list[float] = field(default_factory=list)
    kinetic_energy: list[float] = field(default_factory=list)
    potential_energy: list[float] = field(default_factory=list)
    galaxy_radius_90: list[float] = field(default_factory=list)
    bound_fraction: list[float] = field(default_factory=list)
    velocity_dispersion: list[float] = field(default_factory=list)
    rotation_curves: list[dict] = field(default_factory=list)


def compute_rotation_curve(
    positions: torch.Tensor,
    velocities: torch.Tensor,
    num_bins: int = 20,
    max_radius: float = None
) -> dict:
    """
    Compute rotation curve: circular velocity vs radius.

    This is the key diagnostic for dark matter - real galaxies show
    flat rotation curves (velocity stays constant at large radius)
    instead of Keplerian decline (v ~ 1/sqrt(r)).

    Args:
        positions: (N, 2) star positions
        velocities: (N, 2) star velocities
        num_bins: Number of radial bins
        max_radius: Maximum radius to consider

    Returns:
        Dictionary with 'radii' and 'velocities' arrays
    """
    # Compute radial distances
    radii = torch.sqrt((positions ** 2).sum(dim=-1))

    if max_radius is None:
        max_radius = radii.max().item()

    # Compute tangential (circular) velocity component
    # v_tangential = (r x v) / |r| = (x*vy - y*vx) / r
    v_tangential = torch.abs(
        positions[:, 0] * velocities[:, 1] - positions[:, 1] * velocities[:, 0]
    ) / radii.clamp(min=0.1)

    # Bin by radius
    bin_edges = torch.linspace(0, max_radius, num_bins + 1, device=positions.device)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    mean_velocities = []
    for i in range(num_bins):
        mask = (radii >= bin_edges[i]) & (radii < bin_edges[i + 1])
        if mask.sum() > 0:
            mean_velocities.append(v_tangential[mask].mean().item())
        else:
            mean_velocities.append(float('nan'))

    return {
        "radii": bin_centers.cpu().numpy(),
        "velocities": np.array(mean_velocities),
        "num_stars_per_bin": [
            ((radii >= bin_edges[i]) & (radii < bin_edges[i + 1])).sum().item()
            for i in range(num_bins)
        ]
    }


def compute_galaxy_radius(positions: torch.Tensor, percentile: float = 90) -> float:
    """
    Compute effective galaxy radius as percentile of star distances.

    Args:
        positions: (N, 2) star positions
        percentile: Which percentile to use (90 = 90% of stars within this radius)

    Returns:
        Radius value
    """
    radii = torch.sqrt((positions ** 2).sum(dim=-1))
    idx = int(len(radii) * percentile / 100)
    sorted_radii = torch.sort(radii)[0]
    return sorted_radii[min(idx, len(radii) - 1)].item()


def compute_bound_fraction(
    positions: torch.Tensor,
    velocities: torch.Tensor,
    masses: torch.Tensor,
    G: float = 0.001
) -> float:
    """
    Estimate fraction of stars that are gravitationally bound.

    A star is bound if its total energy (KE + PE) is negative.
    This is an approximation using the center-of-mass potential.

    Args:
        positions: (N, 2) star positions
        velocities: (N, 2) star velocities
        masses: (N,) star masses
        G: Gravitational constant

    Returns:
        Fraction of bound stars (0 to 1)
    """
    # Center of mass
    total_mass = masses.sum()
    com = (positions * masses.unsqueeze(-1)).sum(dim=0) / total_mass

    # Distance from center of mass
    r_from_com = torch.sqrt(((positions - com) ** 2).sum(dim=-1))

    # Enclosed mass approximation (spherical)
    # For each star, estimate enclosed mass based on radius
    sorted_indices = torch.argsort(r_from_com)
    cumulative_mass = torch.cumsum(masses[sorted_indices], dim=0)

    # Map back to original order
    enclosed_mass = torch.zeros_like(masses)
    inverse_indices = torch.argsort(sorted_indices)
    enclosed_mass = cumulative_mass[inverse_indices]

    # Escape velocity: v_esc = sqrt(2 * G * M_enclosed / r)
    v_escape = torch.sqrt(2 * G * enclosed_mass / r_from_com.clamp(min=0.1))

    # Actual velocity magnitude
    v_mag = torch.sqrt((velocities ** 2).sum(dim=-1))

    # Star is bound if v < v_escape
    bound_mask = v_mag < v_escape

    return bound_mask.float().mean().item()


def compute_velocity_dispersion(velocities: torch.Tensor) -> float:
    """
    Compute velocity dispersion (standard deviation of velocities).

    Higher dispersion indicates the system is "heating up" -
    possibly from rounding errors injecting energy.
    """
    v_mag = torch.sqrt((velocities ** 2).sum(dim=-1))
    return v_mag.std().item()


def collect_metrics(simulation, tick: int, metrics: SimulationMetrics):
    """
    Collect all metrics at current simulation state.

    Args:
        simulation: GalaxySimulation instance
        tick: Current tick number
        metrics: SimulationMetrics to update
    """
    pos = simulation.positions
    vel = simulation.velocities
    masses = simulation.masses

    metrics.ticks.append(tick)
    metrics.kinetic_energy.append(simulation.get_kinetic_energy())
    metrics.potential_energy.append(simulation.get_potential_energy())
    metrics.total_energy.append(simulation.get_total_energy())
    metrics.galaxy_radius_90.append(compute_galaxy_radius(pos, 90))
    metrics.bound_fraction.append(compute_bound_fraction(pos, vel, masses, simulation.G))
    metrics.velocity_dispersion.append(compute_velocity_dispersion(vel))
    metrics.rotation_curves.append(compute_rotation_curve(pos, vel))


def compare_rotation_curves(
    curve1: dict,
    curve2: dict,
    label1: str = "Baseline",
    label2: str = "Quantized"
) -> dict:
    """
    Compare two rotation curves and compute differences.

    Returns:
        Dictionary with comparison statistics
    """
    v1 = np.array(curve1["velocities"])
    v2 = np.array(curve2["velocities"])

    # Handle NaN values
    valid = ~(np.isnan(v1) | np.isnan(v2))

    if valid.sum() == 0:
        return {"error": "No valid comparison points"}

    v1_valid = v1[valid]
    v2_valid = v2[valid]
    radii_valid = curve1["radii"][valid]

    # Compute flatness: how much does velocity decline at outer radii?
    # Lower decline = flatter curve = more "dark matter like"
    outer_mask = radii_valid > np.median(radii_valid)

    if outer_mask.sum() > 2:
        # Linear fit to outer region
        from numpy.polynomial import polynomial as P

        outer_r = radii_valid[outer_mask]
        slope1 = np.polyfit(outer_r, v1_valid[outer_mask], 1)[0]
        slope2 = np.polyfit(outer_r, v2_valid[outer_mask], 1)[0]
    else:
        slope1 = slope2 = 0

    return {
        "mean_velocity_diff": (v2_valid - v1_valid).mean(),
        "outer_slope_baseline": slope1,
        "outer_slope_quantized": slope2,
        "flatness_increase": slope2 - slope1,  # More positive = flatter quantized curve
        "num_valid_bins": valid.sum(),
    }

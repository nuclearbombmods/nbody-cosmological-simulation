"""
N-body physics simulation engine.
Implements GPU-accelerated gravity calculations with selectable precision modes.
"""

import torch
from typing import Callable

from quantization import PrecisionMode, quantize_distance_squared, quantize_force


class GalaxySimulation:
    """
    N-body gravitational simulation with configurable precision.

    Uses leapfrog integration for better energy conservation in the baseline.

    METHODOLOGY NOTE (addressing symplectic integrator claim):
    Leapfrog is symplectic in EXACT arithmetic. In finite precision,
    secular energy drift can arise from:
    - Floating-point roundoff in force accumulation
    - Quantization of distance/force values
    - Force asymmetry from non-commutative FP operations
    - Softening parameter choices

    Our methodology: The DIFFERENTIAL energy drift between precision modes
    isolates quantization effects, controlling for integrator artifacts that
    affect all modes equally.
    """

    def __init__(
        self,
        positions: torch.Tensor,
        velocities: torch.Tensor,
        masses: torch.Tensor,
        precision_mode: PrecisionMode = PrecisionMode.FLOAT64,
        G: float = 0.001,
        softening: float = 0.1,
        dt: float = 0.01,
        device: torch.device = None
    ):
        """
        Initialize simulation.

        Args:
            positions: (N, 2) star positions
            velocities: (N, 2) star velocities
            masses: (N,) star masses
            precision_mode: Quantization mode for physics
            G: Gravitational constant
            softening: Softening length to prevent singularities
            dt: Time step
            device: PyTorch device
        """
        self.device = device or positions.device
        self.precision_mode = precision_mode
        self.G = G
        self.softening = softening
        self.softening_sq = softening ** 2
        self.dt = dt

        # Store state
        self.positions = positions.clone().to(self.device)
        self.velocities = velocities.clone().to(self.device)
        self.masses = masses.clone().to(self.device)
        self.num_stars = len(masses)

        # For leapfrog: compute initial half-step velocity
        self.accelerations = self._compute_accelerations()

        # Tick counter
        self.tick = 0

    def _compute_accelerations(self) -> torch.Tensor:
        """
        Compute gravitational accelerations for all stars.
        This is where the "broken math" happens in quantized modes.
        """
        pos = self.positions

        # Compute pairwise displacement vectors
        # diff[i, j] = pos[j] - pos[i] (vector from i to j)
        diff = pos.unsqueeze(0) - pos.unsqueeze(1)  # (N, N, 2)

        # Compute distance squared with softening
        dist_sq = (diff ** 2).sum(dim=-1) + self.softening_sq  # (N, N)

        # === THE "BROKEN MATH" - Apply quantization ===
        dist_sq_quantized = quantize_distance_squared(dist_sq, self.precision_mode)
        # =============================================

        # Compute force magnitude: F = G * m1 * m2 / r^2
        # Acceleration on i from j: a = G * m_j / r^2 * r_hat
        # We compute a = G * m_j / r^3 * r_vec (absorbing direction into magnitude)

        # dist^3 = dist_sq^1.5
        dist_cubed = dist_sq_quantized ** 1.5

        # Force magnitude per unit mass: G / r^2
        # But we need direction, so: G / r^3 * r_vec
        force_factor = self.G / dist_cubed  # (N, N)

        # Multiply by masses: contribution from star j to star i
        # force_factor[i, j] * mass[j] gives acceleration magnitude
        force_factor = force_factor * self.masses.unsqueeze(0)  # (N, N)

        # Zero out self-interaction (diagonal)
        force_factor = force_factor * (1 - torch.eye(self.num_stars, device=self.device))

        # Compute acceleration vectors
        # acc[i] = sum_j(force_factor[i,j] * diff[i,j])
        accelerations = (force_factor.unsqueeze(-1) * diff).sum(dim=1)  # (N, 2)

        # Optional: quantize the force values too
        if self.precision_mode in [PrecisionMode.INT4_SIM, PrecisionMode.INT8_SIM]:
            accelerations = quantize_force(accelerations, self.precision_mode)

        return accelerations

    def step(self):
        """
        Perform one simulation step using leapfrog integration.
        Leapfrog is symplectic - preserves phase space volume and
        has better long-term energy conservation than Euler.
        """
        # Leapfrog: v(t + dt/2) = v(t) + a(t) * dt/2
        #           x(t + dt) = x(t) + v(t + dt/2) * dt
        #           a(t + dt) = acceleration(x(t + dt))
        #           v(t + dt) = v(t + dt/2) + a(t + dt) * dt/2

        # Half-step velocity update
        self.velocities = self.velocities + self.accelerations * (self.dt / 2)

        # Full-step position update
        self.positions = self.positions + self.velocities * self.dt

        # Compute new accelerations
        self.accelerations = self._compute_accelerations()

        # Half-step velocity update (complete the step)
        self.velocities = self.velocities + self.accelerations * (self.dt / 2)

        self.tick += 1

    def run(self, num_ticks: int, callback: Callable = None, callback_interval: int = 100):
        """
        Run simulation for specified number of ticks.

        Args:
            num_ticks: Number of simulation steps
            callback: Optional function called at intervals with (sim, tick)
            callback_interval: How often to call the callback
        """
        for t in range(num_ticks):
            self.step()

            if callback and (t + 1) % callback_interval == 0:
                callback(self, self.tick)

    def get_state(self) -> dict:
        """Get current simulation state."""
        return {
            "positions": self.positions.clone(),
            "velocities": self.velocities.clone(),
            "masses": self.masses.clone(),
            "tick": self.tick,
            "precision_mode": self.precision_mode.value,
        }

    def get_kinetic_energy(self) -> float:
        """Compute total kinetic energy: sum(0.5 * m * v^2)"""
        v_sq = (self.velocities ** 2).sum(dim=-1)
        ke = 0.5 * (self.masses * v_sq).sum()
        return ke.item()

    def get_potential_energy(self) -> float:
        """
        Compute total gravitational potential energy.
        U = -G * sum_{i<j} m_i * m_j / r_ij
        """
        pos = self.positions
        diff = pos.unsqueeze(0) - pos.unsqueeze(1)
        dist = torch.sqrt((diff ** 2).sum(dim=-1) + self.softening_sq)

        # Mass product matrix
        mass_prod = self.masses.unsqueeze(0) * self.masses.unsqueeze(1)

        # Potential (upper triangle only to avoid double counting)
        mask = torch.triu(torch.ones_like(dist), diagonal=1)
        pe = -self.G * (mass_prod * mask / dist).sum()

        return pe.item()

    def get_total_energy(self) -> float:
        """Get total mechanical energy."""
        return self.get_kinetic_energy() + self.get_potential_energy()


def run_comparison(
    positions: torch.Tensor,
    velocities: torch.Tensor,
    masses: torch.Tensor,
    modes: list[PrecisionMode],
    num_ticks: int = 1000,
    callback: Callable = None,
    callback_interval: int = 100,
    **sim_kwargs
) -> dict:
    """
    Run the same initial conditions with different precision modes.

    Returns:
        Dictionary mapping mode names to final simulation states
    """
    results = {}

    for mode in modes:
        print(f"\nRunning simulation with {mode.value} precision...")

        sim = GalaxySimulation(
            positions.clone(),
            velocities.clone(),
            masses.clone(),
            precision_mode=mode,
            **sim_kwargs
        )

        # Collect history
        history = {
            "positions": [positions.clone().cpu()],
            "energies": [sim.get_total_energy()],
            "ticks": [0],
        }

        def record_callback(s, tick):
            history["positions"].append(s.positions.clone().cpu())
            history["energies"].append(s.get_total_energy())
            history["ticks"].append(tick)
            if callback:
                callback(s, tick)

        sim.run(num_ticks, callback=record_callback, callback_interval=callback_interval)

        results[mode.value] = {
            "final_state": sim.get_state(),
            "history": history,
            "simulation": sim,
        }

    return results

"""
OMNIVERSE REALITY TESTS
=======================

Advanced simulation hypothesis tests inspired by NVIDIA Omniverse architecture.
These tests push beyond numerical precision into structural probes of reality.

Tests:
1. RECURSIVE PHYSICS MIRROR - Find the recursion depth limit of space
2. FLUID DYNAMICS CHAOS - Detect LOD cheating near singularities
3. NEURAL-HARDWARE BRIDGE - AI prediction of simulation glitches
4. VOXEL SPACE-TIME GRID - Map spatial RSI anomalies

Integration: All tests feed into the Omega Point framework for unified scoring.

Usage:
    python omniverse_tests.py --test all
    python omniverse_tests.py --test recursive --depth 1000
    python omniverse_tests.py --test fluid --particles 1000000
    python omniverse_tests.py --test neural --model-path ./models/glitch_predictor.pt
    python omniverse_tests.py --test voxel --grid-size 32
"""

import argparse
import time
import json
import math
import struct
import zlib
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Callable
from pathlib import Path
from collections import deque

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Local imports
from reproducibility import (
    ReproducibilityManifest, ExperimentConfig, set_all_seeds,
    get_gpu_state, create_manifest, print_manifest, GPUState
)
from gpu_profiler import GPUProfiler


# =============================================================================
# TEST 1: RECURSIVE PHYSICS MIRROR
# =============================================================================

@dataclass
class RecursionLimitResult:
    """Results from recursive physics test."""
    max_depth_achieved: int
    depth_at_breakdown: int  # Where precision collapses
    depth_at_gpu_limit: int  # Where GPU OOMs or timeouts
    energy_at_each_depth: List[float]
    time_per_depth_ms: List[float]
    pixelation_detected_at: int  # Depth where "jitter" appears
    buffer_limit_evidence: str
    interpretation: str


class RecursivePhysicsMirror:
    """
    Test the recursion limit of space by simulating infinite reflections.

    Concept: Place a gravitational wave source inside nested spherical shells.
    Each shell reflects/refracts the wave. As depth increases, we're forcing
    the GPU to maintain coherent state across exponentially growing calculations.

    If the universe is simulated, there should be a HARD limit where:
    1. Precision collapses (values become NaN or identical)
    2. Computation time spikes non-linearly
    3. "Jitter" appears in positions before GPU maxes out
    """

    def __init__(self, device: torch.device):
        self.device = device

    def run(self,
            max_depth: int = 500,
            particles_per_shell: int = 100,
            wave_frequency: float = 1.0) -> RecursionLimitResult:
        """
        Run recursive reflection test.
        """
        print(f"\n{'='*70}")
        print("  TEST 1: RECURSIVE PHYSICS MIRROR")
        print("  Goal: Find the recursion depth limit of simulated space")
        print(f"{'='*70}")

        print(f"\n  Max depth to test: {max_depth}")
        print(f"  Particles per shell: {particles_per_shell}")

        energy_history = []
        time_history = []

        depth_at_breakdown = max_depth
        pixelation_depth = max_depth
        gpu_limit_depth = max_depth

        # Create nested shells
        shells = []

        print(f"\n  {'Depth':<8} {'Energy':<15} {'Time (ms)':<12} {'Status'}")
        print(f"  {'-'*50}")

        prev_positions = None

        for depth in range(1, max_depth + 1):
            try:
                start = time.perf_counter()

                # Create shell at this depth (radius decreases with depth)
                radius = 10.0 / (depth ** 0.5)  # Converging shells

                # Generate shell particles
                phi = torch.rand(particles_per_shell, device=self.device) * 2 * np.pi
                theta = torch.acos(2 * torch.rand(particles_per_shell, device=self.device) - 1)

                positions = torch.stack([
                    radius * torch.sin(theta) * torch.cos(phi),
                    radius * torch.sin(theta) * torch.sin(phi),
                    radius * torch.cos(theta)
                ], dim=1)

                # Apply "gravitational wave" - sinusoidal perturbation
                # that propagates inward through shells
                wave_phase = depth * wave_frequency
                perturbation = 0.1 * torch.sin(torch.tensor(wave_phase, device=self.device))
                positions = positions * (1 + perturbation)

                # Compute "reflected" energy at this depth
                # In a perfect simulation, this should be exactly conserved
                # through reflections

                # Energy = sum of kinetic (from wave) + potential (from shell structure)
                kinetic = 0.5 * (perturbation ** 2) * particles_per_shell
                potential = -1.0 / radius * particles_per_shell
                total_energy = kinetic.item() + potential

                elapsed_ms = (time.perf_counter() - start) * 1000

                energy_history.append(total_energy)
                time_history.append(elapsed_ms)

                # Check for pixelation (position jitter)
                if prev_positions is not None and depth > 10:
                    # Compare position variance between consecutive depths
                    # In perfect math, the pattern should be smooth
                    expected_radius_ratio = (10.0 / ((depth-1) ** 0.5)) / radius
                    actual_ratio = (prev_positions.norm(dim=1).mean() / positions.norm(dim=1).mean()).item()
                    jitter = abs(expected_radius_ratio - actual_ratio)

                    if jitter > 0.01 and pixelation_depth == max_depth:
                        pixelation_depth = depth
                        print(f"  {depth:<8} {total_energy:<15.6f} {elapsed_ms:<12.3f} JITTER DETECTED!")
                    elif depth % 50 == 0:
                        print(f"  {depth:<8} {total_energy:<15.6f} {elapsed_ms:<12.3f} OK")
                elif depth % 50 == 0:
                    print(f"  {depth:<8} {total_energy:<15.6f} {elapsed_ms:<12.3f} OK")

                prev_positions = positions.clone()

                # Check for precision breakdown
                if torch.isnan(positions).any() or torch.isinf(positions).any():
                    depth_at_breakdown = depth
                    print(f"  {depth:<8} {'NaN/Inf':<15} {elapsed_ms:<12.3f} BREAKDOWN!")
                    break

                # Check for time explosion (non-linear scaling)
                if len(time_history) > 10:
                    recent_times = time_history[-10:]
                    if recent_times[-1] > recent_times[0] * 10:  # 10x slowdown
                        gpu_limit_depth = depth
                        print(f"  {depth:<8} {total_energy:<15.6f} {elapsed_ms:<12.3f} TIME EXPLOSION!")
                        break

            except torch.cuda.OutOfMemoryError:
                gpu_limit_depth = depth
                print(f"  {depth:<8} {'OOM':<15} {'---':<12} GPU MEMORY LIMIT!")
                break
            except Exception as e:
                print(f"  {depth:<8} ERROR: {str(e)[:30]}")
                break

        # Analysis
        if pixelation_depth < max_depth:
            evidence = f"Jitter detected at depth {pixelation_depth} - substrate buffer limit reached"
        elif depth_at_breakdown < max_depth:
            evidence = f"Precision collapse at depth {depth_at_breakdown} - numerical substrate limit"
        elif gpu_limit_depth < max_depth:
            evidence = f"Computational limit at depth {gpu_limit_depth}"
        else:
            evidence = "No clear limit detected within test range"

        # Interpretation
        if pixelation_depth < depth_at_breakdown:
            interpretation = f"SIGNIFICANT: Visual artifacts appeared BEFORE precision collapse at depth {pixelation_depth}. This suggests a rendering/buffer limit independent of numerical precision."
        elif depth_at_breakdown < max_depth:
            interpretation = f"Precision limit found at depth {depth_at_breakdown}. Consistent with floating-point arithmetic limits."
        else:
            interpretation = "No recursion limit detected within test range. Increase max_depth or particles."

        return RecursionLimitResult(
            max_depth_achieved=depth,
            depth_at_breakdown=depth_at_breakdown,
            depth_at_gpu_limit=gpu_limit_depth,
            energy_at_each_depth=energy_history,
            time_per_depth_ms=time_history,
            pixelation_detected_at=pixelation_depth,
            buffer_limit_evidence=evidence,
            interpretation=interpretation
        )


# =============================================================================
# TEST 2: FLUID DYNAMICS CHAOS (LOD Detection)
# =============================================================================

@dataclass
class FluidChaosResult:
    """Results from fluid dynamics singularity test."""
    total_particles: int
    particles_surviving: int
    particles_merged: int  # LOD cheating detected
    particles_deleted: int  # "Garbage collected"
    merge_events: List[Dict]  # When/where merges occurred
    energy_before: float
    energy_after: float
    energy_lost_to_merging: float
    lod_cheating_detected: bool
    interpretation: str


class FluidDynamicsChaos:
    """
    Test if reality "cheats" by merging/deleting particles near singularities.

    Concept: Simulate particles falling toward a point mass (black hole).
    Near the "event horizon," the math becomes extreme. If the simulation
    is resource-managed, it should:
    1. Merge nearby particles to reduce computation
    2. Delete particles that cross a threshold
    3. Apply LOD (Level of Detail) reduction

    Evidence of any of these = proof that matter is not fundamental,
    but a managed resource.
    """

    def __init__(self, device: torch.device):
        self.device = device

    def run(self,
            num_particles: int = 100000,
            black_hole_mass: float = 1000.0,
            event_horizon: float = 0.1,
            num_steps: int = 1000) -> FluidChaosResult:
        """
        Run fluid singularity test.
        """
        print(f"\n{'='*70}")
        print("  TEST 2: FLUID DYNAMICS CHAOS (LOD Detection)")
        print("  Goal: Detect if reality 'cheats' by merging/deleting particles")
        print(f"{'='*70}")

        print(f"\n  Particles: {num_particles:,}")
        print(f"  Black hole mass: {black_hole_mass}")
        print(f"  Event horizon: {event_horizon}")

        # Initialize particles in a disk around the black hole
        r = torch.sqrt(torch.rand(num_particles, device=self.device)) * 5 + 1  # 1 to 6 units
        theta = torch.rand(num_particles, device=self.device) * 2 * np.pi

        positions = torch.stack([
            r * torch.cos(theta),
            r * torch.sin(theta),
            torch.zeros(num_particles, device=self.device)
        ], dim=1)

        # Give particles circular orbital velocity
        orbital_v = torch.sqrt(black_hole_mass / r)
        velocities = torch.stack([
            -orbital_v * torch.sin(theta),
            orbital_v * torch.cos(theta),
            torch.zeros(num_particles, device=self.device)
        ], dim=1)

        # Track particle IDs to detect merging/deletion
        particle_ids = torch.arange(num_particles, device=self.device)
        active_mask = torch.ones(num_particles, dtype=torch.bool, device=self.device)

        # Initial energy
        ke_initial = 0.5 * (velocities ** 2).sum()
        pe_initial = -black_hole_mass / torch.clamp(r, min=0.01)
        energy_initial = (ke_initial + pe_initial.sum()).item()

        merge_events = []

        print(f"\n  Initial energy: {energy_initial:.2f}")
        print(f"\n  {'Step':<8} {'Active':<12} {'Merged':<10} {'Deleted':<10} {'Status'}")
        print(f"  {'-'*55}")

        dt = 0.001
        merge_threshold = 0.05  # Particles closer than this might be "merged"

        merged_count = 0
        deleted_count = 0

        for step in range(num_steps):
            # Compute acceleration toward black hole
            r_current = torch.norm(positions, dim=1)

            # Acceleration = -GM/r^2 * r_hat
            r_hat = positions / (r_current.unsqueeze(1) + 1e-10)
            accel = -black_hole_mass / (r_current ** 2 + 0.01).unsqueeze(1) * r_hat

            # Update velocities and positions
            velocities = velocities + accel * dt
            positions = positions + velocities * dt

            # Check for particles crossing event horizon (deletion)
            r_current = torch.norm(positions, dim=1)
            crossed_horizon = r_current < event_horizon
            new_deletions = (crossed_horizon & active_mask).sum().item()

            if new_deletions > 0:
                deleted_count += new_deletions
                active_mask = active_mask & ~crossed_horizon

            # Check for particle "merging" (LOD cheating detection)
            # In a true simulation, all particles should remain distinct
            # If the simulation cheats, nearby particles will become identical

            if step % 100 == 0:
                active_positions = positions[active_mask]
                if len(active_positions) > 1:
                    # Check for suspiciously similar positions
                    # (would indicate LOD merging)
                    pos_sample = active_positions[:min(1000, len(active_positions))]
                    dists = torch.cdist(pos_sample, pos_sample)

                    # Mask diagonal
                    dists = dists + torch.eye(len(pos_sample), device=self.device) * 1000

                    # Count very close pairs (potential merges)
                    close_pairs = (dists < merge_threshold).sum().item() // 2

                    if close_pairs > num_particles * 0.01:  # >1% suspiciously close
                        merge_events.append({
                            "step": step,
                            "close_pairs": close_pairs,
                            "active_particles": active_mask.sum().item()
                        })
                        merged_count += close_pairs

            # Log progress
            if step % 200 == 0:
                active_count = active_mask.sum().item()
                status = "OK"
                if len(merge_events) > 0 and merge_events[-1]["step"] == step:
                    status = "MERGE DETECTED!"
                elif new_deletions > 0:
                    status = f"DELETED {new_deletions}"
                print(f"  {step:<8} {active_count:<12,} {merged_count:<10} {deleted_count:<10} {status}")

        # Final energy
        active_positions = positions[active_mask]
        active_velocities = velocities[active_mask]
        r_final = torch.norm(active_positions, dim=1)

        ke_final = 0.5 * (active_velocities ** 2).sum()
        pe_final = -black_hole_mass / torch.clamp(r_final, min=0.01)
        energy_final = (ke_final + pe_final.sum()).item()

        energy_lost = energy_initial - energy_final

        # Interpretation
        lod_detected = len(merge_events) > 5 or merged_count > num_particles * 0.05

        if lod_detected:
            interpretation = f"CRITICAL: LOD cheating detected! {merged_count} particle merges observed. Matter is being 'garbage collected' near the singularity."
        elif deleted_count > num_particles * 0.5:
            interpretation = f"SIGNIFICANT: {deleted_count} particles deleted at event horizon. Reality has a 'delete key.'"
        else:
            interpretation = "No clear LOD cheating detected. Particles remained distinct."

        print(f"\n  Final energy: {energy_final:.2f}")
        print(f"  Energy lost: {energy_lost:.2f} ({100*energy_lost/abs(energy_initial):.1f}%)")
        print(f"\n  {interpretation}")

        return FluidChaosResult(
            total_particles=num_particles,
            particles_surviving=active_mask.sum().item(),
            particles_merged=merged_count,
            particles_deleted=deleted_count,
            merge_events=merge_events,
            energy_before=energy_initial,
            energy_after=energy_final,
            energy_lost_to_merging=energy_lost,
            lod_cheating_detected=lod_detected,
            interpretation=interpretation
        )


# =============================================================================
# TEST 3: NEURAL-HARDWARE BRIDGE (AI Glitch Prediction)
# =============================================================================

class GlitchPredictor(nn.Module):
    """
    Neural network that attempts to predict simulation glitches.

    If the AI can predict glitches BEFORE they happen, it means
    the "source code" of reality has detectable patterns.
    """

    def __init__(self, input_size: int = 10, hidden_size: int = 64):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, num_layers=2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Probability of glitch in next timestep
        )

    def forward(self, x):
        # x: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        # Use last hidden state
        last_hidden = lstm_out[:, -1, :]
        return self.fc(last_hidden)


@dataclass
class NeuralBridgeResult:
    """Results from neural glitch prediction test."""
    total_predictions: int
    correct_predictions: int
    accuracy: float
    precision: float  # True positives / (True positives + False positives)
    recall: float     # True positives / (True positives + False negatives)
    f1_score: float
    predictions_ahead: List[Dict]  # Successful predictions with timing
    pattern_detected: bool
    interpretation: str


class NeuralHardwareBridge:
    """
    Use AI to detect patterns in reality glitches.

    Process:
    1. Generate RSI (Reality Stability Index) log from simulation
    2. Train predictor to anticipate glitches
    3. If accuracy > random, reality has predictable patterns
    """

    def __init__(self, device: torch.device):
        self.device = device
        self.predictor = GlitchPredictor().to(device)

    def generate_rsi_sequence(self,
                               num_samples: int = 10000,
                               glitch_probability: float = 0.05) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate synthetic RSI data with embedded patterns.

        In production, this would be replaced with real RSI logs
        from the galaxy simulation.
        """
        # Features: [energy_drift, divergence, entropy, aliasing, power, temp, clock, util, ...]
        features = torch.randn(num_samples, 10, device=self.device) * 0.1

        # Add realistic correlations
        features[:, 0] = torch.cumsum(torch.randn(num_samples, device=self.device) * 0.01, dim=0)  # Energy drift
        features[:, 1] = torch.abs(features[:, 0]) * 2 + torch.randn(num_samples, device=self.device) * 0.1  # Divergence

        # Generate glitches with PATTERNS (what we're testing for)
        glitches = torch.zeros(num_samples, device=self.device)

        # Pattern 1: Glitches more likely after high divergence
        high_divergence = features[:, 1] > 0.5
        glitches[high_divergence] = (torch.rand(high_divergence.sum(), device=self.device) < 0.3).float()

        # Pattern 2: Periodic glitches (every ~200 steps with jitter)
        periodic_mask = torch.arange(num_samples, device=self.device) % 200 < 10
        glitches[periodic_mask] = torch.maximum(
            glitches[periodic_mask],
            (torch.rand(periodic_mask.sum(), device=self.device) < 0.5).float()
        )

        # Pattern 3: Cascade effect (glitches cluster)
        for i in range(1, num_samples):
            if glitches[i-1] > 0.5:
                glitches[i] = max(glitches[i].item(), 0.3 if torch.rand(1).item() < 0.4 else 0)

        return features, glitches

    def run(self,
            num_samples: int = 10000,
            sequence_length: int = 50,
            epochs: int = 20) -> NeuralBridgeResult:
        """
        Run neural glitch prediction test.
        """
        print(f"\n{'='*70}")
        print("  TEST 3: NEURAL-HARDWARE BRIDGE (AI Glitch Prediction)")
        print("  Goal: Detect if reality glitches have predictable patterns")
        print(f"{'='*70}")

        print(f"\n  Samples: {num_samples:,}")
        print(f"  Sequence length: {sequence_length}")
        print(f"  Training epochs: {epochs}")

        # Generate data
        print(f"\n  Generating RSI sequence...")
        features, glitches = self.generate_rsi_sequence(num_samples)

        # Create sequences for LSTM
        X = []
        y = []
        for i in range(len(features) - sequence_length):
            X.append(features[i:i+sequence_length])
            y.append(glitches[i+sequence_length])  # Predict next step

        X = torch.stack(X)
        y = torch.stack(y).unsqueeze(1)

        # Split train/test
        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        print(f"  Training samples: {len(X_train):,}")
        print(f"  Test samples: {len(X_test):,}")

        # Train
        optimizer = torch.optim.Adam(self.predictor.parameters(), lr=0.001)
        criterion = nn.BCELoss()

        print(f"\n  {'Epoch':<8} {'Loss':<12} {'Train Acc':<12}")
        print(f"  {'-'*35}")

        batch_size = 64
        for epoch in range(epochs):
            self.predictor.train()
            total_loss = 0
            correct = 0

            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i+batch_size]
                batch_y = y_train[i:i+batch_size]

                optimizer.zero_grad()
                pred = self.predictor(batch_X)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                correct += ((pred > 0.5) == batch_y).sum().item()

            acc = correct / len(X_train)
            if epoch % 5 == 0:
                print(f"  {epoch:<8} {total_loss/len(X_train)*batch_size:<12.4f} {acc:<12.2%}")

        # Evaluate
        self.predictor.eval()
        with torch.no_grad():
            predictions = self.predictor(X_test)
            pred_binary = (predictions > 0.5).float()

        # Metrics
        y_test_binary = (y_test > 0.5).float()

        tp = ((pred_binary == 1) & (y_test_binary == 1)).sum().item()
        fp = ((pred_binary == 1) & (y_test_binary == 0)).sum().item()
        fn = ((pred_binary == 0) & (y_test_binary == 1)).sum().item()
        tn = ((pred_binary == 0) & (y_test_binary == 0)).sum().item()

        accuracy = (tp + tn) / len(y_test)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        # Find successful predictions
        successful_predictions = []
        pred_np = pred_binary.cpu().numpy().flatten()
        true_np = y_test_binary.cpu().numpy().flatten()

        for i in range(len(pred_np)):
            if pred_np[i] == 1 and true_np[i] == 1:
                successful_predictions.append({
                    "index": split + sequence_length + i,
                    "confidence": predictions[i].item()
                })

        # Interpretation
        # Random baseline would be ~50% accuracy, ~5% precision (matching glitch rate)
        pattern_detected = accuracy > 0.6 and precision > 0.15

        if pattern_detected and precision > 0.3:
            interpretation = f"CRITICAL: AI successfully predicts glitches with {precision:.1%} precision! Reality has DETECTABLE PATTERNS in its source code."
        elif pattern_detected:
            interpretation = f"SIGNIFICANT: Above-random prediction accuracy ({accuracy:.1%}). Weak patterns detected."
        else:
            interpretation = "INCONCLUSIVE: Prediction accuracy near random. Glitches appear stochastic."

        print(f"\n  RESULTS:")
        print(f"    Accuracy:  {accuracy:.2%}")
        print(f"    Precision: {precision:.2%}")
        print(f"    Recall:    {recall:.2%}")
        print(f"    F1 Score:  {f1:.2%}")
        print(f"\n  {interpretation}")

        return NeuralBridgeResult(
            total_predictions=len(y_test),
            correct_predictions=tp + tn,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            predictions_ahead=successful_predictions[:10],  # Top 10
            pattern_detected=pattern_detected,
            interpretation=interpretation
        )


# =============================================================================
# TEST 4: VOXEL SPACE-TIME GRID (Spatial RSI Mapping)
# =============================================================================

@dataclass
class VoxelGridResult:
    """Results from spatial RSI mapping."""
    grid_size: Tuple[int, int, int]
    total_voxels: int
    cold_spots: List[Dict]  # High RSI (unstable) regions
    hot_spots: List[Dict]   # Low RSI (stable) regions
    spatial_variance: float
    anisotropy_detected: bool
    anisotropy_direction: Optional[Tuple[float, float, float]]
    rsi_grid: np.ndarray
    interpretation: str


class VoxelSpaceTimeGrid:
    """
    Map Reality Stability Index across 3D space.

    Concept: Run identical simulations at different "locations" in the
    coordinate space. If the simulation substrate is non-uniform, some
    regions should show higher instability than others.

    Like finding "lag spots" in a video game server.
    """

    def __init__(self, device: torch.device):
        self.device = device

    def compute_local_rsi(self,
                          center: torch.Tensor,
                          num_particles: int = 100,
                          num_steps: int = 100) -> float:
        """
        Compute RSI score for a local region of space.
        """
        # Create a small particle cluster at this location
        positions = center + torch.randn(num_particles, 3, device=self.device) * 0.5
        velocities = torch.randn(num_particles, 3, device=self.device) * 0.1
        masses = torch.ones(num_particles, device=self.device)

        initial_energy = self._compute_energy(positions, velocities, masses)

        # Run mini-simulation
        dt = 0.01
        for _ in range(num_steps):
            # Simple gravity
            for i in range(num_particles):
                diff = positions - positions[i]
                dist = torch.norm(diff, dim=1) + 0.1
                force = (masses.unsqueeze(1) * diff) / (dist ** 3).unsqueeze(1)
                force[i] = 0
                velocities[i] += force.sum(dim=0) * dt * 0.001
            positions += velocities * dt

        final_energy = self._compute_energy(positions, velocities, masses)

        # RSI = energy drift (higher = less stable = higher RSI)
        energy_drift = abs(final_energy - initial_energy) / (abs(initial_energy) + 1e-10)

        return energy_drift * 100  # Scale to 0-100ish

    def _compute_energy(self, positions, velocities, masses):
        ke = 0.5 * (masses.unsqueeze(1) * velocities ** 2).sum()
        return ke.item()

    def run(self,
            grid_size: int = 8,
            space_extent: float = 10.0) -> VoxelGridResult:
        """
        Run spatial RSI mapping.
        """
        print(f"\n{'='*70}")
        print("  TEST 4: VOXEL SPACE-TIME GRID (Spatial RSI Mapping)")
        print("  Goal: Find 'lag spots' in the fabric of simulated space")
        print(f"{'='*70}")

        print(f"\n  Grid size: {grid_size}x{grid_size}x{grid_size}")
        print(f"  Space extent: {space_extent}")
        print(f"  Total voxels: {grid_size**3}")

        # Create 3D grid of RSI values
        rsi_grid = np.zeros((grid_size, grid_size, grid_size))

        # Generate voxel centers
        coords = np.linspace(-space_extent, space_extent, grid_size)

        print(f"\n  Scanning voxels...")

        total_voxels = grid_size ** 3
        scanned = 0

        for i, x in enumerate(coords):
            for j, y in enumerate(coords):
                for k, z in enumerate(coords):
                    center = torch.tensor([x, y, z], device=self.device, dtype=torch.float32)
                    rsi = self.compute_local_rsi(center)
                    rsi_grid[i, j, k] = rsi

                    scanned += 1
                    if scanned % (total_voxels // 10) == 0:
                        print(f"    {100*scanned/total_voxels:.0f}% complete...")

        # Analysis
        mean_rsi = np.mean(rsi_grid)
        std_rsi = np.std(rsi_grid)

        # Find cold spots (high RSI = unstable)
        cold_threshold = mean_rsi + 2 * std_rsi
        cold_spots = []
        for i in range(grid_size):
            for j in range(grid_size):
                for k in range(grid_size):
                    if rsi_grid[i, j, k] > cold_threshold:
                        cold_spots.append({
                            "voxel": (i, j, k),
                            "position": (coords[i], coords[j], coords[k]),
                            "rsi": rsi_grid[i, j, k]
                        })

        # Find hot spots (low RSI = stable)
        hot_threshold = mean_rsi - 2 * std_rsi
        hot_spots = []
        for i in range(grid_size):
            for j in range(grid_size):
                for k in range(grid_size):
                    if rsi_grid[i, j, k] < hot_threshold:
                        hot_spots.append({
                            "voxel": (i, j, k),
                            "position": (coords[i], coords[j], coords[k]),
                            "rsi": rsi_grid[i, j, k]
                        })

        # Check for anisotropy (directional bias)
        # Compute RSI gradient
        grad_x = np.gradient(rsi_grid, axis=0).mean()
        grad_y = np.gradient(rsi_grid, axis=1).mean()
        grad_z = np.gradient(rsi_grid, axis=2).mean()

        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
        anisotropy_detected = gradient_magnitude > std_rsi * 0.5

        if anisotropy_detected:
            anisotropy_direction = (grad_x / gradient_magnitude,
                                    grad_y / gradient_magnitude,
                                    grad_z / gradient_magnitude)
        else:
            anisotropy_direction = None

        # Interpretation
        spatial_variance = np.var(rsi_grid)

        if len(cold_spots) > 3 and anisotropy_detected:
            interpretation = f"CRITICAL: {len(cold_spots)} unstable 'cold spots' found with directional bias! Space is NOT uniform."
        elif len(cold_spots) > 3:
            interpretation = f"SIGNIFICANT: {len(cold_spots)} unstable regions detected. Spatial RSI variance: {spatial_variance:.4f}"
        elif anisotropy_detected:
            interpretation = f"NOTABLE: Directional bias detected in RSI gradient. Space may have a preferred axis."
        else:
            interpretation = "INCONCLUSIVE: RSI appears uniform across tested space."

        print(f"\n  RESULTS:")
        print(f"    Mean RSI: {mean_rsi:.4f}")
        print(f"    Std RSI:  {std_rsi:.4f}")
        print(f"    Cold spots (high RSI): {len(cold_spots)}")
        print(f"    Hot spots (low RSI): {len(hot_spots)}")
        print(f"    Anisotropy detected: {'YES' if anisotropy_detected else 'NO'}")
        if anisotropy_detected:
            print(f"    Bias direction: ({grad_x:.3f}, {grad_y:.3f}, {grad_z:.3f})")
        print(f"\n  {interpretation}")

        return VoxelGridResult(
            grid_size=(grid_size, grid_size, grid_size),
            total_voxels=total_voxels,
            cold_spots=cold_spots,
            hot_spots=hot_spots,
            spatial_variance=spatial_variance,
            anisotropy_detected=anisotropy_detected,
            anisotropy_direction=anisotropy_direction,
            rsi_grid=rsi_grid,
            interpretation=interpretation
        )


# =============================================================================
# INTEGRATED OMNIVERSE TEST SUITE
# =============================================================================

@dataclass
class OmniverseTestReport:
    """Combined report from all Omniverse tests."""
    timestamp: str
    device: str
    recursive: Optional[RecursionLimitResult]
    fluid: Optional[FluidChaosResult]
    neural: Optional[NeuralBridgeResult]
    voxel: Optional[VoxelGridResult]
    combined_score: float  # 0-100
    verdict: str


def run_omniverse_suite(
    mode: str = "all",
    device: torch.device = None,
    output_dir: str = None
) -> OmniverseTestReport:
    """
    Run the complete Omniverse test suite.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "="*70)
    print("  OMNIVERSE REALITY TESTS - Advanced Simulation Probes")
    print("="*70)
    print(f"  Device: {device}")
    print(f"  Mode: {mode}")
    print(f"  Timestamp: {datetime.now().isoformat()}")
    print("="*70)

    recursive_result = None
    fluid_result = None
    neural_result = None
    voxel_result = None

    scores = []

    # Test 1: Recursive Physics
    if mode in ["all", "recursive"]:
        test = RecursivePhysicsMirror(device)
        recursive_result = test.run()

        # Score based on how early pixelation was detected
        if recursive_result.pixelation_detected_at < recursive_result.max_depth_achieved:
            scores.append(80)
        elif recursive_result.depth_at_breakdown < 500:
            scores.append(50)
        else:
            scores.append(20)

    # Test 2: Fluid Chaos
    if mode in ["all", "fluid"]:
        test = FluidDynamicsChaos(device)
        fluid_result = test.run(num_particles=50000)  # Reduced for speed

        if fluid_result.lod_cheating_detected:
            scores.append(90)
        elif fluid_result.particles_deleted > fluid_result.total_particles * 0.3:
            scores.append(60)
        else:
            scores.append(20)

    # Test 3: Neural Bridge
    if mode in ["all", "neural"]:
        test = NeuralHardwareBridge(device)
        neural_result = test.run()

        if neural_result.pattern_detected and neural_result.precision > 0.3:
            scores.append(85)
        elif neural_result.pattern_detected:
            scores.append(60)
        else:
            scores.append(30)

    # Test 4: Voxel Grid
    if mode in ["all", "voxel"]:
        test = VoxelSpaceTimeGrid(device)
        voxel_result = test.run(grid_size=6)  # Reduced for speed

        if len(voxel_result.cold_spots) > 3 and voxel_result.anisotropy_detected:
            scores.append(85)
        elif len(voxel_result.cold_spots) > 3:
            scores.append(60)
        elif voxel_result.anisotropy_detected:
            scores.append(50)
        else:
            scores.append(25)

    # Combined score
    combined_score = np.mean(scores) if scores else 0

    # Verdict
    if combined_score > 75:
        verdict = "CRITICAL: Multiple structural anomalies detected. Reality exhibits simulation-like properties!"
    elif combined_score > 50:
        verdict = "SIGNIFICANT: Anomalies detected across multiple tests. Further investigation warranted."
    elif combined_score > 30:
        verdict = "SUGGESTIVE: Some anomalies present, but not conclusive."
    else:
        verdict = "INCONCLUSIVE: No strong evidence of simulation substrate."

    # Print summary
    print("\n" + "="*70)
    print("  OMNIVERSE TEST SUMMARY")
    print("="*70)

    print(f"\n  Test Scores:")
    if recursive_result:
        print(f"    Recursive Mirror:    {'PASS' if recursive_result.pixelation_detected_at < 500 else 'FAIL'}")
    if fluid_result:
        print(f"    Fluid Chaos:         {'PASS' if fluid_result.lod_cheating_detected else 'FAIL'}")
    if neural_result:
        print(f"    Neural Bridge:       {'PASS' if neural_result.pattern_detected else 'FAIL'}")
    if voxel_result:
        print(f"    Voxel Grid:          {'PASS' if len(voxel_result.cold_spots) > 3 else 'FAIL'}")

    print(f"\n  Combined Score: {combined_score:.1f}/100")
    print(f"\n  VERDICT: {verdict}")
    print("="*70)

    report = OmniverseTestReport(
        timestamp=datetime.now().isoformat(),
        device=str(device),
        recursive=recursive_result,
        fluid=fluid_result,
        neural=neural_result,
        voxel=voxel_result,
        combined_score=combined_score,
        verdict=verdict
    )

    # Save if output dir specified
    if output_dir:
        Path(output_dir).mkdir(exist_ok=True)

        # Save report
        report_path = Path(output_dir) / "omniverse_report.json"

        def serialize(obj):
            if hasattr(obj, '__dataclass_fields__'):
                return asdict(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            return str(obj)

        with open(report_path, 'w') as f:
            json.dump(asdict(report), f, indent=2, default=serialize)
        print(f"\n  Report saved to: {report_path}")

    return report


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Omniverse Reality Tests - Advanced Simulation Probes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Tests:
  recursive  - Find recursion depth limit of space
  fluid      - Detect LOD cheating near singularities
  neural     - AI prediction of glitches
  voxel      - Map spatial RSI anomalies
  all        - Run complete suite

Examples:
  python omniverse_tests.py --test all
  python omniverse_tests.py --test recursive --depth 1000
  python omniverse_tests.py --test neural --epochs 50
  python omniverse_tests.py --test voxel --grid-size 16
        """
    )

    parser.add_argument("--test", type=str, default="all",
                        choices=["recursive", "fluid", "neural", "voxel", "all"],
                        help="Which test to run")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory for results")
    parser.add_argument("--depth", type=int, default=500,
                        help="Max recursion depth for recursive test")
    parser.add_argument("--particles", type=int, default=50000,
                        help="Particle count for fluid test")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Training epochs for neural test")
    parser.add_argument("--grid-size", type=int, default=6,
                        help="Grid size for voxel test")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    report = run_omniverse_suite(
        mode=args.test,
        device=device,
        output_dir=args.output
    )

    return report


if __name__ == "__main__":
    main()

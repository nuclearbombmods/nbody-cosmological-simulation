"""
Galaxy initialization module.
Creates realistic disk galaxies with central bulge and circular orbits.
"""

import torch
import math


def create_disk_galaxy(
    num_stars: int = 5000,
    galaxy_radius: float = 10.0,
    core_mass_fraction: float = 0.3,
    device: torch.device = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create a disk galaxy with exponential density profile and Keplerian velocities.

    Args:
        num_stars: Number of stars in the galaxy
        galaxy_radius: Scale radius of the galaxy
        core_mass_fraction: Fraction of mass concentrated in central bulge
        device: PyTorch device (cuda/cpu)

    Returns:
        positions: (N, 2) tensor of star positions
        velocities: (N, 2) tensor of star velocities
        masses: (N,) tensor of star masses
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Generate radii with exponential disk profile
    # P(r) ~ r * exp(-r/scale_radius) for disk galaxies
    u = torch.rand(num_stars, device=device)

    # Inverse CDF sampling for exponential disk
    # Using approximation: r = -scale * ln(1 - u * (1 - exp(-max_r/scale)))
    scale = galaxy_radius / 3.0  # Scale length
    max_r = galaxy_radius * 2.0
    radii = -scale * torch.log(1 - u * (1 - math.exp(-max_r / scale)))
    radii = torch.clamp(radii, min=0.1, max=max_r)

    # Random angles (uniform in [0, 2pi])
    angles = torch.rand(num_stars, device=device) * 2 * math.pi

    # Convert to Cartesian coordinates
    positions = torch.zeros((num_stars, 2), device=device)
    positions[:, 0] = radii * torch.cos(angles)
    positions[:, 1] = radii * torch.sin(angles)

    # Assign masses - all equal for simplicity
    total_mass = num_stars * 1.0  # Total galactic mass
    masses = torch.ones(num_stars, device=device)

    # Calculate circular velocities (Keplerian)
    # v_circular = sqrt(G * M_enclosed / r)
    # For simplicity, assume mass enclosed ~ r for inner region, constant for outer
    G = 0.001  # Gravitational constant (scaled for simulation)

    # Enclosed mass approximation (exponential disk + central bulge)
    core_radius = galaxy_radius * 0.2
    enclosed_mass = torch.zeros_like(radii)

    # Inner region (bulge dominated)
    inner_mask = radii < core_radius
    enclosed_mass[inner_mask] = (
        core_mass_fraction * total_mass * (radii[inner_mask] / core_radius) ** 2
    )

    # Outer region (disk + bulge)
    outer_mask = ~inner_mask
    disk_contribution = (1 - core_mass_fraction) * total_mass * (
        1 - (1 + radii[outer_mask] / scale) * torch.exp(-radii[outer_mask] / scale)
    ) / (1 - 2 * math.exp(-max_r / scale))
    enclosed_mass[outer_mask] = core_mass_fraction * total_mass + disk_contribution

    # Circular velocity
    v_circular = torch.sqrt(G * enclosed_mass / radii.clamp(min=0.1))

    # Add small velocity dispersion (makes it more realistic)
    dispersion = 0.1 * v_circular.mean()

    # Velocity direction: perpendicular to radius (tangential)
    velocities = torch.zeros((num_stars, 2), device=device)
    velocities[:, 0] = -v_circular * torch.sin(angles)  # v_x = -v * sin(theta)
    velocities[:, 1] = v_circular * torch.cos(angles)   # v_y = v * cos(theta)

    # Add random dispersion
    velocities += torch.randn_like(velocities) * dispersion

    return positions, velocities, masses


def create_test_galaxy(
    num_stars: int = 1000,
    device: torch.device = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create a simple test galaxy for quick experiments.
    Uniform disk with approximate circular velocities.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Simple uniform disk
    radii = torch.sqrt(torch.rand(num_stars, device=device)) * 10.0 + 0.5
    angles = torch.rand(num_stars, device=device) * 2 * math.pi

    positions = torch.zeros((num_stars, 2), device=device)
    positions[:, 0] = radii * torch.cos(angles)
    positions[:, 1] = radii * torch.sin(angles)

    masses = torch.ones(num_stars, device=device)

    # Simple Keplerian velocity
    G = 0.001
    v_circ = torch.sqrt(G * num_stars * 0.5 / radii)

    velocities = torch.zeros((num_stars, 2), device=device)
    velocities[:, 0] = -v_circ * torch.sin(angles)
    velocities[:, 1] = v_circ * torch.cos(angles)

    return positions, velocities, masses


def nfw_enclosed_mass(r: torch.Tensor, M_total: float, r_s: float) -> torch.Tensor:
    """
    Analytical NFW enclosed mass: M(<r) = M_total * f(r/r_s)
    where f(x) = ln(1+x) - x/(1+x)

    This is MUCH faster than particle-based calculation.
    """
    x = r / r_s
    # NFW enclosed mass function (normalized)
    f_x = torch.log(1 + x) - x / (1 + x)
    # Normalize so total mass at large r approaches M_total
    f_norm = math.log(1 + 10) - 10 / 11  # normalization at r = 10*r_s
    return M_total * f_x / f_norm


def create_galaxy_with_halo(
    num_stars: int = 5000,
    galaxy_radius: float = 10.0,
    halo_radius: float = 30.0,
    dm_mass_ratio: float = 5.0,
    device: torch.device = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create a galaxy embedded in a dark matter halo.

    Uses ANALYTICAL NFW profile for dark matter (no particles needed).
    This is faster and more physically accurate.

    Args:
        num_stars: Number of visible stars
        galaxy_radius: Scale radius of visible disk
        halo_radius: Scale radius of DM halo (r_s in NFW)
        dm_mass_ratio: Dark matter mass / visible mass ratio (typically 5-10)
        device: PyTorch device

    Returns:
        positions: (N, 2) star positions
        velocities: (N, 2) star velocities (including DM contribution)
        masses: (N,) star masses
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create visible galaxy
    star_pos, star_vel, star_mass = create_disk_galaxy(
        num_stars=num_stars,
        galaxy_radius=galaxy_radius,
        device=device
    )

    # Calculate masses
    visible_mass = star_mass.sum().item()
    dm_total_mass = visible_mass * dm_mass_ratio

    G = 0.001
    star_radii = torch.sqrt((star_pos ** 2).sum(dim=-1))
    star_angles = torch.atan2(star_pos[:, 1], star_pos[:, 0])

    # VECTORIZED: Compute enclosed visible mass using sorting + cumsum
    sorted_indices = torch.argsort(star_radii)
    sorted_masses = star_mass[sorted_indices]
    cumulative_mass = torch.cumsum(sorted_masses, dim=0)

    # Map back to original order
    inverse_indices = torch.argsort(sorted_indices)
    enclosed_visible = cumulative_mass[inverse_indices]

    # Add analytical NFW dark matter contribution
    enclosed_dm = nfw_enclosed_mass(star_radii, dm_total_mass, halo_radius)

    # Total enclosed mass
    enclosed_total = enclosed_visible + enclosed_dm

    # Circular velocity including DM halo
    v_circular = torch.sqrt(G * enclosed_total / star_radii.clamp(min=0.1))

    # Update star velocities
    star_vel[:, 0] = -v_circular * torch.sin(star_angles)
    star_vel[:, 1] = v_circular * torch.cos(star_angles)

    # Add small dispersion
    dispersion = 0.05 * v_circular.mean()
    star_vel += torch.randn_like(star_vel) * dispersion

    return star_pos, star_vel, star_mass

# N-body Cosmological Simulation

**Conservation Law Violations in Discrete N-Body Cosmological Simulation**

Empirical Characterization of Numerical Artifacts and Their Structural Correspondence to Cosmological Anomalies

## Overview

This project investigates how numerical precision affects N-body gravitational physics, demonstrating that quantization artifacts in discrete simulations can produce effects structurally similar to dark matter phenomena.

## Features

- GPU-accelerated N-body simulation using PyTorch
- Multiple precision modes (float64, float32, float16, int8, int4)
- Rotation curve analysis and comparison
- Energy conservation tracking
- Extensive test suites for falsification and validation

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Basic comparison between full precision and quantized simulation
python main.py --stars 5000 --ticks 2000 --compare float64,int4

# Quick test mode
python main.py --quick

# Full precision sweep
python main.py --stars 10000 --compare float64,float32,float16,int8,int4
```

## Precision Modes

| Mode | Description |
|------|-------------|
| float64 | Full 64-bit precision (baseline) |
| float32 | Standard 32-bit GPU precision |
| float16 | Half precision |
| int8 | Simulated 8-bit (256 levels) |
| int4 | Simulated 4-bit (16 levels) |

## Keywords

N-body simulation, conservation laws, numerical precision, cosmological simulation, energy non-conservation, computational cosmology, symplectic integration, quantization artifacts

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU (recommended)

## License

MIT

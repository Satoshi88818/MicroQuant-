```markdown
# MicroQuant V6 — Microtubule Lattice Quantum Processor Simulator

**Version 6.0**  
*Full implementation of the V6 roadmap: multi-parameter quantum Fisher information, variational quantum eigensolver (VQE), Floquet theory, surface-code quantum error correction on 13-PF cylinders, and molecular-dynamics-realistic disorder.*

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org)
[![QuTiP](https://img.shields.io/badge/QuTiP-5.x-green)](https://qutip.org)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## What is MicroQuant?

**MicroQuant** is a high-performance, open-source Python framework for simulating **microtubule lattices as quantum processors**. It models tubulin dimers arranged in a cylindrical geometry (13 protofilaments by default) and treats them as a network of interacting qubits under realistic biological conditions.

The simulator captures:
- Dipole–dipole interactions (nearest-neighbour and long-range)
- Polariton cavity coupling
- Thermal, phonon-correlated, and 1/f noise
- Continuous homodyne measurement and Bayesian feedback
- Full quantum evolution with GPU/JAX acceleration

**V6** dramatically expands the quantum-information capabilities, making MicroQuant a complete platform for studying quantum metrology, error correction, driven dynamics, and disorder in biological quantum systems.

---

## Key Features (V6)

| Module | Feature | Description |
|--------|---------|-------------|
| `analytics/QFIMatrix.py` | **Multi-parameter QFI** | Full Symmetric Logarithmic Derivative (SLD) tensor → n×n Quantum Fisher Information matrix for arbitrary generators. Cramér–Rao bounds and trajectory analysis. |
| `control/VQEPulse.py` | **Variational Quantum Eigensolver** | Hardware-efficient Ry + nearest-neighbour CZ ansatz with L-BFGS-B, COBYLA, or JAX-Adam optimisers. Returns optimal pulses and ground-state fidelity. |
| `analytics/FloquetAnalysis.py` | **Floquet Theory** | Quasi-energies, stroboscopic propagator, Magnus expansion (order 2), frequency scans, and heat-absorption rates for periodically driven lattices. |
| `qec/SurfaceCodeCylinder.py` | **Topological QEC** | Surface-code-like stabilisers on the native 13-PF cylinder topology. MWPM decoding (pymatching) or greedy fallback. Logical error rates vs. physical error rate. |
| `disorder/MDForceField.py` | **Realistic Disorder** | Load MD trajectories (NPZ/CSV/HDF5) or generate phonon-DOS-matched correlated Gaussian noise. Thermal scaling, spatial correlations, and power spectra. |

**All V5 capabilities are retained** (GRAPE optimal control, homodyne SME, finite-size scaling, quantum volume benchmarks, etc.).

---

## Strengths

- **Scientifically rigorous** — Implements state-of-the-art quantum information tools (full QFI tensor, Floquet–Magnus, surface-code adaptation to cylinder).
- **Biologically faithful** — Realistic tubulin parameters, temperature-dependent disorder, phonon DOS from literature, 13-PF geometry.
- **Highly modular & extensible** — Single `SimulationConfig` dataclass controls everything. New modules are self-contained.
- **Performance-oriented** — Optional JAX, CuPy, and GPU backends; efficient NumPy/QuTiP core.
- **Excellent testing & validation** — Comprehensive `test_v6.py` suite (shape checks, PSD, unitarity, Lyapunov equation, etc.) plus `run_full_suite.py` benchmarks.
- **Research-ready** — Trajectory analysis, summary statistics, and publication-quality outputs included.

---

## Current Limitations

- **Exponential scaling**: Classical simulation limited to small axial segments (N ≲ 8–10 sites per protofilament). Full 13-PF × N=6 systems (78 qubits) are feasible only for selected analyses.
- **QEC syndrome extraction** is approximate (especially X-stabilisers) for speed; exact projective measurement is available but slower.
- **VQE & GRAPE** currently limited to shallow circuits; deeper ansätze or tensor-network VQE not yet implemented.
- **No native support** for distributed computing or tensor-network backends (MPS/PEPS) for large-scale simulations.
- **Dependency on optional packages** (JAX, pymatching, h5py) for full feature set.
- **Biological interpretation** remains speculative — the code is a **simulation tool**, not a claim about microtubule quantum computation in vivo.

---

## Installation

```bash
# 1. Core dependencies
pip install qutip numpy scipy matplotlib plotly dash pytest

# 2. Optional but recommended accelerators
pip install jax jaxlib diffrax          # CPU / GPU JAX backend
# For NVIDIA GPU:
# pip install "jax[cuda12_pip]" jaxlib diffrax -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

pip install cupy-cuda12x                # CuPy GRAPE acceleration
pip install pymatching                  # Optimal MWPM decoder
pip install h5py                        # MD trajectory loading
```

**Quick verification**:
```bash
python experiments/run_full_suite.py
python -m pytest tests/test_v6.py -q
```

---

## Quick Start

```python
from core.config import SimulationConfig, InteractionModel, NoiseType
from core.QuantumLatticeProcessor import QuantumLatticeProcessor

cfg = SimulationConfig(
    N=6,
    n_protofilaments=13,
    interaction=InteractionModel.LONG_RANGE_DIPOLE,
    noise=NoiseType.THERMAL_UNIFORM,
    temperature_K=310.0,
    qec_enabled=True,
    vqe_n_layers=3,
    floquet_drive_freq=0.5,
    md_disorder_file=None,          # or path to .npz/.h5
)

proc = QuantumLatticeProcessor(cfg)

# Run V6 analyses
F = proc.compute_qfi_matrix()           # Full QFI tensor
vqe_result = proc.run_vqe()             # Variational ground state
floquet_result = proc.run_floquet()     # Quasi-energy spectrum
qec_report = proc.run_qec(n_rounds=500) # Logical error rates
proc.apply_md_disorder()                # Realistic disorder
```

Full examples are in `experiments/` and Jupyter notebooks (coming soon).

---

## Potential Uses & Applications

**Research domains**
- Quantum biology & microtubule coherence studies
- Quantum metrology in noisy biological environments
- Topological quantum error correction on non-Euclidean (cylindrical) lattices
- Floquet engineering of driven quantum systems
- Realistic noise modelling for quantum sensing in cells

**Specific use cases**
- Benchmarking multiparameter estimation protocols (QFI matrix)
- Designing variational pulse sequences for ground-state preparation
- Testing fault-tolerant thresholds on microtubule-like geometries
- Comparing synthetic vs. MD-derived disorder effects on coherence
- Educational platform for teaching quantum simulation, QEC, and Floquet physics
- Precursor to hybrid quantum-classical algorithms running on future biological or synthetic microtubule-inspired hardware

---

## Suggested Future Roadmap (V7 & beyond)

### Near-term (V6.1–V6.5)
- Tensor-network backends (QuTiP + MPS via `quimb` or `tensornetwork`)
- Adaptive Trotter / higher-order Magnus for strong driving
- Full projective syndrome measurement + circuit-level noise
- Automated parameter sweeps & hyperparameter optimisation (Optuna integration)

### Medium-term (V7)
- Machine-learning-assisted state preparation (quantum GANs, reinforcement learning)
- Hybrid quantum-classical optimisation (VQE + classical MD feedback)
- Open-source web dashboard (Dash/Streamlit) for interactive lattice visualisation
- Support for additional QEC codes (colour codes, heavy-hex, etc.)
- Integration with real experimental microtubule data (cryo-EM, spectroscopy)

### Long-term vision
- Cloud/HPC deployment for large-scale simulations
- Interface to quantum hardware simulators (e.g., IBM Qiskit, Rigetti, neutral atoms)
- Digital twin of living microtubule networks
- Quantum advantage demonstrations in biological metrology

---

## Contributing

We welcome contributions! Please see `CONTRIBUTING.md` (to be created).

1. Fork the repo
2. Create a feature branch
3. Add tests and update documentation
4. Submit a PR

---

## Citation

If you use MicroQuant in your research, please cite:

```bibtex
@software{MicroQuantV6,
  author = {MicroQuant Development Team},
  title  = {MicroQuant V6: Microtubule Lattice Quantum Processor Simulator},
  year   = {2026},
```

---

## License

MIT License 

---

**Built with curiosity, rigour, and a love for quantum biology.**

---

*Last updated: April 2026 by James Squire*
```


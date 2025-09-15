# Poincaré Disk Analysis with Centered Metrics

A Python toolkit for visualizing strain energy density on the Poincaré disk with fundamental domain boundaries and metric centering transformations.

## Overview

This code implements the Poincaré disk model for visualizing 2D metric tensors through stereographic projection. It applies centering transformations using a gamma parameter to shift the reference point in metric space, followed by Lagrange reduction to map metrics to the fundamental domain.

## Features

- **Stereographic projection** with centering transformation using γ = (4/3)^(1/4)
- **Lagrange reduction** of metric tensors for fundamental domain analysis
- **Fundamental domain boundary visualization** showing C₁₂ = 0, C₁₂ = C₁₁, and C₁₂ = C₂₂
- **Strain energy density computation** for square and triangular lattices
- **Interactive contour labeling** at midpoints of boundary curves

## Mathematical Background

The fundamental domain is defined by the constraints:
- C₁₂ ≥ 0
- C₂₂ ≥ C₁₁  
- 2C₁₂ ≤ C₁₁

The centering transformation uses matrix H with γ = (4/3)^(1/4):
```
H = γ * [[√(2+√3)/2, √(2-√3)/2], 
         [√(2-√3)/2, √(2+√3)/2]]
```

## Installation

```bash
git clone https://github.com/cnrs-oguzumut/poincare-disk-analysis.git
cd poincare-disk-analysis
pip install -r requirements.txt
```

## Quick Start

Run the main analysis:
```bash
python main.py
```

This generates a PDF visualization in the `figures/` directory showing the energy landscape on the Poincaré disk with fundamental domain boundaries.

## Usage Examples

### Basic Usage
```python
from src.projections import Cij_from_stereographic_projection_tr
from src.lagrange import lagrange_reduction
from src.energy import interatomic_phi0_from_Cij
from src.plotting import poincare_plot_energy_with_precise_boundaries

# Generate coordinates
x, y = np.linspace(-.999, .999, num=1000), np.linspace(-.999, .999, num=1000)
x_, y_ = np.meshgrid(x, y)

# Apply centering transformation
c11, c22, c12, c11t, c22t, c12t = Cij_from_stereographic_projection_tr(x_, y_)

# Perform Lagrange reduction
c11_red, c22_red, c12_red, iterations = lagrange_reduction(c11, c22, c12)

# Compute energy and visualize
phi0 = interatomic_phi0_from_Cij(c11_red, c22_red, c12_red, 'square')
```

### Run Examples
```bash
python examples/basic_usage.py
```

## Project Structure

```
poincare-disk-analysis/
├── README.md
├── requirements.txt
├── main.py                    # Main execution script
├── src/
│   ├── __init__.py
│   ├── projections.py         # Stereographic projections & centering
│   ├── lagrange.py           # Lagrange reduction algorithms
│   ├── energy.py             # Energy density calculations
│   └── plotting.py           # Visualization functions
├── examples/
│   └── basic_usage.py        # Usage examples
├── tests/
│   └── test_projections.py   # Unit tests
└── figures/                  # Output directory
```

## Parameters

### Lattice Types
- `'square'` - Square lattice with c₂ = c₁
- `'triangular'` - Triangular lattice with c₂ = 0

### Energy Scale
- Modify `c_scale` in main.py (default: 1e-19 for interatomic potential)

### Discretization
- Adjust `disc` parameter (default: 1000) for resolution vs. speed tradeoff

## Output

The main script generates:
- `figures/poincare_disk_analysis_1.pdf` - Energy landscape with fundamental domain boundaries
- Black contour lines showing the three fundamental domain boundaries
- Labels positioned at contour midpoints for clarity

## Dependencies

- numpy >= 1.20.0
- matplotlib >= 3.3.0  
- scipy >= 1.7.0

## Testing

Run unit tests:
```bash
python -m pytest tests/ -v
```

Or run individual test files:
```bash
python tests/test_projections.py
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{poincare_disk_analysis,
  author = {Oguz-Umut Salman},
  title = {Poincaré Disk Analysis with Centered Metrics},
  url = {https://github.com/cnrs-oguzumut/poincare-disk-analysis},
  year = {2024}
}
```

## Contact

- **Author**: Oguz-Umut Salman
- **Institution**: CNRS, Université Sorbonne Paris Nord
- **GitHub**: [@cnrs-oguzumut](https://github.com/cnrs-oguzumut)

## Acknowledgments

- Developed for research in reconstructive phase transitions and plasticity analysis
- Based on the Poincaré representation for mapping atomistic simulations to continuum mechanics

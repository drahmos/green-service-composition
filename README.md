# Green Service Composition

**Energy-Aware Web Service Composition: A Multi-Objective Optimization Framework for Sustainable Cloud Computing**

A novel research framework for optimizing web service compositions with energy consumption as a first-class optimization objective, developed for ICWS conference submission.

## Overview

This repository contains:
- 🌍 **Energy Consumption Model** for web services
- ⚡ **Multi-Objective Optimization Algorithms** (NSGA-II-Green, MOPSO-Green)
- 📊 **Comprehensive Evaluation Framework** with statistical analysis
- 📈 **Visualization Tools** for Pareto front analysis
- 📁 **Synthetic Dataset Generator** for reproducible experiments

## Quick Start

```python
from src.algorithms import NSGAIIGreen
from src.datasets import generate_dataset

# Generate a test composition problem
problem = generate_dataset(num_categories=10, services_per_category=20)

# Run optimization
algorithm = NSGAIIGreen(population_size=100, num_generations=200)
pareto_front = algorithm.optimize(problem)

print(f"Found {len(pareto_front)} Pareto-optimal solutions")
```

## Installation

```bash
git clone https://github.com/drahmos/green-service-composition.git
cd green-service-composition
pip install -r requirements.txt
```

## Running on Google Colab

See [`docs/tutorial.ipynb`](docs/tutorial.ipynb) for a complete Colab-compatible tutorial.

## Documentation

- [Technical Specification](TECHNICAL_SPECIFICATION.md)
- [API Reference](docs/api_reference.md)
- [Tutorial](docs/tutorial.ipynb)

## Results

Experimental results and visualizations are published in [`results/`](results/).

## Citation

If you use this code, please cite:

```bibtex
@misc{green-service-composition-2025,
  author = {Ahmed Moustafa},
  title = {Energy-Aware Web Service Composition: A Multi-Objective Optimization Framework},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/drahmos/green-service-composition}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

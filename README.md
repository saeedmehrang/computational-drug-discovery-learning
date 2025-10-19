# Computational Drug Discovery Learning

A simple repo for collecting learning code snippets related to Computational Drug Discovery.

## 🚀 Quick Start (30 Seconds)

```bash
# Clone/navigate to project
cd computational-drug-discovery-learning

# Install all dependencies (production + dev)
uv sync

# Activate environment
source .venv/bin/activate

# Install the modules as package
uv pip install -e .

# Run tests
pytest tests/ -v

# Run main script
python torsional_diffusion.py
```

## 📋 Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Testing](#testing)
- [Development](#development)
- [Usage](#usage)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## 🔬 Overview

This project implements a **torsional diffusion model** for generating molecular conformations. Key features:

- **EGNN Backbone**: SE(3)-equivariant message passing for molecular geometry
- **Torsional Representation**: Models only flexible degrees of freedom (more efficient than full 3D diffusion)
- **Circular Diffusion**: Handles periodic nature of torsion angles [-π, π]
- **Heavy-Atom Torsions**: Correctly identifies and extracts only chemically meaningful torsion angles

### Key Fix: Heavy-Atom Torsion Extraction

The `MolecularTorsionAnalyzer` was fixed to extract only **heavy-atom torsions** (excluding hydrogen atoms):
- **Before**: Ibuprofen returned 8 torsion angles (including H-C-C-H dihedrals)
- **After**: Ibuprofen correctly returns 4 heavy-atom torsions ✅

## 💻 Installation

### Prerequisites

- Python 3.13+
- [uv](https://github.com/astral-sh/uv) package manager

Install uv if you haven't already:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Setup

```bash
# Install all dependencies (including dev dependencies)
uv sync

# This will:
# 1. Create .venv/ directory
# 2. Install production dependencies (torch, rdkit, etc.)
# 3. Install dev dependencies (pytest, black, ruff)
# 4. Install package in editable mode
# 5. Lock versions in uv.lock
```

### Activating Environment

```bash
# Linux/Mac
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

## 📁 Project Structure

```
.
├── torsional_diffusion.py   # Main implementation
│   ├── MolecularTorsionAnalyzer   # Torsion extraction
│   ├── EGNN_Layer                 # Equivariant GNN layer
│   ├── TorsionDenoiser            # Noise prediction network
│   └── TorsionalDiffusionModel    # Complete diffusion model
│
├── tests/
│   └── test_torsion.py      # Pytest test suite (30 tests)
│
├── pyproject.toml           # Project metadata and dependencies
├── uv.lock                  # Locked dependency versions
├── .venv/                   # Virtual environment
└── README.md                # This file
```

## 📦 Dependencies

### Production Dependencies

Defined in `[project.dependencies]`:

| Package | Version | Purpose |
|---------|---------|---------|
| **torch** | ≥2.0.0 | PyTorch for deep learning |
| **torch-geometric** | 2.7.0 | Graph neural networks |
| **numpy** | ≥1.24.0 | Numerical computing |
| **matplotlib** | ≥3.7.0 | Visualization |
| **rdkit** | 2025.9.1 | Cheminformatics toolkit |

### Development Dependencies

Defined in `[tool.uv.dev-dependencies]`:

| Package | Version | Purpose |
|---------|---------|---------|
| **pytest** | ≥8.4.2 | Testing framework |
| **black** | ≥23.0.0 | Code formatter |
| **ruff** | ≥0.1.0 | Fast linter |
| **ipykernel** | ≥7.0.1 | Jupyter support |

### Managing Dependencies

```bash
# Add production dependency
uv add scipy

# Add dev dependency
uv add --dev pytest-cov

# Remove dependency
uv remove package-name

# Update all dependencies
uv sync --upgrade

# Install only production dependencies (no dev tools)
uv sync --no-dev
```

## 🧪 Testing

The project includes a comprehensive pytest test suite with **30 tests** covering:

### Test Categories

1. **TestTorsionExtraction** (21 tests)
   - Linear alkanes (propane, butane, pentane, hexane)
   - Branched alkanes (isobutane, 2-methylbutane, 2-methylpentane)
   - Aromatic compounds (benzene, toluene, ethylbenzene, propylbenzene)
   - Drug molecules (ibuprofen, aspirin, caffeine)
   - Functional groups (alcohols, ethers, amines, amides, acids)

2. **TestTorsionProperties** (4 tests)
   - All dihedral atoms are heavy atoms (not hydrogen)
   - Torsion angles are in [-π, π] range
   - Torsion count ≤ rotatable bond count
   - Bond indices appear in dihedral quartets

3. **TestIbuprofenRegression** (2 tests)
   - Regression test for the original bug (8 torsions → 4 torsions)
   - All Ibuprofen torsions use only heavy atoms

4. **TestEdgeCases** (3 tests)
   - Molecules with no rotatable bonds
   - Terminal bonds are excluded
   - Minimum size requirements for torsions

### Running Tests

```bash
# Activate environment
source .venv/bin/activate

# Install the modules as package
uv pip install -e .

# Run all tests with verbose output
pytest tests/ -v

# Run with quiet output (summary only)
pytest tests/ -q

# Run specific test class
pytest tests/test_torsion.py::TestIbuprofenRegression -v

# Run with coverage (if pytest-cov is installed)
pytest tests/ --cov=torsional_diffusion

# Run from test file directly
python tests/test_torsion.py
```

### Test Results

```
============================== test session starts ==============================
30 passed, 1 warning in 3.84s
```

**Success Rate: 100% ✓**

All tests confirm:
- Heavy-atom torsion extraction works correctly
- The Ibuprofen bug is fixed (4 torsions, not 8)
- Edge cases are handled properly
- Torsion properties are valid

## 🛠️ Development

### Code Quality Tools

#### Formatting with Black
```bash
# Format all Python files
black *.py tests/*.py

# Check formatting without changing files
black --check *.py
```

#### Linting with Ruff
```bash
# Lint all files
ruff check *.py tests/*.py

# Auto-fix issues
ruff check --fix *.py
```

### Development Workflow

```bash
# 1. Create feature branch
git checkout -b feature-name

# 2. Install dev dependencies and the modules as package
uv sync
uv pip install -e .

# 3. Make changes to code

# 4. Run tests
pytest tests/ -v

# 5. Format code
black *.py tests/*.py

# 6. Lint code
ruff check *.py tests/*.py

# 7. Commit and push
git add .
git commit -m "Description of changes"
git push origin feature-name
```

### Jupyter Notebooks

The dev environment includes `ipykernel` for Jupyter support:

```bash
# Start Jupyter
jupyter notebook

# Or use VS Code's built-in notebook support
# The kernel is auto-detected as "comp-drug-discovery-learning"
```

### VS Code Integration

Recommended settings (`.vscode/settings.json`):

```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
  "python.formatting.provider": "black",
  "python.linting.enabled": true,
  "python.linting.ruffEnabled": true,
  "python.testing.pytestEnabled": true,
  "python.testing.pytestArgs": ["tests/", "-v"]
}
```

### Import Fix for Tests in Subdirectory

When tests are in a `tests/` subdirectory, install the package in editable mode:

```bash
uv pip install -e .
```

This allows tests to import from `torsional_diffusion` module regardless of directory structure.

## 🎯 Usage

### Basic Example

```python
from rdkit import Chem
from rdkit.Chem import AllChem
from torsional_diffusion import MolecularTorsionAnalyzer

# Create molecule from SMILES
smiles = 'CC(C)Cc1ccc(cc1)C(C)C(=O)O'  # Ibuprofen
mol = Chem.MolFromSmiles(smiles)
mol = Chem.AddHs(mol)

# Generate 3D coordinates
AllChem.EmbedMolecule(mol)
AllChem.MMFFOptimizeMolecule(mol)

# Extract torsion angles
analyzer = MolecularTorsionAnalyzer()
torsion_info = analyzer.extract_torsion_angles(mol)

print(f"Found {len(torsion_info)} torsion angles")
for i, torsion in enumerate(torsion_info, 1):
    print(f"Torsion {i}: {torsion.dihedral_atoms} = {torsion.angle:.2f} rad")
```

### Output
```
Found 4 torsion angles
Torsion 1: (0, 1, 3, 4) = -2.99 rad
Torsion 2: (1, 3, 4, 5) = -1.88 rad
Torsion 3: (6, 7, 10, 11) = 1.06 rad
Torsion 4: (7, 10, 12, 13) = 0.56 rad
```

### Running the Main Script

```bash
python torsional_diffusion.py
```

This will:
1. Process 8 example molecules
2. Extract torsion angles
3. Train a torsional diffusion model
4. Generate new conformers
5. Display results

## 🐛 Troubleshooting

### "uv not found"
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### "Module not found" errors
Ensure dependencies are installed and venv is activated:
```bash
uv sync
source .venv/bin/activate
```

### Import errors in tests (tests/ subdirectory)
Install package in editable mode:
```bash
uv pip install -e .
```

### Dependency conflicts
Update lock file:
```bash
rm uv.lock
uv sync
```

### Complete fresh start
```bash
rm -rf .venv uv.lock
uv sync
```

### VS Code doesn't recognize imports
1. Press `Cmd+Shift+P` (or `Ctrl+Shift+P`)
2. Type "Python: Select Interpreter"
3. Choose `.venv/bin/python`

## 📄 License

See LICENSE file for details.

## 🙏 Acknowledgments

This implementation is based on modern research in:
- Torsional diffusion models
- Equivariant graph neural networks
- Molecular conformation generation

---


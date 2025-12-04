# Environment Setup Guide for Mac Apple Silicon

## Quick Start (Recommended)

### Step 1: Clone the Repository
```bash
git clone <repo-url>
cd MAPF-NeurIPS-25
```

### Step 2: Create Environment from YAML
```bash
conda env create -f environment_apple_silicon.yml
```

### Step 3: Activate Environment
```bash
conda activate mapf_env
```

### Step 4: Verify Installation
```bash
python -c "import torch; print(torch.__version__); print(torch.device('mps' if torch.backends.mps.is_available() else 'cpu'))"
```

---

## Environment Files Provided

### 1. `environment_apple_silicon.yml` (RECOMMENDED)
- **Best for**: Fresh installation on Mac Apple Silicon
- **Benefits**:
  - Uses conda-forge for better Apple Silicon (osx-arm64) compatibility
  - Flexible version constraints (not pinned)
  - Cleaner, more maintainable
  - Resolves dependencies automatically
- **Size**: Smaller, faster to download

### 2. `mapf_env.yaml` (ORIGINAL)
- **Original environment from development machine**
- Pinned to exact versions
- May have architecture-specific packages
- Use only if `environment_apple_silicon.yml` fails

---

## Detailed Installation Instructions

### Option A: Using `environment_apple_silicon.yml` (BEST)

```bash
# Create environment
conda env create -f environment_apple_silicon.yml

# Activate
conda activate mapf_env

# Verify
python -c "import torch, torchvision, torchaudio; print('PyTorch:', torch.__version__)"
```

### Option B: Using Original `mapf_env.yaml`

```bash
# Create environment (may need to resolve some packages)
conda env create -f mapf_env.yaml

# If it fails, try with channel priority:
conda env create -f mapf_env.yaml --channel-priority strict

# Activate
conda activate mapf_env
```

### Option C: Manual Installation (If YAML Files Fail)

```bash
# Create environment
conda create -n mapf_env python=3.10

# Activate
conda activate mapf_env

# Add conda-forge channel
conda config --add channels conda-forge

# Install core dependencies
conda install pytorch torchvision torchaudio numpy scipy pandas

# Install visualization
conda install matplotlib seaborn opencv pillow plotly

# Install utilities
conda install pyyaml requests tqdm jupyter ipython

# Install via pip
pip install gymnasium torch-geometric imageio moviepy rich tqdm
```

---

## Architecture Notes for Apple Silicon

### Hardware Detection
```bash
# Check if Apple Silicon is available:
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
```

### PyTorch on Apple Silicon
- **CPU**: Works on all Macs
- **MPS (Metal Performance Shaders)**: Native GPU acceleration for Apple Silicon
- **Install**: `pytorch::pytorch::*=*cpu*` (use CPU for stability, works with MPS fallback)

### Key Differences from Intel Mac
| Aspect | Apple Silicon | Intel Mac |
|--------|---------------|-----------|
| Architecture | ARM64 (osx-arm64) | x86_64 |
| GPU Support | MPS (Metal) | CUDA (NVIDIA) |
| Python Version | 3.10+ recommended | 3.9+ |
| Conda Channel | conda-forge | defaults |

---

## Troubleshooting

### Issue: `UnsatisfiableError` when creating environment

**Solution 1**: Try with channel priority
```bash
conda env create -f environment_apple_silicon.yml --channel-priority strict
```

**Solution 2**: Update conda
```bash
conda update conda
conda update conda-forge::conda
```

**Solution 3**: Use mamba (faster solver)
```bash
conda install mamba
mamba env create -f environment_apple_silicon.yml
```

### Issue: PyTorch not found or wrong version

**Solution**: Explicitly install PyTorch for Mac
```bash
conda activate mapf_env
conda install -c pytorch pytorch torchvision torchaudio
```

### Issue: OpenCV or ffmpeg missing

**Solution**: Install via conda-forge
```bash
conda activate mapf_env
conda install -c conda-forge opencv ffmpeg
```

### Issue: Permission denied errors

**Solution**: Verify conda installation
```bash
conda info --all
# Check that conda is properly initialized
conda init zsh  # or bash, depending on your shell
```

---

## Verifying Your Installation

### Complete Verification Script
```bash
# Activate environment
conda activate mapf_env

# Run these commands
python -c "
import sys
import torch
import numpy as np
import pandas as pd
import opencv_python
import matplotlib
import gymnasium

print('✓ Python:', sys.version)
print('✓ PyTorch:', torch.__version__)
print('✓ MPS Available:', torch.backends.mps.is_available())
print('✓ NumPy:', np.__version__)
print('✓ Pandas:', pd.__version__)
print('✓ Matplotlib:', matplotlib.__version__)
print('✓ Gymnasium:', gymnasium.__version__)
print('All core dependencies loaded successfully!')
"
```

---

## Saving Your Environment (After Installation)

### Export Current Environment
```bash
# Full export (with exact versions)
conda env export > environment_exact.yml

# Export without build specs
conda env export --no-builds > environment_clean.yml

# Export only explicitly installed packages
conda env export --from-history > environment_history.yml
```

### Creating Portable Requirements
```bash
# Create pip requirements for sharing
pip freeze > requirements.txt

# Later recreate with:
pip install -r requirements.txt
```

---

## Updating Environment

### Add New Package
```bash
conda activate mapf_env
conda install package_name
# OR
pip install package_name
```

### Update All Packages
```bash
conda activate mapf_env
conda update --all
```

### Update Specific Package
```bash
conda activate mapf_env
conda install package_name=desired_version
```

---

## Environment Comparison

### Current Development Machine
- **OS**: macOS (Apple Silicon)
- **Python**: 3.10.16
- **PyTorch**: 2.6.0.dev20241112
- **Key Packages**: Listed in mapf_env.yaml
- **Channels**: pytorch-nightly, conda-forge, defaults

### Recommended for Reproduction
- **OS**: macOS (Apple Silicon)
- **Python**: 3.10.x
- **PyTorch**: 2.x (stable, via conda-forge)
- **Key Packages**: Same functionality, flexible versions
- **Channels**: conda-forge, defaults

---

## Platform-Specific Considerations

### macOS Big Sur (11) and Later
- Recommended for Apple Silicon
- Full Metal Performance Shaders support

### Monterey (12) and Later
- Recommended for optimal performance
- Better MPS stability

### Ventura (13) and Later
- Latest support for all Python packages
- Best compatibility

---

## Quick Reference

```bash
# Typical workflow
conda create -f environment_apple_silicon.yml  # Create
conda activate mapf_env                        # Activate
conda list                                     # See all packages
conda search package_name                      # Search
conda install package_name                     # Install
conda remove package_name                      # Remove
conda env remove -n mapf_env                   # Delete environment
```

---

## Getting Help

If you encounter issues:

1. Check Python version: `python --version` (should be 3.10+)
2. Check conda version: `conda --version`
3. Check channel priorities: `conda config --show channels`
4. Try updating conda: `conda update -n base conda`
5. Try mamba solver: `pip install mamba-solver`

For PyTorch issues, visit: https://pytorch.org/get-started/locally/

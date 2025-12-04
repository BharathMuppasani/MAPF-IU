# Quick Start: Mac Apple Silicon Setup

## 🚀 Ultra-Fast Setup (5 minutes)

### Prerequisites
- Mac with Apple Silicon (M1/M2/M3/M4)
- Conda or Miniconda installed
- Terminal (zsh or bash)

### One-Command Installation

```bash
conda env create -f environment_apple_silicon.yml && conda activate mapf_env
```

Done! ✅

---

## ✅ Verify Installation

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('MPS:', torch.backends.mps.is_available())"
```

Expected output:
```
PyTorch: 2.x.x
MPS: True
```

---

## 📁 File Guide

| File | Purpose | When to Use |
|------|---------|------------|
| `environment_apple_silicon.yml` | **RECOMMENDED** - Clean, flexible, Apple Silicon optimized | Fresh install on new Mac |
| `mapf_env.yaml` | Original environment with pinned versions | If Apple Silicon version fails |
| `requirements.txt` | Python packages only (pip) | As backup or reference |
| `ENVIRONMENT_SETUP.md` | Detailed setup guide with troubleshooting | Need help or deep understanding |
| `QUICKSTART_MAC.md` | This file - fastest setup | Just want to run experiments |

---

## 🔧 If Something Goes Wrong

### Error: `UnsatisfiableError`
```bash
# Try with conda-forge:
conda remove -n mapf_env --all
conda create -f environment_apple_silicon.yml --channel-priority strict
```

### Error: PyTorch not working
```bash
conda activate mapf_env
conda install pytorch torchvision torchaudio -c pytorch
```

### Error: OpenCV missing
```bash
conda activate mapf_env
conda install opencv -c conda-forge
```

### Start Fresh (Nuclear Option)
```bash
conda env remove -n mapf_env
conda env create -f environment_apple_silicon.yml
conda activate mapf_env
```

---

## 📊 Performance Notes

### GPU (Metal Performance Shaders)
```python
# Check if using GPU:
import torch
print(torch.backends.mps.is_available())  # Should be True
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
```

### Expected Performance
- **M1/M2**: ~2-3x faster than CPU for some operations
- **M3/M4**: ~3-4x faster
- **CPU fallback**: Always available, works fine for MAPF

---

## 📦 Managing Environment

```bash
# List all packages
conda list

# Add package
conda install package_name

# Update package
conda install package_name=desired_version

# Remove package
conda remove package_name

# Switch between environments
conda activate mapf_env
conda deactivate

# Save current state
conda env export > my_environment.yml
```

---

## 🎯 Common Tasks

### Run Experiments
```bash
conda activate mapf_env
python run_exp.py --args
```

### Training
```bash
conda activate mapf_env
python train.py --config ...
```

### Visualization
```bash
conda activate mapf_env
python visualize_plan.py
```

### Jupyter Notebook
```bash
conda activate mapf_env
jupyter notebook
```

---

## 🔍 What's in the Environment?

### Core Libraries
- **PyTorch**: Deep Learning framework
- **NumPy/Pandas**: Numerical computing
- **Matplotlib/Seaborn**: Visualization

### MAPF-Specific
- **Gymnasium**: RL environment
- **OpenCV**: Computer vision
- **NetworkX**: Graph algorithms

### Utilities
- **Jupyter**: Interactive notebooks
- **tqdm**: Progress bars
- **Rich**: Colored output

---

## 📝 System Info

### Check Your Machine
```bash
# Check architecture
uname -m
# Should output: arm64 (Apple Silicon)

# Check macOS version
sw_vers
# Should be Big Sur (11) or later

# Check conda
conda info
# Should show: platform: osx-arm64
```

### Python Version
```bash
python --version
# Should be 3.10.x
```

---

## 🆘 Need Help?

1. **Check ENVIRONMENT_SETUP.md** - Detailed guide with all options
2. **Check PyTorch docs**: https://pytorch.org
3. **Common conda errors**: Visit https://docs.conda.io/

---

## 💡 Pro Tips

### Speed Up Installation
```bash
# Use mamba (faster solver)
pip install mamba-solver
conda install -f environment_apple_silicon.yml --solver=mamba
```

### Keep Multiple Environments
```bash
# Clone environment
conda create --name mapf_env_backup --clone mapf_env
```

### Save Current State
```bash
# After making changes, save
conda env export > environment_snapshot.yml
```

---

## 📌 Cheat Sheet

```bash
# Setup
conda env create -f environment_apple_silicon.yml

# Activate
conda activate mapf_env

# Verify
python -c "import torch; print(torch.__version__)"

# Run
python run_exp.py

# Deactivate
conda deactivate
```

---

## ✨ You're All Set!

Your environment is ready to:
- Run MAPF experiments
- Train RL agents
- Visualize planning results
- Generate reports

Happy experimenting! 🎉

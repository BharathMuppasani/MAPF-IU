# Conda Environment Setup Guide

## 📦 Files Created for Environment Reproduction

### For Apple Silicon Mac (RECOMMENDED)

#### 1. **`environment_apple_silicon.yml`** ⭐ PRIMARY
- **Purpose**: Clean, optimized environment for Mac Apple Silicon
- **Best for**: Fresh installation on new Mac with M1/M2/M3/M4
- **Features**:
  - Uses conda-forge (better ARM64 support)
  - Flexible version constraints
  - Fast dependency resolution
  - Apple Silicon optimized PyTorch (CPU with MPS support)
- **Size**: ~800 MB installed
- **Setup Time**: 5-10 minutes

**Usage**:
```bash
conda env create -f environment_apple_silicon.yml
conda activate mapf_env
```

---

#### 2. **`mapf_env.yaml`** (ORIGINAL)
- **Purpose**: Original environment from development machine
- **Best for**: Exact reproduction if Apple Silicon version fails
- **Features**:
  - Pinned exact versions
  - All dependencies explicitly listed
  - PyTorch nightly channel versions
- **Size**: ~1.2 GB installed
- **Setup Time**: 10-15 minutes

**Usage**:
```bash
conda env create -f mapf_env.yaml
conda activate mapf_env
```

---

### For Pip Installation (BACKUP)

#### 3. **`requirements.txt`**
- **Purpose**: Python packages list for pip
- **Best for**: As reference or if conda fails completely
- **Important**: PyTorch should be installed via conda first!

**Usage**:
```bash
# Install PyTorch first (via conda)
conda create -n mapf_env python=3.10
conda activate mapf_env
conda install pytorch torchvision torchaudio

# Then install other packages
pip install -r requirements.txt
```

---

### Documentation Files

#### 4. **`ENVIRONMENT_SETUP.md`** (COMPREHENSIVE)
- **7 KB reference guide**
- Complete instructions for all scenarios
- Troubleshooting section
- Architecture-specific notes
- Verification scripts

#### 5. **`QUICKSTART_MAC.md`** (FAST)
- **4.5 KB quick reference**
- Fastest setup path
- Common error fixes
- Performance tips
- Cheat sheet

#### 6. **`CONDA_SETUP_GUIDE.md`** (THIS FILE)
- Overview of all files
- Decision matrix
- Comparison table

---

## 🎯 Which File Should I Use?

### Decision Tree

```
┌─ New Mac, Apple Silicon? YES
│  └─ First time? YES
│     └─ Use: environment_apple_silicon.yml ⭐
│  └─ Had it before? YES
│     └─ Use: environment_from_history.yml or mapf_env.yaml
│
└─ Something failed? YES
   └─ Use: environment_apple_silicon.yml (with --channel-priority strict)
   └─ If that fails: Use mapf_env.yaml
   └─ If both fail: Use pip + requirements.txt
```

### Quick Comparison

| Aspect | `environment_apple_silicon.yml` | `mapf_env.yaml` | `requirements.txt` |
|--------|--------------------------------|-----------------|-------------------|
| **For Mac?** | ✅ YES (native) | ✅ YES (may need adjustments) | ✅ YES (with PyTorch first) |
| **Apple Silicon?** | ✅ Optimized | ⚠️ Original (may work) | ✅ Works fine |
| **Version Control** | Flexible (>=) | Fixed (==) | Flexible (>=) |
| **Install Time** | 5-10 min | 10-15 min | 10 min (PyTorch: 5 min) |
| **Size** | ~800 MB | ~1.2 GB | Variable |
| **Dependency Solver** | Faster | Slower | Pip (different) |
| **GPU Support** | MPS (Metal) | CPU/Nightly | CPU/MPS |
| **Recommended** | ✅ YES | If above fails | Backup only |

---

## 📋 Step-by-Step Instructions

### Scenario 1: Fresh Mac, First Time Setup

```bash
# 1. Clone repo
git clone <repo>
cd MAPF-NeurIPS-25

# 2. Create environment (1 command)
conda env create -f environment_apple_silicon.yml

# 3. Activate
conda activate mapf_env

# 4. Verify
python -c "import torch; print(torch.__version__)"

# 5. Run experiments
python run_exp.py
```

**Expected**: Complete in ~10 minutes ✅

---

### Scenario 2: Existing Mac, Replicating Environment

```bash
# 1. Check if environment exists
conda env list

# 2. Remove old environment (if exists)
conda env remove -n mapf_env

# 3. Create fresh environment
conda env create -f environment_apple_silicon.yml

# 4. Verify
python -c "import torch; import gymnasium; print('OK')"
```

**Expected**: Complete in ~10 minutes ✅

---

### Scenario 3: Installation Fails

```bash
# 1. Try with stricter channel priority
conda env create -f environment_apple_silicon.yml --channel-priority strict

# 2. If still fails, use original
conda env remove -n mapf_env
conda env create -f mapf_env.yaml

# 3. If both fail, use pip method
conda create -n mapf_env python=3.10
conda activate mapf_env
conda install pytorch torchvision torchaudio
pip install -r requirements.txt
```

**Expected**: One of the three methods should work ✅

---

## 🔧 Environment Details

### Python Version
- **Configured**: Python 3.10.x
- **Minimum**: Python 3.8
- **Recommended**: Python 3.10+

### Key Packages

**Core Data Science**:
- numpy ≥ 2.0.0
- pandas ≥ 2.0.0
- scipy

**Machine Learning**:
- PyTorch 2.x (latest stable)
- Torchvision
- Torchaudio
- Gymnasium (RL)
- Torch-Geometric (Graph)

**Visualization**:
- Matplotlib 3.8+
- OpenCV 4.8+
- Plotly
- Seaborn

**Utilities**:
- Jupyter, IPython
- tqdm, Rich
- pyyaml, requests

### Mac-Specific Configuration

**PyTorch on Apple Silicon**:
```python
# GPU (MPS) - Automatic Fallback
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# Check availability
torch.backends.mps.is_available()  # Should be True on M1/M2/M3/M4
```

**Expected Performance**:
- M1/M2: CPU fallback works fine for MAPF
- M3/M4: Native MPS acceleration available
- All: Better than original setup in many cases

---

## 📊 Environment Comparison

### Original Development Environment (mapf_env.yaml)
```
OS: macOS (Apple Silicon)
Python: 3.10.16
PyTorch: 2.6.0.dev20241112 (nightly)
Channels: pytorch-nightly, conda-forge, defaults
Total Size: ~1.2 GB
```

### Optimized for Reproduction (environment_apple_silicon.yml)
```
OS: macOS (Apple Silicon)  ← Same
Python: 3.10.x  ← Same
PyTorch: 2.x stable  ← Updated
Channels: conda-forge, defaults  ← Simplified
Total Size: ~800 MB  ← Smaller
```

---

## ✅ Verification Checklist

After environment creation, verify with:

```bash
# 1. Activate environment
conda activate mapf_env

# 2. Check Python
python --version  # Should be 3.10.x

# 3. Check PyTorch
python -c "import torch; print(f'PyTorch {torch.__version__}')"

# 4. Check MPS (Metal GPU)
python -c "import torch; print(f'MPS Available: {torch.backends.mps.is_available()}')"

# 5. Check core packages
python -c "import numpy, pandas, gymnasium, torch_geometric; print('All OK!')"

# 6. List all packages
conda list
```

**Expected Output**:
```
Python 3.10.x
PyTorch 2.x.x
MPS Available: True
All OK!
```

---

## 🚀 Next Steps

1. **Create environment**: Use `environment_apple_silicon.yml`
2. **Activate**: `conda activate mapf_env`
3. **Verify**: Run verification commands above
4. **Explore**: Check `run_exp.py`, `train.py`, `visualize_plan.py`
5. **Experiment**: Start with simple scenarios

---

## 📚 Documentation Reference

| File | Size | Purpose |
|------|------|---------|
| `ENVIRONMENT_SETUP.md` | 7.0 KB | Complete guide + troubleshooting |
| `QUICKSTART_MAC.md` | 4.5 KB | Fast reference + cheat sheet |
| `CONDA_SETUP_GUIDE.md` | This file | Overview + comparison |
| `environment_apple_silicon.yml` | 1.0 KB | Configuration (use this!) |
| `mapf_env.yaml` | Original | Backup configuration |
| `requirements.txt` | 1.3 KB | Pip requirements |

---

## 🆘 Troubleshooting Quick Links

- **Can't find conda?** → See ENVIRONMENT_SETUP.md "Getting Help" section
- **UnsatisfiableError?** → Try with `--channel-priority strict`
- **PyTorch not working?** → Reinstall: `conda install pytorch -c pytorch`
- **OpenCV missing?** → `conda install opencv -c conda-forge`
- **MPS not available?** → Normal on Intel Macs, CPU fallback works

---

## 💡 Pro Tips for Mac Users

1. **Faster Installation**: Use `mamba` instead of `conda`
   ```bash
   pip install mamba-solver
   mamba env create -f environment_apple_silicon.yml
   ```

2. **Keep Backups**: Save current state
   ```bash
   conda env export > my_environment.yml
   ```

3. **Multiple Environments**: For A/B testing
   ```bash
   conda create --name mapf_env_v2 --clone mapf_env
   ```

4. **Check Updates**: Keep packages current
   ```bash
   conda update --all
   ```

---

## 📝 Summary

**TL;DR for Mac Apple Silicon:**

```bash
# 1. Create environment (pick ONE method)
conda env create -f environment_apple_silicon.yml  # ⭐ RECOMMENDED

# 2. Activate
conda activate mapf_env

# 3. Verify
python -c "import torch; print(torch.__version__)"

# 4. Start working
python run_exp.py
```

**That's it!** Your environment is ready. 🎉

---

## 📞 Support

- **Questions**: See `ENVIRONMENT_SETUP.md`
- **Quick help**: See `QUICKSTART_MAC.md`
- **Issues**: Check troubleshooting sections in both files
- **Advanced**: Visit conda docs: https://docs.conda.io/

---

*Last Updated: 2024*
*For: Mac Apple Silicon (M1/M2/M3/M4)*
*Python: 3.10+*
*PyTorch: 2.x*

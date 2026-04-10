# Learned-Approximate Message Passing under Karhunen-Loève Modeling for Fluid Antenna Systems

Fluid antenna systems (FASs) can harvest substantial spatial diversity with a limited number of radio-frequency (RF) chains, yet reliable channel acquisition remains challenging due to the large number of candidate ports and the scarcity of pilot resources. 

This repository implements our proposed **physics-guided compressive channel reconstruction framework** for FAS channel estimation. The framework incorporates:
- **Karhunen-Loève (KL) Basis**: Constructed from angle-of-arrival (AoA) statistics to yield an information-theoretically optimal low-dimensional representation of the FAS channel.
- **Block-Hopping Sampling Strategy**: Efficiently probes the spatial aperture for pilot acquisition with limited RF resources.
- **KL-LAMP Network**: A KL-domain learned approximate message passing network that unfolds AMP iterations with learnable parameters, enhancing robustness against noise and model mismatch.

The project simulates realistic FAS channels across both Rayleigh and Rician fading scenarios, evaluating the proposed KL-LAMP method alongside an enhanced baseline (FCNet) and traditional methods (OMP, LMMSE, and GAMP). Simulation results corroborate that this method achieves competitive NMSE performance in the medium-to-high signal-to-noise ratio (SNR) regime with a favorable performance-complexity tradeoff.

## Project Structure

- **`FAS_Channel_Model.py`**: Defines the physical channel model for FAS, including large-scale fading, small-scale spatial correlation, and directional Angle of Arrival (AoA) with Rician K-factor support.
- **`FAS_Dataset_Generator.py`**: Generates synthetic channel realization datasets (`.npz` files) for training and evaluation across different fading scenarios.
- **`FAS_KLLAMP.py`**: Implements and trains the KL-LAMP neural network architecture, which unfolds the AMP algorithm over the KL basis for efficient and accurate channel estimation.
- **`FAS_FCnet.py`**: Implements and trains an enhanced Fully Connected Network (FCNet) baseline for channel estimation comparison.
- **`eval_rician_scenarios.py`**: The main evaluation script. Loads datasets and trained models, evaluates all methods under various noise levels (SNR), and generates performance comparison plots (NMSE vs. SNR).

## Environment Requirements

The project is developed in Python and relies on PyTorch and standard scientific computing libraries.

### Dependencies
- Python 3.8+
- PyTorch (1.10+, CUDA support is recommended for faster training and evaluation)
- NumPy
- SciPy
- Matplotlib

### Environment Setup
You can easily create the required environment using Conda:

```bash
conda create -n KLLAMP python=3.10
conda activate KLLAMP
pip install numpy scipy matplotlib torch torchvision
```

## Usage Instructions

### 1. Dataset Generation
Prepare the synthetic channel data by running the dataset generator. It will create dataset files (`.npz`) for different scenarios like Rayleigh fading and Rician fading (e.g., K = 5dB, 10dB, 15dB).
```bash
python FAS_Dataset_Generator.py
```

### 2. Model Training
Train the deep learning models (LAMP and FCNet). The scripts will load the generated datasets and output the trained network weights as `.pth` checkpoint files.

```bash
python FAS_KLLAMP.py --mode all --epochs 200 --D 12 --M 8 --T 2 --S 20000
python FAS_FCnet.py --mode all --epochs 200 --D 12 --M 8 --T 2 --S 20000 --enhanced
```

**Customizable Command-Line Arguments:**
- `--mode`: Training mode, `all` to train all scenarios or `single` for Rayleigh only (default: `all`).
- `--epochs`: Number of training epochs (default: `200`).
- `--D`: KL basis dimension (default: `12`).
- `--M`: Number of sampled ports per pilot slot (default: `8`).
- `--T`: Number of pilot slots (default: `2`).
- `--S`: Number of channel samples used for training (default: `20000`).
- `--enhanced` / `--basic`: (FCNet only) Use the enhanced architecture with residual blocks and BatchNorm, or the basic MLP architecture (default: `--enhanced`).

### 3. Evaluation
Once the models are trained and checkpoints (e.g., `lamp_fas_best_...pth` and `fcnet_fas_best_...pth`) are available, run the comprehensive evaluation script. It supports multi-scenario testing and creates comparison plots.

```bash
python eval_rician_scenarios.py --D 12 --M 8 --T 2 --max_samples 2000 --omp_sparsity 10
```

**Customizable Command-Line Arguments:**
- `--D`: KL basis dimension (default: 12)
- `--M`: Number of sampled ports per pilot slot (default: 8)
- `--T`: Number of pilot slots (default: 2)
- `--max_samples`: Maximum number of random channel samples to evaluate (default: 2000)
- `--omp_sparsity`: OMP algorithm assumed sparsity level (default: 10)

This script outputs formatted performance metrics in the terminal and plots a 2x2 grid comparing OMP, LMMSE, GAMP, Genie-LMMSE, FCNet, and LAMP. The resulting figure will be saved automatically as `fig_nmse_combined.png`.

## Citation

If you find this code or our paper useful in your research, please consider citing our work:

```bibtex
@ARTICLE{11478428,
  author={Wu, Yuanhui and Zhang, Zhentian and Jiang, Hao and Wong, Kai-Kit and Chae, Chan-Byoung},
  journal={IEEE Wireless Communications Letters}, 
  title={Learned-Approximate Message Passing under Karhunen-Lo{\`e}ve Modeling for Fluid Antenna Systems}, 
  year={2026},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/LWC.2026.3682568}
}
```

# XAIPath Implementation

This repository contains the complete implementation of the XAIPath framework for temporal-environmental explainable AI in bacterial detection.

## Repository Structure

```
XAIPath/
├── src/                          # Source code
│   ├── xaipath_model.py         # Core XAIPath model implementation
│   ├── dataset.py               # Dataset handling and data loading
│   ├── train.py                 # Training pipeline
│   ├── evaluate.py              # Evaluation and analysis
│   └── utils.py                 # Utility functions
├── config.json                  # Configuration file
├── requirements.txt             # Python dependencies
├── main.py                     # Main pipeline script
└── generate_figures.py         # Figure generation script
```

## Key Features

### 1. XAIPath Model (`xaipath_model.py`)
- **Temporal Encoder**: Learnable sinusoidal embeddings for growth dynamics
- **Environmental Encoder**: Context modeling for biochemical conditions
- **Cross-Attention**: Integration of temporal and spatial features
- **Multi-Modal Explainability**: Grad-CAM, SHAP, and LIME integration
- **Consistency Constraints**: Temporal and environmental consistency losses

### 2. Dataset Management (`dataset.py`)
- **BacterialDataset**: Custom dataset class for microscopic images
- **Temporal Metadata**: Growth phase and environmental condition handling
- **Data Augmentation**: Biologically-realistic transformations
- **Stratified Splitting**: Balanced train/validation/test splits

### 3. Training Pipeline (`train.py`)
- **XAIPathTrainer**: Complete training framework
- **Multi-Stage Optimization**: Learning rate scheduling and early stopping
- **Comprehensive Metrics**: Precision, recall, F1-score, and explanation quality
- **Model Checkpointing**: Best model saving and restoration

### 4. Evaluation Framework (`evaluate.py`)
- **Performance Analysis**: Temporal and environmental robustness
- **Explainability Assessment**: Attention map quality and expert validation
- **Ablation Studies**: Component contribution analysis
- **Visualization Generation**: All paper figures

## Installation

1. **Clone the repository** (after extracting code.zip)
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Quick Start
```bash
# Generate all figures and results
python generate_figures.py

# Run complete pipeline (training + evaluation)
python main.py

# Run with custom configuration
python main.py --config custom_config.json
```

### Training Only
```bash
python src/train.py --data_dir /path/to/data --num_epochs 100
```

### Evaluation Only
```bash
python src/evaluate.py --model_path /path/to/model.pth
```

## Configuration

The `config.json` file contains all hyperparameters:

```json
{
  "model": {
    "temporal_dim": 128,
    "env_dim": 64,
    "lambda_temp": 0.1,
    "lambda_env": 0.05
  },
  "training": {
    "num_epochs": 100,
    "batch_size": 32,
    "learning_rate": 0.001
  }
}
```

## Key Results

- **Performance**: 94.7% precision, 91.3% recall, 92.9% F1-score
- **Temporal Robustness**: 89.4% accuracy in early growth phases
- **Environmental Robustness**: 92.1% ± 1.8% across conditions
- **Explainability**: 91.7% localization accuracy
- **Ablation Impact**: 11.9% drop without temporal+environmental components

## Model Architecture

### Temporal Encoding
```python
φ_t(t) = [sin(ω₁t + φ₁), cos(ω₁t + φ₁), ..., sin(ω_{d/2}t + φ_{d/2}), cos(ω_{d/2}t + φ_{d/2})]
```

### Environmental Gating
```python
F_env = g_e ⊙ F_visual + (1 - g_e) ⊙ F_baseline
```

### Combined Loss
```python
L_total = L_cls + λ_temp * L_temp + λ_env * L_env
```

## Data Format

The framework expects microscopic images with metadata:
- **Images**: High-resolution (1024×768) microscopic images
- **Labels**: 0=background, 1=Salmonella, 2=mixed culture
- **Time**: Growth time in hours (0.5-4.0)
- **Environment**: 0=without onion, 1=with onion

## Reproducibility

All experiments are reproducible with fixed random seeds:
```python
np.random.seed(42)
torch.manual_seed(42)
```

## Citation

If you use this code, please cite our paper:
```bibtex
@article{alsobeh2025xaipath,
  title={XAIPath: Temporal-Environmental Explainable AI Framework for Co-Contaminated Food Pathogen Detection in Microscopic Imaging},
  author={AlSobeh, Anas and AbuGhazaleh, Amer and Dhahir, Namariq and Rababa, Malek},
  journal={Conference Proceedings},
  year={2024}
}
```

## License

This project is licensed under the MIT License.

## Contact

For questions or issues, please contact:
- Anas AlSobeh: anas.alsobeh@siu.edu
- GitHub Issues: [Create an issue](https://github.com/aalosbeh/XAIPath/issues)

## Acknowledgments

This research is funded by the United States Department of Agriculture, National Institute of Food and Agriculture (USDA-NIFA), under Grant No. 2023-67017-39455.


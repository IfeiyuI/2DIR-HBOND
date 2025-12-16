# Dynamic Prediction of Protein Physicochemical Properties Using 2D IR Spectroscopy with Machine Learning

## Overview

This repository contains the source code for our study, which employs **two-dimensional infrared (2D IR) spectroscopy** as a machine learning descriptor to predict the physicochemical properties of proteins. By leveraging spectral signals as input, the model learns mappings between 2D IR spectra and protein physicochemical properties, enabling quantitative analysis of previously unseen spectra.

This work addresses two key scientific challenges:

1. **Feature Extraction from Spectral Data**: Developing machine learning models capable of capturing physicochemical properties embedded within complex spectral signals.
2. **Application to Protein Folding Dynamics**: Extending the predictive framework to model the dynamic folding processes of proteins, offering insights into their physicochemical changes.

![Project Overview](assets/fig1.png)
Workflow for protein structure prediction based on 2DIR spectroscopy. (A) shows the construction of the dataset, including the extraction of protein structures and their spectral features from both static and dynamic datasets. (B) details the generation of the pre-training dataset, illustrating how 2D IR spectra are simulated by mapping physicochemical properties such as hydrogen bond count, hydrophobic solvent-accessible surface area (SASA), isoelectric point (pI), protein length, radius of gyration (Rg), and contact order (CO) to the spectra. (C) presents the architecture of the Swin-Transformer-based model, which combines convolutional layers to extract spectral features and refines them through layer-by-layer convolution and self-attention mechanisms, ultimately predicting the physicochemical properties of the proteins.

---

## Repository Structure

```
- config.py                 # Defines hyperparameters and experiment settings.
- model.py                 # Loads and modifies a pre-trained Swin Transformer model for spectral data analysis.
- dataload.py              # Handles data preprocessing, including loading, normalization, and transformation.
- train_base_model.py      # Baseline training using static spectral data.
- train_transfer_model.py  # Transfer learning on dynamic protein spectra.
- test_base_model.py       # Evaluation of the baseline model.
- test_transfer_model.py   # Evaluation of the transfer learning model.
```

---

## Pretrained Model

The Swin Transformer backbone is initialized with ImageNet-pretrained weights.
Please download the pretrained model and place it in:

```bash
./swin_pretrain_model/
```

You can modify the path in `model.py` if needed.

---

## Dataset

The dataset used in this study is hosted on **Zenodo** for public access and reproducibility:

📥 [Download Dataset from Zenodo](https://zenodo.org/record/xxxxx)

After downloading:

```bash
mkdir data
mv downloaded_dataset.zip data/
unzip data/downloaded_dataset.zip -d data/
```

Dataset structure:
- `base/` folder: Static proteins (single structure)
- `dynamic/` folder: MD-simulated trajectories (6 proteins with multiple time frames)

Each sample includes:
- 2D IR spectrum image in PNG format
- Corresponding label Excel sheet with:
  - `H`: Hydrogen bond count
  - `SASA_Hydrophobic`: Hydrophobic surface area
  - `L`: Protein length
  - `pI`: Isoelectric point
  - `Rg`: Radius of gyration
  - `CO`: Contact order
---

## Usage

Run the baseline model training with:

```bash
python train_base_model.py --config configs/base_config.yaml
```

Train the transfer learning model with:

```bash
python train_transfer_model.py --config configs/transfer_config.yaml
```

To evaluate a trained model:

```bash
python model.py --input data/test_spectra/
```

---

## Results

The figures below summarize the performance of the proposed framework on both static and dynamic datasets, as well as its interpretability through attention-based spectral analysis.

![Static Dataset Results](assets/fig2.png)

Prediction results of the model pre-trained on the static dataset, covering six physicochemical properties: hydrogen bond count, hydrophobic SASA, isoelectric point (pI), protein length, contact order (CO), and radius of gyration (Rg). Diagonal dashed lines indicate ideal correlations.


![Dynamic Folding Results](assets/fig3.png)

Transfer learning results for protein folding trajectories simulated on the Anton supercomputer, showing time-resolved predictions of hydrogen bond count and hydrophobic SASA for four representative proteins (Trp-cage, α3D, cln025, and NUG2).

![Attention Analysis](assets/fig4.png)

Attention-based interpretation of 2D IR spectra during protein folding, illustrating how the model highlights spectral regions associated with changes in hydrogen bond patterns.

---

## Citation

This work is not yet published. If referencing, please cite as:

```bibtex
@misc{wang2025,
  author    = {Zhen Wang},
  title     = {Dynamic Prediction of Protein Physicochemical Properties Using 2D IR Spectroscopy with Machine Learning},
  year      = {2025},
  note      = {Manuscript in preparation}
}
```

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Contact

📧 Email: [feiyufei859@gmail.com](mailto:feiyufei859@gmail.com)  
🛠️ Or open an issue in this repository for questions or suggestions.


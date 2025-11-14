# 2TDformer: DSP-Enhanced Transformer for Weak Single Phase-to-Ground Fault Dual-Time Detection

Official implementation of the paper:

**Weak Single Phase-to-Ground Fault Time Detection with Learnable-Parameter-Driven DSP-Enhanced Transformer**  
*IEEE Transactions on Smart Grid*, 2025.

---

## ⭐ Overview

2TDformer is a DSP-enhanced Transformer designed for dual-time detection (fault initiation and duration) of weak and nonlinear Single Phase-to-Ground Faults (SPGFs) in Resonantly Grounded Distribution Networks (RGDNs).

It integrates two learnable DSP modules:

- **TWFA** — Transient Weak Feature-Enhancement Attention  
- **LNWR** — Learnable Nonlinear Waveform Reconstruction  

These modules amplify weak transients and capture nonlinear multi-scale spectral characteristics.  
2TDformer jointly optimizes all DSP parameters via backpropagation, significantly improving detection accuracy.

The model achieves:

- **97.10% detection accuracy**
- **≤ 1.4 ms initiation error**
- **≤ 2.8 ms duration error**
- **35–58% improvement** over 11 baseline models


## 📐 Model Architecture

The overall architecture of 2TDformer is illustrated in the figure below:

<p align="center">
  <img src="figures/structure.png" width="70%">
</p>

The structures of TWFA and LNWR are shown side-by-side below:

<p align="center">
  <img src="figures/TWFA.png" width="35%" style="margin-right: 2%;">
  <img src="figures/LNWR_new.png" width="34%">
</p>


## 🔧 Installation

Clone the repository:

```bash
git clone https://github.com/smartlab-hfut/2TDformer-for-Weak-Fault-Time-Detection.git
cd 2TDformer-for-Weak-Fault-Time-Detection
```


Install dependencies:
```
pip install -r requirements.txt
```

🚀 Usage
```
python main.py --data_path ./data --batch_size 16 --epochs 50 --lr 0.001 --num_heads 6  --d_hid 400 --d_inner 400 --n_layers 1 --dropout 0.0 --mul_dim 3 --step_size 50
```

## 📡 Dataset

This repository provides the processed training and testing datasets used for evaluating 2TDformer.  
All data files are stored in the `data/` directory:


### **Data Format**

- `train_data2.csv` / `test_data2.csv`  
  Contain waveform samples collected from distribution networks of SGCCAH-RWD.  
  Each sample corresponds to a sequence of **400 time steps × 6 electrical quantities**

- `train_label2.csv` / `test_label2.csv`  
Contain the corresponding binary fault labels with the same temporal length (400 points):

### **Signal Description**

Each sample contains six channels representing:

1. Phase voltages (Ua, Ub, Uc)  
2. Phase currents (Ia, Ib, Ic)

All signals are normalized and standardized automatically during preprocessing.

### **Usage in Code**

The training and testing datasets are loaded through:

```python
train_loader, test_loader = prepare_dataset("./data", batch_size=16)
```


## 📊 Experimental Results

This section presents key experimental results demonstrating the effectiveness of 2TDformer in weak SPGF dual-time detection, compared with baseline models and ablation variants.

---

### **1. Performance Comparison with Baseline Models**

<p align="center">
  <img src="figures/performance_heatmap.png" width="25%" style="margin-right: 2%;">
  <img src="figures/predict_line.png" width="35%">
</p>

2TDformer achieves the best performance across all major metrics, including  
**Accuracy (0.9710), Precision (0.9494), Recall (0.9578), F1-score (0.9486), and AUC (0.9961)**.


---

### **2. Ablation Study**

<p align="center">
  <img src="figures/ablation_radar.png" width="35%">
</p>

The ablation results demonstrate the contributions of TWFA and LNWR:

- Removing **TWFA** significantly reduces the detection accuracy.  
- Removing **LNWR** affects modeling of nonlinear arc-restriking behavior.  
- Using a fixed STFT (no learnable DSP) leads to degraded F1-score and AUC.  
- The **full 2TDformer** achieves the most balanced and highest performance across all metrics.

These results confirm the effectiveness of each DSP module and their synergistic impact in the complete architecture.

## 📚 Citation

If you use this repository, please cite the following paper:

```bibtex
@article{2tdformer2025,
  title={Weak Single Phase-to-Ground Fault Time Detection with Learnable-Parameter-Driven DSP-Enhanced Transformer},
  author={Luo, Huan and Li, Qiyue and Zhou, Hao and Sun, Wei and Li, Weitao and Liu, Zhi and Ji, Yusheng and Ding, Lijian},
  journal={IEEE Transactions on Smart Grid},
  year={2025}
}


@article{Li_2023_Incipient,
  title = {Incipient Fault Detection in Power Distribution System: A Time–Frequency Embedded Deep-Learning-Based Approach},
  shorttitle = {Incipient Fault Detection in Power Distribution System},
  author = {Li, Qiyue and Luo, Huan and Cheng, Hong and Deng, Yuxing and Sun, Wei and Li, Weitao and Liu, Zhi},
  year = {2023},
  journal = {IEEE Trans. Instrum. Meas.},
  volume = {72},
  pages = {1--14},
  issn = {1557-9662},
  urldate = {2024-11-11},
  eventtitle = {IEEE Transactions on Instrumentation and Measurement}
}
```

For questions or discussions, please contact:

- **Huan Luo**, Hefei University of Technology  
  Email: luohuan@mail.hfut.edu.cn  

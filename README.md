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


The overall architecture of 2TDformer is illustrated in the figure below:

![2TDformer Architecture](figures/model_architecture.png)




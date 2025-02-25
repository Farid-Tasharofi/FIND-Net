# FIND-Net: Fourier-Integrated Network with Dictionary Kernels for Metal Artifact Reduction
FIND-Net (Fourier-Integrated Network with Dictionary Kernels) is a deep learning model for Metal Artifact Reduction (MAR) in CT imaging. It integrates Fast Fourier Convolution (FFC) and trainable Gaussian filtering to suppress artifacts while preserving anatomical structures. 

<!-- 
# FIND-Net: Fourier-Integrated Network with Dictionary Kernels for Metal Artifact Reduction

![FIND-Net Logo](https://your-logo-url.com) Optional: Replace with a relevant image -->

## Overview
FIND-Net is a deep learning-based framework for **Metal Artifact Reduction (MAR)** in CT imaging. It extends the **DICDNet** architecture by incorporating **Fourier domain processing** and **trainable Gaussian filtering**, enhancing artifact suppression while preserving anatomical structures.

This repository provides:
- **Implementation of FIND-Net** for MAR.
- **Training and inference scripts** for testing the model.
- **Evaluation metrics and dataset handling**.

## Features
✅ **Hybrid Frequency-Spatial Processing**: Integrates **Fast Fourier Convolution (FFC)** for improved feature extraction.  
✅ **Trainable Gaussian Filtering**: Enhances frequency selectivity while preserving critical anatomical details.  
✅ **Efficient MAR Performance**: Reduces metal artifacts while maintaining high structural fidelity.  
✅ **Benchmark Comparisons**: Achieves state-of-the-art results against existing MAR methods.  

## Repository Structure

📂 FIND-Net/  
│── 📂 Model/              # Core model components (FIND-Net, ProxNet, FFC)  
│── 📂 Dataset/            # Dataset processing scripts  
│── 📂 utils/              # Helper functions (image saving, initialization)  
│── 📜 findnet.py          # Main FIND-Net model  
│── 📜 ProxNet.py          # Proximal network implementation  
│── 📜 ffc.py              # Fast Fourier Convolution with Gaussian filtering  
│── 📜 test_FINDNet.py     # Evaluation and testing script  
│── 📜 requirements.txt    # Required dependencies  
│── 📜 LICENSE             # License information  
│── 📜 README.md           # This file  



## Installation
To use FIND-Net, clone this repository and install the required dependencies:

```sh
git clone https://github.com/yourusername/FIND-Net.git
cd FIND-Net
pip install -r requirements.txt

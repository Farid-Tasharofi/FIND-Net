# FIND-Net: Fourier-Integrated Network with Dictionary Kernels for Metal Artifact Reduction
FIND-Net (Fourier-Integrated Network with Dictionary Kernels) is a deep learning model for Metal Artifact Reduction (MAR) in CT imaging. It integrates Fast Fourier Convolution (FFC) and trainable Gaussian filtering to suppress artifacts while preserving anatomical structures. 

## Abstract
Metal artifacts, caused by high-density metallic implants in computed tomography (CT) imaging, severely degrade image quality, complicating diagnosis and treatment planning. While existing deep learning algorithms have achieved notable success in Metal Artifact Reduction (MAR), they often struggle to suppress artifacts while preserving structural details. To address this challenge, we propose FIND-Net (Fourier-Integrated Network with Dictionary Kernels), a novel MAR framework that integrates frequency and spatial domain processing to achieve superior artifact suppression and structural preservation. 
FIND-Net incorporates Fast Fourier Convolution (FFC) layers and trainable Gaussian filtering, treating MAR as a hybrid task operating in both spatial and frequency domains. This approach enhances global contextual understanding and frequency selectivity, effectively reducing artifacts while maintaining anatomical structures. Experiments on synthetic datasets show that FIND-Net achieves statistically significant improvements over state-of-the-art MAR methods, with a 3.07\% MAE reduction, 0.18\% SSIM increase, and 0.90\% PSNR improvement, confirming robustness across varying artifact complexities. Furthermore, evaluations on real-world clinical CT scans confirm FIND-Net’s ability to minimize modifications to clean anatomical regions while effectively suppressing metal-induced distortions. These findings highlight FIND-Net’s potential for advancing MAR performance, offering superior structural preservation and improved clinical applicability.


## Overview


![FIND-Net Architecture](Figures/FIND-Net.png) <!-- Ensure the path is correct -->



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



## Installation
To use FIND-Net, clone this repository and install the required dependencies:

```sh
git clone https://github.com/yourusername/FIND-Net.git
cd FIND-Net
pip install -r requirements.txt
```


## Usage
### Testing the Model

To evaluate FIND-Net on a test dataset, use the provided `test.sh` script, which automates the setup and execution of the testing process.

#### **Step 1: Configure the Model Selection**
Before running the test script, ensure that the correct model configuration is set:

1. **Choose a model directory** (`MODEL_DIRECTORY` in `test.sh`):
   - **`FINDNet`** → Standard FIND-Net model  
   - **`FINDNet_no_GF`** → FIND-Net without Gaussian filtering  
   - **`DICDNet`** → Baseline DICDNet model  

2. **Modify `ProxNet.py` settings (located in the `Model` folder)**:
   - For **FINDNet** or **FINDNet_no_GF**:
     ```python
     FINDNet_Mnet = True
     FINDNet_Xnet = True
     ```
   - For **DICDNet**:
     ```python
     FINDNet_Mnet = False
     FINDNet_Xnet = False
     ```

3. **Modify `ffc.py` settings (located in the `Model` folder)**:
   - For **FINDNet** (with Gaussian filtering enabled):
     ```python
     Gaussian_filter = True
     ```
   - For **FINDNet_no_GF** (without Gaussian filtering):
     ```python
     Gaussian_filter = False
     ```

#### **Step 2: Execute the Test Script**
Once the configurations are correctly set, run the following command in the terminal:

```sh
bash test.sh
```

## Performance Comparison of MAR Approaches
| Methods         | Large Metal → Small Metal  | Average |
|---------------|---------------------------|---------|
| **LI**        | 35.5 / 0.866 / 35.9 | 29.6 / 0.899 / 38.1 | 23.4 / 0.929 / 39.9 | 26.6 / 0.913 / 38.9 |
| **DICDNet**   | 23.9 / 0.918 / 38.7 | 21.1 / 0.941 / 41.2 | 15.6 / 0.962 / 43.8 | 18.3 / 0.951 / 42.4 |
| **OSCNet**    | 23.2 / 0.920 / 39.0 | 21.9 / 0.938 / 40.4 | 16.2 / 0.959 / 42.9 | 18.8 / 0.948 / 41.7 |
| **OSCNet+**   | 23.5 / 0.918 / 39.0 | 21.9 / 0.938 / 40.8 | 16.2 / 0.959 / 43.2 | 18.8 / 0.948 / 42.0 |
| **FIND-Net-NoGF** | 24.0 / 0.917 / 38.7 | 21.0 / 0.941 / 41.3 | 15.5 / 0.962 / 43.8 | 18.2 / 0.951 / 42.5 |
| **FIND-Net**  | **22.9** / **0.925** / **39.2** | **20.9** / **0.942** / **41.4** | **15.2** / **0.963** / **44.3** | **17.9** / **0.952** / **42.8** |

**Table:** Performance comparison of different MAR approaches in terms of **MAE ↓ / SSIM ↑ / PSNR ↑** across varying metal sizes.

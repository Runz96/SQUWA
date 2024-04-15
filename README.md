# SQUWA (Signal Quality Weighted Fusion of Attentional Convolution and Recurrent Neural Network)

## Authors: [Runze Yan](https://scholar.google.com/citations?user=GnubTzsAAAAJ&hl=en) (ryan30@emory.edu), Cheng Ding, Ran Xiao, Alex Fedorov, Randall J Lee, Fadi Nahab, [Xiao Hu](https://www.nursing.emory.edu/faculty-staff/xiao-hu) (xiao.hu@emory.edu)

## SQUWA Paper: [CHIL 2024](https://chilconference.org/)


## Overview of SQUWA

We present a new DNN architecture, SQUWA, for AF detection using PPG data, which includes an innovative attention mechanism. Unlike traditional methods that discard low-quality signals, SQUWA dynamically weighs PPG segments based on their signal quality, directly incorporating this into the AF detection process. This mechanism prioritizes higher-quality segments during prediction and reduces the influence of noisier ones, optimizing the use of data in the overall analysis. Additionally, it processes data points individually rather than as a uniform sample, enhancing detection accuracy and effectiveness. The design principles of SQUWA could also be applied to other fields like human activity and speech recognition, addressing similar issues with noisy data.

<p align="center">
<img src="https://github.com/Runz96/SQUWA/blob/main/image/noisy_ppg.png">
</p>


## Installation and Setup

### 1: Download the Repo

First, clone the GitHub repository:

```
git clone https://github.com/Runz96/SQUWA
cd Raincoat
```


### 2: Set Up Environment

To install the core environment dependencies of Raincoat, use `environment.yml` in `config` folder:
```
conda env create -f environment.yml
```

### 3: Set up configurations
Confiure the path of training set and validation set in `train_adapt.yaml` in `config` folder, training set will not be shared for ethical reasons, except for one publicly accessible


### 4. Train a Model

Too train a model:
```
python train.py
```

### Citation
If you find *SQUWA* useful for your research, please consider citing this paper:

### Lisence
*SQUWA* codebase is under MIT license. For individual dataset usage, please refer to the dataset license found in the website.




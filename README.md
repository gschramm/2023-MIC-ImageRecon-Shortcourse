# 2023-MIC-ImageRecon-Shortcourse

## Slides 

The power point slides from my presentation at the IEEE MIC/NSS short course "Medical Image Reconstruction from Foundations to AI"
can be found in [slides/ML_recon_hands_on_georg_schramm.pptx](slides/ML_recon_hands_on_georg_schramm.pptx)

## Setup

**(1) Get the all the code** of this mini-tutorial by **(a) cloning** this repository
```
git clone https://github.com/gschramm/2023-MIC-ImageRecon-Shortcourse.git
```
**or (b) downloading a zip** of the code files using this [link](https://github.com/gschramm/2023-MIC-ImageRecon-Shortcourse/archive/refs/heads/main.zip)

**(2) Create a virtual conda environment** containg all packages we need.
All the packages we need are defined in `environment.yml` availalbe from **conda-forge**
```
conda env create -f environment.yml
```

**(3 optional) to run projections on CUDA GPUs (highly recommended)**
```
conda install -c conda-forge cupy
```



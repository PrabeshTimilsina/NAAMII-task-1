# Femur and Tibia Segmentation from CT Scans

This Python project performs automatic segmentation of **femur** and **tibia** bones from 3D CT scans in `.nii.gz` format. It uses classical image processing techniques like thresholding, denoising, and morphological cleaning to identify and separate bone regions from medical scans.

---

## 🧠 Overview

**What it does:**
- Loads a CT scan
- Denoises the image using a Gaussian filter
- Segments bone structures based on Hounsfield Unit (HU) thresholding
- Cleans binary masks by removing small noise and filling holes
- Separates femur and tibia using connected component analysis
- Saves output masks as new `.nii.gz` files

---

## 🧰 Tools & Libraries

- **SimpleITK** – Medical image reading and writing
- **NumPy** – Numerical array operations
- **SciPy (ndimage)** – Image filtering and labeling
- **scikit-image** – Morphological operations like noise removal
- **argparse** – Command-line arguments

---

##  Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/femur-tibia-segmentation.git
cd femur-tibia-segmentation
```
### 2. Install dependencies
pip install -r requirements.txt

### 3. Run
```bash
python bone_segmentation.py --input path/to/ct_scan.nii.gz
```
The output mask will be saved in ./output folder

For specific expansion metrics like 2 mm, 4 mm

```bash
python expanded_mask.py --masks path/to/masks and --expansion value of expansion
```

For random expansion within original and 2mm expansion
```bash
python random_expansion_contour.py --max_expansion max expansion value and --random_expansion random expansion value
```

For landmard detetion, the output will be saved in csv file
```bash
python landmark_detection.py
```

## Use Cases
Medical image preprocessing

Dataset creation for training deep learning models

Bone structure analysis and measurement

Augmenting surgical planning tools

## Author
Prabesh Timilsina
Medical Image Processing | Computer Engineering Graduate
📅 May 2025


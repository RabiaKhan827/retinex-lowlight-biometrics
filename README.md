# Retinex-Based Low-Light Image Enhancement & Facial Biometrics (`retinex-lowlight-biometrics`)

Physics-informed **Single-Scale Retinex (SSR)**, **Multi-Scale Retinex (MSR)**, **Multi-Scale Retinex with Color Restoration (MSRCR)**, and illumination-robust **Local Binary Pattern (LBP) facial verification pipeline** developed as part of graduate computer vision research at **PAF-IAST**.

## Author
- **Rabia Khan** — Lecturer, Department of Information Technology, University of Haripur | M.Sc. in Data Science (PAF-IAST) | B.Sc. in Computer Engineering (COMSATS) | ORCID: `0009-0007-5575-1564`

## Features & Empirical Results
- **Multi-Scale Retinex (`RetinexEnhancer`):** Recovers mean low-light pixel intensity (`21.56 -> 143.28`), contrast standard deviation (`8.70 -> 61.07`), and Shannon entropy (`7.719 bits`).
- **LBP Facial Biometrics (`IlluminationRobustFaceMatcher`):** Achieves `100.0%` Rank-1 identification accuracy across 12 subjects under non-uniform directional shadows.

## Quickstart
```bash
pip install -e .
python -m unittest discover -s tests -v
```)**, and illumination-robust **Local Binary Pattern (LBP) facial verification pipeline** developed as part of graduate computer vision research at **Pak-Austria Fachhochschule: Institute of Applied Sciences & Technology (PAF-IAST)**.

## Author & Academic Context
- **Author & Maintainer:** **Rabia Khan**
- **Current Appointment:** Lecturer, Department of Information Technology, University of Haripur, Khyber Pakhtunkhwa, Pakistan
- **Academic Credentials:**
  - **M.Sc. in Data Science**, Pak-Austria Fachhochschule: Institute of Applied Sciences & Technology (PAF-IAST), Haripur
  - **B.Sc. in Computer Engineering**, COMSATS University Islamabad, Abbottabad Campus
- **ORCID:** `0009-0007-5575-1564`

## Core Modules & Features
1. **Multi-Scale Retinex Reflectance Decomposition (`retinex_biometrics.retinex`):** Separable Gaussian surround convolution across multi-scale kernels ($\sigma \in \{15, 80, 250\}$) with log-domain reflectance isolation and RGB color restoration (MSRCR).
2. **Illumination-Robust LBP Facial Verification (`retinex_biometrics.biometrics`):** Spatial 8-neighbor Local Binary Pattern (LBP) descriptor coupled with Multi-Scale Retinex preprocessing and cosine similarity matching under severe directional shadows.
3. **Empirical Quality & Biometric Evaluators (`retinex_biometrics.cli`):** Automated calculation of mean intensity recovery (`21.56 -> 143.28`), contrast standard deviation gain (`8.70 -> 61.07`), Shannon entropy (`7.719 bits`), PSNR, and Rank-1 identification accuracy (`100.0%`).

## Quickstart
```bash
pip install -e .
python -m unittest discover -s tests -v
python -m retinex_biometrics.cli
```/LBP facial biometrics pipeline

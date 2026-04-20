BPS on MAAP — Jupyter Notebooks
This repository contains the Jupyter notebooks to install and run the BPS Processor Suite (BPS) on the ESA MAAP Coding/Experiment environment.

Repository Contents
- BPS_installation.ipynb: BPS installation procedure
- BPS_Run.ipynb: BPS processing chain (L1F → L1 → STA → L2A)

Prerequisites
1. Set up the SCRIPTS folder
Place the SCRIPTS folder under /home/jovyan/:
/home/jovyan/SCRIPTS/
├── JOBuilder.py
├── biofetch.yml
└── CONFIGURATION_FILE/  ← unzip the configuration file folder here
    ├── AUX_441/         
    │   ├── BIO_AUX_INS_*
    │   ├── BIO_AUX_PP1_*
    │   ├── BIO_AUX_PPS_*
    │   ├── BIO_AUX_PP2_2A_*
    │   ├── BIO_AUX_PP2_AB_*
    │   ├── BIO_AUX_PP2_FD_*
    │   └── BIO_AUX_PP2_FH_*
    └── AUX_USER/
└── ....

⚠️ Make sure to unzip the CONFIGURATION_FILE folder and place the AUX files (INS, PP1, PPS, PP2_2A, PP2_AB, PP2_FD, PP2_FH) inside the correct AUX folder before running any notebook.

3. Place the BPS tarballs
Place the BPS tarballs in the corresponding version folder under /home/jovyan/SW/:
/home/jovyan/SW/BPS_V441/ (example)
├── bps-bundle-v4.4.1.tar.gz
└── btk-vt-4.4.1.tar.gz

Usage
Step 1 — Install BPS
Open and run BPS_installation.ipynb. Set BPS_VERSION at the top to the version you want to install and run all cells in order.
Step 2 — Run BPS
Open and run BPS_Run.ipynb. Place your input products in INPUT_FOLDER/Inputs/ and set INPUT_FOLDER and PROCESSOR_VERSION at the top, then run all cells in order.

Storage Notes
Files stored under /home/jovyan/ persist across sessions on MAAP. Everything outside this directory may be permanently deleted after 7 days of inactivity. It is recommended to also keep a copy of the tarballs in my-private-bucket for long-term backup.

Processing Chain
RAW0S + RAW0M + AUX_ORB + AUX_ATT + AUX_TEC
        │
        ▼
   L1F chain  →  SLC framed products
        │
        ▼
   L1 chain   →  L1 products
        │
        ▼
   STA chain  →  Stack products
        │
        ▼
   L2A chain  →  L2A products

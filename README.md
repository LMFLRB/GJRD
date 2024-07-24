# Generalized Jensen-Renyi's Divergence (GJRD)

## Introduction

The generalized Jensen-Renyi's divergence (GJRD), serving as a robust tool for estimating the divergence between multiple distributions, is introduced to handle machine learning problems involving data from multiple distributions or sources. This work specifies GJRD-based deep clustering as a case study.

## Code utilization

1. **Install Required Packages:**

   First, install the necessary packages as specified in `requirement.txt`:

   ```sh
   pip install -r requirement.txt
2. **Train the Model:**
   You can train the required model by running `run_dataset_loss.py`.
   Users can modify the code to decide which `datasets` or `network` configurations to run.
   Some basic settings are written in the `configs.yaml` file:
   ```sh
   python run_dataset_loss.py
3.  **python run_dataset_loss.py**
   Users can also directly obtain the results of our stored model by running `eval_pretrained_model.py`:
   ```sh
   python eval_pretrained_model.py

Feel free to modify the configurations and experiment with different settings to fit your needs.

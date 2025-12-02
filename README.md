# Whole-Brain Model Fitting to EEG Power Spectral Density (PSD)

## 🧠 Project Overview

This project implements a whole-brain dynamic modeling approach, specifically using the **Hopf Bifurcation Model**, to simulate and fit the empirical **Power Spectral Density (PSD)** derived from Electroencephalography (EEG) data.

The core goal is to estimate a set of neurophysiologically-informed parameters (including local dynamics, global coupling, and heterogeneity driven by neurotransmitter receptor maps) that allow the model to best reproduce the observed EEG PSD across different experimental conditions. The model fitting employs a sophisticated optimization strategy utilizing **PyTorch**, featuring separate learning rates for model parameters and hyperparameters, as well as a selection of loss functions and learning rate schedulers. 

---

## 📁 Repository Structure

The repository is organized to clearly separate input data, neuromaps, and the main model code.

### 1. `struct_data`

Contains structural data essential for the model:
* **`leadfield.npy`**: The initial **Leadfield Matrix (LFM)** that maps source-space brain activity to scalp-level EEG channels.
* **`wll.npy`**: Initial values for the **Connection Gain Matrix** (or "effective connectivity" weights).

### 2. `conn_data`

Holds parcellation and structural connectivity data:
* **`atlas_data.csv`**: Contains ROI labels and MNI coordinates for the parcellation scheme.
* **`Schaefer2018_200Parcels_7Networks_count.csv`**: The **Structural Connectivity (SC)** matrix defining physical connections between regions (based on the Schaefer 2018 parcellation).

### 3. `data`

Contains the empirical EEG data used for model fitting:
* **`sub27sess02_{key}.fif`**: Example raw EEG data files (in `.fif` format) for various experimental conditions (e.g., `FullSession`, `PlanDay`, `Meditation`, `MentalMath`).

### 4. `neuromaps`

Stores a collection of neurotransmitter receptor and transporter density maps used to introduce **regional heterogeneity** into the model's parameters:
* Files like `T1T2_parc.npy`, `H3_parc.npy`, `D1_parc.npy`, `GABAa_parc.npy`, etc., are loaded and normalized to create the heterogeneity maps (h_maps).

### 5. `Results`

This directory is created during execution and stores the final output:
* **`sub27sess02_{key}_{loss_method}_{sched_type}_fitting_results.pkl`**: Pickled Python objects containing the `Model_fitting` class instance, which includes the trained model parameters and full training statistics.

---

## 💻 Code

The entire project logic is contained within a single file:

* **`psd_main_neuromaps.py`** 
    * Defines the core PyTorch modules: `par`, `ParamsHP`, `AbstractNMM`, `COVHOPF`, `AbstractLoss`, `CostsTS`, `CostsHP`, and the training pipeline `Model_fitting`.
    * Handles data loading, model initialization, optimization, and results saving.

---

## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/veronean/Meditation_EEG_code.git](https://github.com/veronean/Meditation_EEG_code.git)
    cd Meditation_EEG_code
    ```

2.  **Set up a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    # venv\Scripts\activate   # Windows
    ```

3.  **Install dependencies:**

    Create a `requirements.txt` file:
    ```bash
    numpy
    scipy
    pandas
    torch
    scikit-learn
    mne
    pathlib
    ```

    Then run:
    ```bash
    pip install -r requirements.txt
    ```

---

## 🚀 Usage

The fitting process is executed through the `main` function, which loops over different experimental conditions and parameter initializations.

1.  **Ensure Data is Present:** Verify that all data files are correctly placed in the corresponding directories (`struct_data`, `conn_data`, `data`, and `neuromaps`).

2.  **Run the script:**
    ```bash
    python psd_main_neuromaps.py
    ```

3.  **Monitor Output:** The terminal will display the loss, pseudo FC correlation, cosine similarity, and R2 metrics for each epoch.

4.  **Check Results:** The optimized models and training statistics will be saved as `.pkl` files in the **`Results`** directory.

---

## ⚙️ Model and Fitting Details

### **COVHOPF (Whole Brain Model)**

The `COVHOPF` class implements the **Hopf Bifurcation Model** which simulates the cross-spectrum of neural activity. Key fitted parameters include:
* **Local Dynamics ($\mathbf{a}$)**: Bifurcation parameter that is heterogeneously modulated by normalized neuromaps.
* **Intrinsic Frequency ($\mathbf{\omega}$)**: The natural oscillation frequency of local nodes.
* **Global Coupling ($\mathbf{g}$)**: Scales the influence of structural connectivity.
* **Effective Connectivity ($\mathbf{wll}$)**: Connection gain matrix used to scale the Structural Connectivity (SC) (if `fit_gains=True`).

### **Cost Function (CostsHP)**

The `CostsHP` class implements a composite cost function:

```math
$$\text{Total Loss} = w_{\text{cost}} \cdot \text{Loss}_{\text{main}} + \text{Loss}_{\text{prior}} + \text{Reg}_{\text{term}}$$
```

* **Main Loss ($\text{Loss}_{\text{main}}$)**: Calculated between the $\text{log}_{10}$ of the simulated and empirical PSDs (in dB-space) using methods like `'mse'`, `'log_fro'`, or `'pearson'`.
* **Prior Loss ($\text{Loss}_{\text{prior}}$)**: A Bayesian regularization term that anchors parameters near their defined prior means ($\mu$) with respect to their prior variance ($\sigma^2$).
* **Regularization Term ($\text{Reg}_{\text{term}}$)**: Includes L2 regularization for non-prior-constrained parameters and for the weights of the heterogeneity maps.

### **Optimization (Model\_fitting)**

The training pipeline uses:
* **Two Optimizers**: Separate **Adam** optimizers are used for:
    1.  **Model Parameters**: Local parameters and connection matrices.
    2.  **Hyperparameters**: The prior means and variances ($\mu, \sigma^2$) when `fit_hyper=True`.
* **Learning Rate Scheduling**: Options include `'OneCycleLR'` or `'ReduceLROnPlateau'` to manage the learning process.

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1.  Fork the repository.
2.  Create a descriptive feature branch (e.g., `git checkout -b feature/new-loss-method`).
3.  Commit your changes.
4.  Submit a Pull Request with a clear explanation of your modifications.

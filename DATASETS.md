
# MedVerse AI — Dataset and Model Documentation

This document describes the data organization and model-artifact requirements for the MedVerse AI project.

The repository contains application source code and AI/ML research code. Dataset files, large medical image collections, generated feature arrays, and trained model checkpoints are maintained separately from GitHub.

This separation helps keep the repository manageable and avoids distributing data without confirming its licensing and usage permissions.

---

## 1. Dataset Organization

The local project uses a `data/` directory for datasets, prepared data, manifests, and sample inputs.

The following directories have been used in the project:

```text
data/
│
├── chest_xray/
│
├── chexpert_final/
│
├── chexpert_prepared/
│
├── clinical_text/
│
├── manifests/
│
├── sample_reports/
│
└── sample_xrays/
```

These directories are maintained locally and are not distributed as part of the GitHub repository.

The exact files available in each directory depend on the local dataset setup.

---

## 2. Data Categories

### 2.1 Chest X-ray Data

The project includes chest X-ray data for medical image classification experiments.

The image-processing experiments explore classification using convolutional neural networks, including DenseNet121.

The local data organization includes:

- `chest_xray/` — chest X-ray dataset files.
- `chexpert_final/` — locally maintained CheXpert-related data.
- `chexpert_prepared/` — prepared data used for experimentation.

The datasets are not included in this repository.

Before running image-training or evaluation scripts, configure the expected dataset paths and verify the directory structure required by the corresponding script.

### 2.2 Clinical Text Data

The project includes clinical text data used for clinical information processing and multimodal experiments.

The local `clinical_text/` directory has been used for files such as:

- `clinical_text_dataset.csv`
- `multimodal_dataset.csv`
- `fusion_experiment_dataset.csv`

Additional intermediate or processed files may exist locally.

These files are not included in the GitHub repository.

The exact columns, labels, and preprocessing requirements should be checked against the corresponding experiment scripts and dataset files.

### 2.3 Sample Medical Reports

The `sample_reports/` directory is intended for sample report inputs used during OCR and report-processing development.

Sample documents should be anonymized and must not contain personally identifiable patient information.

### 2.4 Sample X-ray Images

The `sample_xrays/` directory is intended for sample medical image inputs used during development and testing.

Only use images that are authorized for the intended research or testing purpose.

### 2.5 Dataset Manifests

The `manifests/` directory contains dataset organization or preparation metadata used by the project.

Manifests may describe file paths, labels, or dataset splits. Their actual schema depends on the relevant experiment.

---

## 3. Model Artifacts

Trained model weights and generated artifacts are maintained locally.

Examples of artifact formats used by the project include:

| Extension | Typical purpose |
|---|---|
| `.pth` | PyTorch model checkpoints |
| `.pt` | PyTorch model files |
| `.h5` | Keras / TensorFlow model files |
| `.keras` | Keras model files |
| `.onnx` | ONNX model exports |
| `.joblib` | Serialized machine-learning models or preprocessing objects |
| `.npz` | NumPy arrays and extracted features |

Not every format is necessarily required by every component.

Model files must be placed in the locations expected by the corresponding inference or experiment scripts.

Do not assume that a model is available merely because its source code is present.

---

## 4. Dataset Setup

### Step 1: Obtain the datasets

Obtain the required datasets from their respective authorized sources.

Follow each dataset's license, terms of use, research agreement, attribution requirements, and redistribution restrictions.

Do not download or redistribute restricted datasets without the required permissions.

### Step 2: Prepare the local directory structure

Create the required data directories under the project root.

Example:

```text
MedVerse-AI/
└── data/
    ├── chest_xray/
    ├── chexpert_final/
    ├── chexpert_prepared/
    ├── clinical_text/
    ├── manifests/
    ├── sample_reports/
    └── sample_xrays/
```

### Step 3: Verify the dataset paths

Check the relevant experiment configuration and scripts to identify the exact filenames and directory paths expected by each workflow.

The directory names shown in this document describe the project's local organization; they do not guarantee that every experiment can run without additional preprocessing.

### Step 4: Prepare the data

Run the preprocessing steps required by the relevant experiment.

Depending on the workflow, this may include:

- Image resizing and normalization.
- Dataset splitting.
- Label preparation.
- Clinical text cleaning.
- Feature extraction.
- Manifest generation.
- Validation of file paths and labels.

Use the preprocessing configuration associated with the experiment to avoid inconsistencies between training and evaluation.

### Step 5: Configure model artifacts

Place trained checkpoints and supporting preprocessing artifacts in the expected locations.

Verify that the model architecture, input dimensions, class labels, and preprocessing configuration match the saved checkpoint.

---

## 5. Data and Model Files Excluded from Git

The repository's `.gitignore` excludes large or local-only files, including:

- Local databases.
- Uploaded files.
- Dataset files.
- Large medical image directories.
- Model checkpoints.
- Generated NumPy feature arrays.
- Compressed archives.
- Local environment files.

This means cloning the repository does not automatically download the datasets or trained model artifacts.

The GitHub repository contains the source code and documentation needed to understand the project, while the corresponding research data must be configured separately.

---

## 6. Data Privacy and Responsible Use

Medical data can contain sensitive personal information.

When working with medical datasets:

1. Use data only for purposes permitted by the dataset's license or research agreement.
2. Do not upload identifiable patient records to GitHub.
3. Do not commit local databases containing patient information.
4. Remove personally identifiable information from sample reports.
5. Do not expose API keys, credentials, or access tokens in source code.
6. Follow institutional and dataset-specific research requirements.

---

## 7. Reproducibility Notes

Reproducing an experiment requires more than the source code.

For reliable reproduction, record:

- Dataset name and version.
- Dataset source and access date.
- Preprocessing configuration.
- Training and validation split.
- Random seed.
- Model architecture.
- Hyperparameters.
- Software and dependency versions.
- Evaluation metrics.
- Model checkpoint used for evaluation.

Where applicable, retain experiment configuration files and evaluation outputs alongside the corresponding research code.

Dataset access and model availability may differ between development environments.

---

## 8. Important Disclaimer

The datasets and models used in MedVerse AI are intended for academic and research experimentation.

Model outputs are not clinically validated diagnoses and must not be used as the sole basis for medical decisions.

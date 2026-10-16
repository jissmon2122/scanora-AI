# Scanora AI — Model Card: Deep Neural Network for Mammography Analysis

## Model Details
- **Developer:** Scanora Research Team
- **Model Version:** 1.0.0-research
- **Release Date:** October 2026
- **Model Type:** Multi-Task Deep Convolutional Neural Network (EfficientNet-B2 backbone adapted for 1-channel medical grayscale) + Faster R-CNN Lesion Localization Head.
- **Framework:** PyTorch 2.4+
- **License:** Open Academic & Research Prototype License

## Intended Use
- **Primary Use Case:** Computer-Aided Detection (CADe/CADx) research and second-opinion decision support for radiologists reviewing screening mammograms.
- **Out of Scope & Prohibited Uses:**
  - Standalone primary diagnostic determination without qualified radiologist oversight.
  - Triage decision without clinician review.
  - Emergency room acute trauma imaging.

## Training Data & Provenance
- **Primary Benchmark:** VinDr-Mammo (PhysioNet Credentialed Access, 5,000 examinations, 20,000 FFDM images).
- **Secondary Evaluation Benchmark:** CBIS-DDSM (The Cancer Imaging Archive, 2,620 scanned film studies).
- **Data Splitting Protocol:** Strict patient-level partitioning (70% Train, 15% Validation, 15% Test). Zero patient overlap between splits to prevent intra-patient cross-projection leakage.

## Performance & Evaluation Protocol
- **Evaluation Criteria:**
  - Receiver Operating Characteristic Area Under the Curve (ROC-AUC).
  - Precision-Recall Area Under the Curve (PR-AUC).
  - Sensitivity at 90% Specificity.
  - Expected Calibration Error (ECE) via Temperature Scaling.
- **Calibration:** Uncalibrated deep neural networks exhibit overconfidence on out-of-distribution cases. Scanora incorporates post-hoc Platt / Temperature scaling on a held-out validation set.

## Ethical Considerations & Fairness
- **Demographic Imbalance:** Datasets predominantly reflect specific demographic populations (VinDr-Mammo: Vietnamese; DDSM: North American). Performance across varying racial and demographic groups may differ due to variations in baseline breast density.
- **Dense Breast Limitations:** Mammographic sensitivity is known to be lower in extremely dense breast tissue (ACR Density D) due to tissue masking effects. Scanora displays a breast density flag whenever dense tissue is identified.

## Clinical Safety Disclaimer
**NOT FDA CLEARED OR CE MARKED.** Scanora is an investigational software prototype designed solely for research, validation, and educational purposes. It does not replace independent histopathological or clinical radiological evaluation. Never promise 100% diagnostic accuracy.

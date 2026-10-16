# Scanora AI — Model Card: Deep Neural Network for Mammography Analysis

## Model Details
- **Developer:** Scanora Research & Development Team
- **Model Version:** 1.0.0-research
- **Release Date:** October 2026
- **Architecture:** 
  - **Classifier:** Custom Dual-Head EfficientNet-B2 (modified `initial_conv` with single-channel 1×512×512 input, global average pooling, dropout, malignancy prediction head outputting scalar logit, breast density head outputting 4-class logits [A, B, C, D], post-hoc Temperature Scaler for probability calibration).
  - **Detector:** Faster R-CNN with ResNet-50-FPN backbone adapted for mammographic lesion bounding box proposal and classification (Mass, Suspicious Calcification, Architectural Distortion, Asymmetry).
- **Framework:** PyTorch 2.4+ / Torchvision
- **License:** Open Academic & Research Prototype License

---

## Model Lifecycle & Verification States

Scanora enforces a strict, four-tier lifecycle state machine managed by `CheckpointValidator`:

| Lifecycle State | Description | Permitted Operation |
|---|---|---|
| `MODEL_UNAVAILABLE` | Checkpoint missing, corrupt, incomplete, or contains only pre-trained ImageNet backbone weights without task fine-tuning. | **Clinical inference strictly blocked.** API returns HTTP 503 (`SERVICE_UNAVAILABLE`). No clinical risk percentages may be emitted. |
| `SIMULATED_DEMO` | Explicit demo mode enabled with clear amber visual cues. | Demonstration and workflow testing only. Returns `simulation-fallback` identifier. Explicitly labeled as simulated findings; prohibited from patient decision-making. |
| `RESEARCH_UNVALIDATED` | Checkpoint physically verified, task-trained on mammography data, and structurally intact. | Investigational and academic research use only. Radiologist decision-support research. Not approved for autonomous diagnostic triage. |
| `VALIDATED_FOR_INTENDED_USE` | Multi-reader multi-case (MRMC) clinical validation completed, independent clinical trial evidence recorded, and formal regulatory clearance (e.g. FDA 510(k), CE mark) verified with token. | Permitted for defined clinical intended-use setting with credentialed clinician oversight. **Training or internal evaluation alone can never grant this state.** |

> [!IMPORTANT]
> **Strict Safety Gate:** If any checkpoint claims `VALIDATED_FOR_INTENDED_USE` in its metadata without cryptographically verified `regulatory_clearance_id` and `independent_reader_study` records, `CheckpointValidator` automatically downgrades the state to `RESEARCH_UNVALIDATED` and logs an audit security warning.

---

## Current Physical Deployment Status

- **Status on Disk:** Currently in `MODEL_UNAVAILABLE` state unless a verified trained checkpoint file is mounted to `ai_service/checkpoints/mammography_classifier.pt`.
- **Pre-training Separation:** General computer vision weights (e.g. ImageNet) are strictly isolated from mammography task-specific fine-tuning. Forwarding images through random initialized linear heads to generate diagnostic scores is strictly prevented by code-level runtime assertions and API-level HTTP 503 error responses.

---

## Intended Use & Clinical Scope

### Intended Use
- **Primary Use Case:** Computer-Aided Detection (CADe) and Computer-Aided Diagnosis (CADx) research for radiologists and oncology researchers reviewing digital screening mammography (FFDM).
- **Secondary Opinion:** Investigational second-read assistant to flag potential regions of interest and compute calibrated risk probabilities alongside radiologist interpretation.

### Out of Scope & Prohibited Uses
- **Autonomous primary diagnosis:** Never use as a replacement for certified radiologist review or tissue biopsy.
- **Standalone triage:** Automated dismissal of patients without human clinical review is strictly prohibited.
- **Trauma / Emergency imaging:** Model is trained exclusively on breast screening mammograms and is not validated for general thoracic or acute trauma imaging.
- **100% Accuracy Claims:** Never claim or imply zero error or 100% diagnostic accuracy.

---

## Training Data & Provenance Requirements

For legitimate supervised model training, Scanora requires credentialed, ethically approved datasets:

1. **VinDr-Mammo (Full-Field Digital Mammography):**
   - Source: PhysioNet Credentialed Data Access Agreement (Hanoi Medical University Hospital & Hospital 108).
   - Sample Size: 5,000 examinations (20,000 DICOM / PNG images).
   - Annotations: Breast-level BI-RADS (1-5), ACR density (A-D), and lesion bounding boxes confirmed by expert radiologists.
2. **CBIS-DDSM (Curated Breast Imaging Subset of DDSM):**
   - Source: The Cancer Imaging Archive (TCIA).
   - Sample Size: 2,620 scanned film studies with biopsy-verified pathology labels (benign vs. malignant).

### Partitioning & Leakage Prevention Protocol
- **Strict Patient-Level Splitting:** 70% Train, 15% Validation, 15% Independent Test.
- **Zero Patient Overlap:** All four standard screening projections (L-CC, L-MLO, R-CC, R-MLO) for any individual patient are strictly quarantined to a single split to eliminate intra-patient data leakage.
- **Deterministic Seeding:** Fixed seed (`seed=42`) across PyTorch, NumPy, and CUDA to ensure 100% reproducible data splits.

---

## Independent Evaluation Protocol

Model performance must be verified on a held-out patient-level test set using multi-dimensional clinical metrics:

1. **Discrimination Metrics:**
   - **ROC-AUC (Receiver Operating Characteristic Area Under Curve):** Overall discriminatory power across all operating thresholds.
   - **95% Percentile Bootstrap Confidence Interval:** Evaluated via 200 stratified bootstrap resamples to report uncertainty bounds `[CI_lower, CI_upper]`.
   - **PR-AUC (Precision-Recall Area Under Curve):** Essential metric for imbalanced cancer prevalence (~2-5% in screening populations).
   - **Sensitivity & Specificity:** Operating point metrics at clinical decision threshold (default: 0.50).
   - **Sensitivity at 90% Specificity:** High-specificity operating point for screening triage.
   - **False Negative Rate (FNR):** Critical patient-safety metric quantifying missed malignancies ($FNR = 1 - \text{Sensitivity} = FN / (TP + FN)$).
   - **Positive Predictive Value (PPV) & Negative Predictive Value (NPV).**

2. **Calibration & Uncertainty:**
   - **Brier Score:** Mean squared difference between predicted probabilities and binary outcomes.
   - **Expected Calibration Error (ECE):** 10-bin reliability diagram calibration error.
   - **Prediction Entropy:** Binary entropy metric flagging ambiguous or indeterminate cases ($H(p) > 0.55$).

3. **Subgroup Fairness & Robustness Analysis:**
   - **By View:** Separate performance evaluation on Cranio-Caudal (CC) vs. Medio-Lateral Oblique (MLO) projections.
   - **By Breast Density:** Separate evaluation across ACR density categories:
     - `Density_A`: Almost entirely fatty.
     - `Density_B`: Scattered areas of fibroglandular density.
     - `Density_C`: Heterogeneously dense.
     - `Density_D`: Extremely dense (high masking risk).

4. **Patient-Level Aggregation:**
   - Maximum-pooled probability across all views of a patient's study to measure true clinical patient-level discrimination.

---

## Target Clinical Benchmarks vs. Empirical Results

| Metric | Target Research Benchmark | Status |
|---|---|---|
| Image-Level ROC-AUC | $\ge 0.8500$ | Unvalidated target (requires physical test set evaluation) |
| Patient-Level ROC-AUC | $\ge 0.8800$ | Unvalidated target |
| Expected Calibration Error (ECE) | $\le 0.0500$ | Unvalidated target |
| Sensitivity at 90% Specificity | $\ge 0.7000$ | Unvalidated target |
| False Negative Rate (FNR) | $\le 0.1200$ | Unvalidated target |

---

## Ethical Considerations & Limitations

- **Demographic Bias:** VinDr-Mammo represents Vietnamese patients; CBIS-DDSM represents North American patients. Baseline breast density and cancer subtype distributions vary geographically; multi-center external validation is mandatory before broader deployment.
- **Tissue Masking Effect in Dense Breasts:** High density (ACR C/D) can obscure non-calcified masses on mammography. The system automatically surfaces density warning flags when dense tissue is detected.
- **Orthogonal View Discordance:** In cases where CC and MLO views yield conflicting risk estimates, the multi-view aggregator flags the case as indeterminate and recommends manual radiological review.

---

## Regulatory Compliance & Safety Notice

**NOT FDA APPROVED OR CE MARKED AS A MEDICAL DEVICE.**  
Scanora is an investigational software prototype designed exclusively for scientific validation, algorithmic research, and educational demonstration. It must never be used as a standalone or primary diagnostic device.

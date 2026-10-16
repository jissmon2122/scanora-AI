# Mammography Dataset Acquisition & Reproducibility Guide

This document provides transparent, reproducible instructions for acquiring, verifying, and licensing research mammography datasets for Scanora.

---

## 1. VinDr-Mammo (Full-Field Digital Mammography)

### Overview
- **Source:** PhysioNet (`https://physionet.org/content/vindr-mammo/1.0.0/`)
- **Type:** Full-Field Digital Mammography (FFDM), 20,000 images from 5,000 examinations.
- **Annotations:** Breast-level BI-RADS (1–5), Breast Density (A–D), and radiologist lesion bounding boxes.
- **License:** PhysioNet Credentialed Data Use Agreement (v1.5.0).

### Access Steps
1. Create an account on [PhysioNet](https://physionet.org/).
2. Complete the required **CITI Program "Data or Specimens Only Research"** human research training.
3. Submit a credentialed access request on the VinDr-Mammo project page.
4. Sign the Data Use Agreement pledging not to attempt re-identification.
5. Once approved, download via the PhysioNet AWS CLI command or direct transfer:

```bash
# Example AWS CLI download once credentials are bound
aws s3 sync --no-sign-request s3://physionet-open/vindr-mammo/1.0.0/ ai_service/data/vindr_mammo/
```

### Directory Structure Expected by Scanora
```
ai_service/data/vindr_mammo/
├── images/
│   ├── [study_id]/
│   │   ├── [image_id].dicom
├── breast-level_annotations.csv
└── finding_annotations.csv
```

---

## 2. CBIS-DDSM (Curated Breast Imaging Subset of DDSM)

### Overview
- **Source:** The Cancer Imaging Archive (TCIA) (`https://www.cancerimagingarchive.net/collection/cbis-ddsm/`)
- **Type:** Scanned film mammography, 2,620 studies.
- **Annotations:** Pathology-confirmed ground truth (`MALIGNANT`, `BENIGN`, `BENIGN_WITHOUT_CALLBACK`), ROI lesion masks.
- **License:** Creative Commons Attribution 3.0 Unported (CC BY 3.0).

### Access Steps
1. Visit the [TCIA CBIS-DDSM Collection Page](https://wiki.cancerimagingarchive.net/display/Public/CBIS-DDSM).
2. Download the `.tcia` manifest files using the NBIA Data Retriever:
   - `mass_case_description_train_set.csv`
   - `mass_case_description_test_set.csv`
   - `calc_case_description_train_set.csv`
   - `calc_case_description_test_set.csv`
3. Download images via the NBIA Data Retriever CLI:

```bash
# Run NBIA Data Retriever with the manifest file
./nbia-data-retriever -d manifest.tcia -o ai_service/data/cbis_ddsm/
```

---

## 3. Dataset Governance & Ethics Rules

1. **Patient De-Identification:** Never commit raw patient DICOM files containing un-anonymized metadata tags (PatientName, PatientID, PatientBirthDate).
2. **Patient-Level Isolation:** Models trained in Scanora strictly enforce patient-level splitting to prevent training/test leakage across views (e.g. MLO and CC views of the same patient MUST remain in the same split).
3. **Domain Shift Awareness:** Digitized analog film mammograms (CBIS-DDSM) exhibit significantly different noise characteristics and resolution profiles than modern full-field digital mammography (VinDr-Mammo). Models evaluated on one domain must not claim generalizability to the other without empirical domain shift evaluation.

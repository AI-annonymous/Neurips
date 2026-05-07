"""
===============================================================================
Extract W_pre (upstream causal candidate variables) from MIMIC-IV
and link to MIMIC-CXR studies.
===============================================================================

Purpose:
    For the Graph-Constrained Two-Stage Search method, W_pre = U = (U1, ..., Up)
    are candidate upstream causes that sit on the path  U -> Y  in the ADMG.
    Stage 1 searches over subsets V_Y ⊆ U to minimize I(A; Y | V_Y) + λ|V_Y|.

    This script builds the full candidate matrix W_pre by:
      1. Linking MIMIC-CXR studies to MIMIC-IV admissions via subject_id
      2. Enforcing temporal precedence (diagnosis BEFORE the CXR)
      3. Mapping ICD-9/10 codes to clinically meaningful binary indicators
      4. Optionally adding lab-derived and procedure-derived variables
      5. Outputting a (study-level) binary matrix ready for Stage 1 search

Usage:
    Adjust PATHS at the top, then run:
        python extract_wpre_mimic.py

Output:
    wpre_matrix.csv — one row per (subject_id, study_id), columns are binary W_pre candidates
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# =============================================================================
# 0. PATHS — adjust these to your local setup
# =============================================================================
MIMIC_CXR_DIR = Path("./mimic-cxr-jpg/2.0.0")
MIMIC_IV_DIR  = Path("/shared/Data/MIMICIV/physionet.org/files/mimiciv/3.1")

# MIMIC-CXR files
CXR_LABELS_FILE  = MIMIC_CXR_DIR / "mimic-cxr-2.0.0-chexpert.csv"
CXR_META_FILE    = MIMIC_CXR_DIR / "mimic-cxr-2.0.0-metadata.csv"
CXR_SPLITS_FILE  = MIMIC_CXR_DIR / "mimic-cxr-2.0.0-split.csv"  # optional

# MIMIC-IV files
ADMISSIONS_FILE  = MIMIC_IV_DIR / "hosp" / "admissions.csv.gz"
DIAGNOSES_FILE   = MIMIC_IV_DIR / "hosp" / "diagnoses_icd.csv.gz"
PROCEDURES_FILE  = MIMIC_IV_DIR / "hosp" / "procedures_icd.csv.gz"
PATIENTS_FILE    = MIMIC_IV_DIR / "hosp" / "patients.csv.gz"
LABEVENTS_FILE   = MIMIC_IV_DIR / "hosp" / "labevents.csv.gz"
D_ICD_DIAG_FILE  = MIMIC_IV_DIR / "hosp" / "d_icd_diagnoses.csv.gz"  # ICD lookup

# Output
OUTPUT_FILE = Path("./wpre_matrix.csv")

# =============================================================================
# 1. TEMPORAL LOOKBACK CONFIG
# =============================================================================
# How far back (in days) to look for prior diagnoses relative to CXR study date.
# None = all prior history. Recommended: 365 or None depending on your causal assumptions.
LOOKBACK_DAYS: Optional[int] = None  # Set to 365 for 1-year window, None for all history


# =============================================================================
# 2. ICD CODE MAPPINGS — The Candidate W_pre Variables
# =============================================================================
# Each entry defines one candidate U_j in W_pre.
# Keys = variable name, Values = dict with 'icd10' and 'icd9' prefix lists.
#
# Design rationale:
#   - These are clinically established upstream causes of cardiomegaly,
#     pleural effusion, or conditions whose ABSENCE characterizes no_finding.
#   - The list is intentionally BROAD (overcomplete). The Stage 1 search
#     will select the sparse subset V*_Y that actually explains I(A; Y | V).
#   - Prefixes are used: e.g., 'I50' matches I50.0, I50.1, I50.20, etc.

WPRE_ICD_MAP: Dict[str, Dict[str, List[str]]] = {

    # =========================================================================
    # CARDIOVASCULAR
    # =========================================================================
    "chf": {
        "desc": "Congestive heart failure",
        "icd10": ["I50"],
        "icd9":  ["428"],
    },
    "hypertension": {
        "desc": "Essential and secondary hypertension",
        "icd10": ["I10", "I11", "I12", "I13", "I15", "I16"],
        "icd9":  ["401", "402", "403", "404", "405"],
    },
    "ihd_cad": {
        "desc": "Ischemic heart disease / coronary artery disease",
        "icd10": ["I20", "I21", "I22", "I23", "I24", "I25"],
        "icd9":  ["410", "411", "412", "413", "414"],
    },
    "valvular_disease": {
        "desc": "Valvular heart disease (rheumatic and non-rheumatic)",
        "icd10": ["I05", "I06", "I07", "I08", "I09", "I34", "I35", "I36", "I37"],
        "icd9":  ["394", "395", "396", "397", "424"],
    },
    "cardiomyopathy": {
        "desc": "Cardiomyopathy (dilated, hypertrophic, other)",
        "icd10": ["I42", "I43"],
        "icd9":  ["425"],
    },
    "atrial_fibrillation": {
        "desc": "Atrial fibrillation and flutter",
        "icd10": ["I48"],
        "icd9":  ["427.3"],  # 427.31 AFib, 427.32 AFlutter
    },
    "pericardial_disease": {
        "desc": "Pericarditis, pericardial effusion",
        "icd10": ["I30", "I31", "I32"],
        "icd9":  ["420", "423"],
    },
    "pulmonary_hypertension": {
        "desc": "Pulmonary hypertension",
        "icd10": ["I27"],
        "icd9":  ["416"],
    },
    "congenital_heart": {
        "desc": "Congenital heart disease",
        "icd10": ["Q20", "Q21", "Q22", "Q23", "Q24", "Q25"],
        "icd9":  ["745", "746", "747"],
    },

    # =========================================================================
    # PULMONARY
    # =========================================================================
    "copd": {
        "desc": "COPD and emphysema",
        "icd10": ["J43", "J44"],
        "icd9":  ["492", "496"],
    },
    "pneumonia": {
        "desc": "Pneumonia (bacterial, viral, organism unspecified)",
        "icd10": ["J12", "J13", "J14", "J15", "J16", "J17", "J18"],
        "icd9":  ["480", "481", "482", "483", "484", "485", "486"],
    },
    "pulmonary_embolism": {
        "desc": "Pulmonary embolism",
        "icd10": ["I26"],
        "icd9":  ["415.1"],
    },
    "tuberculosis": {
        "desc": "Tuberculosis (pulmonary and extrapulmonary)",
        "icd10": ["A15", "A16", "A17", "A18", "A19"],
        "icd9":  ["010", "011", "012", "013", "014", "015", "016", "017", "018"],
    },
    "asthma": {
        "desc": "Asthma",
        "icd10": ["J45", "J46"],
        "icd9":  ["493"],
    },
    "ild_pulm_fibrosis": {
        "desc": "Interstitial lung disease / pulmonary fibrosis",
        "icd10": ["J84"],
        "icd9":  ["516"],
    },

    # =========================================================================
    # RENAL
    # =========================================================================
    "ckd": {
        "desc": "Chronic kidney disease (all stages)",
        "icd10": ["N18"],
        "icd9":  ["585"],
    },
    "esrd_dialysis": {
        "desc": "End-stage renal disease / dialysis status",
        "icd10": ["N19", "Z99.2"],
        "icd9":  ["586", "V45.1"],
    },
    "nephrotic_syndrome": {
        "desc": "Nephrotic syndrome",
        "icd10": ["N04"],
        "icd9":  ["581"],
    },
    "aki": {
        "desc": "Acute kidney injury",
        "icd10": ["N17"],
        "icd9":  ["584"],
    },

    # =========================================================================
    # HEPATIC
    # =========================================================================
    "cirrhosis": {
        "desc": "Liver cirrhosis (alcoholic and non-alcoholic)",
        "icd10": ["K70.3", "K74"],
        "icd9":  ["571.2", "571.5", "571.6"],
    },
    "liver_failure": {
        "desc": "Hepatic failure",
        "icd10": ["K72"],
        "icd9":  ["570", "572.2"],
    },

    # =========================================================================
    # METABOLIC / ENDOCRINE
    # =========================================================================
    "diabetes": {
        "desc": "Diabetes mellitus (type 1 and 2)",
        "icd10": ["E10", "E11", "E13"],
        "icd9":  ["250"],
    },
    "obesity": {
        "desc": "Obesity",
        "icd10": ["E66"],
        "icd9":  ["278.0"],
    },
    "thyroid_disease": {
        "desc": "Thyroid disorders (hypo/hyperthyroidism)",
        "icd10": ["E01", "E02", "E03", "E04", "E05", "E06"],
        "icd9":  ["240", "241", "242", "243", "244", "245", "246"],
    },
    "hyperlipidemia": {
        "desc": "Hyperlipidemia / dyslipidemia",
        "icd10": ["E78"],
        "icd9":  ["272"],
    },

    # =========================================================================
    # ONCOLOGIC
    # =========================================================================
    "lung_cancer": {
        "desc": "Malignant neoplasm of lung/bronchus",
        "icd10": ["C34"],
        "icd9":  ["162"],
    },
    "breast_cancer": {
        "desc": "Malignant neoplasm of breast",
        "icd10": ["C50"],
        "icd9":  ["174", "175"],
    },
    "lymphoma": {
        "desc": "Lymphoma (Hodgkin and non-Hodgkin)",
        "icd10": ["C81", "C82", "C83", "C84", "C85", "C86"],
        "icd9":  ["200", "201", "202"],
    },
    "metastatic_cancer": {
        "desc": "Secondary / metastatic malignancy",
        "icd10": ["C77", "C78", "C79"],
        "icd9":  ["196", "197", "198"],
    },
    "leukemia": {
        "desc": "Leukemia",
        "icd10": ["C91", "C92", "C93", "C94", "C95"],
        "icd9":  ["204", "205", "206", "207", "208"],
    },

    # =========================================================================
    # INFECTIOUS
    # =========================================================================
    "sepsis": {
        "desc": "Sepsis / septicemia",
        "icd10": ["A40", "A41", "R65.2"],
        "icd9":  ["038", "995.91", "995.92"],
    },
    "hiv": {
        "desc": "HIV / AIDS",
        "icd10": ["B20"],
        "icd9":  ["042"],
    },

    # =========================================================================
    # AUTOIMMUNE / RHEUMATIC
    # =========================================================================
    "autoimmune_rheumatic": {
        "desc": "Systemic autoimmune / rheumatic disease (SLE, RA, scleroderma, etc.)",
        "icd10": ["M05", "M06", "M32", "M33", "M34", "M35"],
        "icd9":  ["710", "714"],
    },

    # =========================================================================
    # SUBSTANCE USE / BEHAVIORAL
    # =========================================================================
    "alcohol_use": {
        "desc": "Alcohol use disorder / dependence",
        "icd10": ["F10"],
        "icd9":  ["303", "305.0"],
    },
    "tobacco_use": {
        "desc": "Tobacco / nicotine use",
        "icd10": ["F17", "Z87.891"],
        "icd9":  ["305.1", "V15.82"],
    },
    "substance_use_other": {
        "desc": "Other substance use disorders (opioids, cocaine, etc.)",
        "icd10": ["F11", "F12", "F13", "F14", "F15", "F16", "F19"],
        "icd9":  ["304", "305.2", "305.3", "305.4", "305.5", "305.6", "305.7", "305.9"],
    },

    # =========================================================================
    # HEMATOLOGIC
    # =========================================================================
    "anemia": {
        "desc": "Anemia (iron deficiency, chronic disease, other)",
        "icd10": ["D50", "D51", "D52", "D53", "D63", "D64"],
        "icd9":  ["280", "281", "282", "283", "284", "285"],
    },
    "coagulopathy": {
        "desc": "Coagulation disorders",
        "icd10": ["D65", "D66", "D67", "D68"],
        "icd9":  ["286"],
    },

    # =========================================================================
    # OTHER RELEVANT CONDITIONS
    # =========================================================================
    "sleep_apnea": {
        "desc": "Sleep apnea (obstructive, central)",
        "icd10": ["G47.3"],
        "icd9":  ["327.2", "780.57"],
    },
    "fluid_overload": {
        "desc": "Fluid overload / volume overload",
        "icd10": ["E87.7"],
        "icd9":  ["276.69"],
    },
}


# =============================================================================
# 3. PROCEDURE-BASED W_pre CANDIDATES (from procedures_icd)
# =============================================================================
# Prior procedures can be upstream causes of findings on CXR.
WPRE_PROC_MAP: Dict[str, Dict[str, List[str]]] = {
    "prior_cabg": {
        "desc": "Prior coronary artery bypass graft",
        "icd10": ["0210", "0211", "0212", "0213"],  # ICD-10-PCS
        "icd9":  ["36.1"],                            # ICD-9-CM procedure
    },
    "prior_valve_surgery": {
        "desc": "Prior valve replacement / repair",
        "icd10": ["02RF", "02RG", "02RH", "02RJ"],
        "icd9":  ["35.1", "35.2"],
    },
    "prior_dialysis": {
        "desc": "History of dialysis procedures",
        "icd10": ["5A1D"],
        "icd9":  ["39.95"],
    },
    "prior_thoracentesis": {
        "desc": "Prior thoracentesis (drainage of pleural effusion)",
        "icd10": ["0W9B", "0W9930Z"],
        "icd9":  ["34.04", "34.91"],
    },
    "prior_chemo_radiation": {
        "desc": "Prior chemotherapy or radiation therapy",
        "icd10": ["3E04305"],  # simplified
        "icd9":  ["99.25", "92.2"],
    },
}


# =============================================================================
# 4. HELPER FUNCTIONS
# =============================================================================

def icd_matches_any_prefix(code: str, prefixes: List[str]) -> bool:
    """Check if an ICD code matches any of the given prefixes."""
    code_clean = str(code).strip().upper()
    for prefix in prefixes:
        prefix_clean = prefix.strip().upper()
        if code_clean.startswith(prefix_clean):
            return True
    return False


def load_cxr_cohort() -> pd.DataFrame:
    """
    Load MIMIC-CXR labels + metadata, return DataFrame with:
      subject_id, study_id, StudyDate, and task labels.
    """
    print("[1/6] Loading MIMIC-CXR labels and metadata...")

    labels = pd.read_csv(CXR_LABELS_FILE)
    meta = pd.read_csv(CXR_META_FILE)

    # Merge to get StudyDate per study
    # metadata has one row per DICOM; aggregate to study level
    study_dates = (
        meta.groupby(["subject_id", "study_id"])["StudyDate"]
        .first()
        .reset_index()
    )

    cxr = labels.merge(study_dates, on=["subject_id", "study_id"], how="left")

    # Parse StudyDate (format: YYYYMMDD)
    cxr["StudyDate"] = pd.to_datetime(cxr["StudyDate"], format="%Y%m%d", errors="coerce")

    print(f"    Loaded {len(cxr)} CXR studies for {cxr['subject_id'].nunique()} patients.")

    # Show label distribution for the three tasks of interest
    for task in ["Cardiomegaly", "Pleural Effusion", "No Finding"]:
        if task in cxr.columns:
            counts = cxr[task].value_counts(dropna=False)
            print(f"    {task}: +1={counts.get(1.0, 0)}, "
                  f"-1(uncertain)={counts.get(-1.0, 0)}, "
                  f"0={counts.get(0.0, 0)}, "
                  f"NaN={cxr[task].isna().sum()}")

    return cxr


def load_mimic_iv_tables() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the three core MIMIC-IV tables needed for W_pre extraction.
    Returns: (admissions, diagnoses, procedures)
    """
    print("[2/6] Loading MIMIC-IV tables...")

    admissions = pd.read_csv(
        ADMISSIONS_FILE,
        usecols=["subject_id", "hadm_id", "admittime", "dischtime"],
        parse_dates=["admittime", "dischtime"],
    )
    print(f"    admissions: {len(admissions)} rows")

    diagnoses = pd.read_csv(
        DIAGNOSES_FILE,
        usecols=["subject_id", "hadm_id", "icd_code", "icd_version"],
        dtype={"icd_code": str, "icd_version": int},
    )
    print(f"    diagnoses_icd: {len(diagnoses)} rows")

    procedures = pd.read_csv(
        PROCEDURES_FILE,
        usecols=["subject_id", "hadm_id", "icd_code", "icd_version"],
        dtype={"icd_code": str, "icd_version": int},
    )
    print(f"    procedures_icd: {len(procedures)} rows")

    return admissions, diagnoses, procedures


def temporal_join_cxr_to_admissions(
    cxr: pd.DataFrame,
    admissions: pd.DataFrame,
    lookback_days: Optional[int] = None,
) -> pd.DataFrame:
    """
    Join CXR studies to MIMIC-IV admissions with temporal precedence.

    For each CXR study, find all admissions for the same subject_id
    where admittime <= StudyDate (and optionally within lookback window).

    This enforces the causal constraint: U must temporally precede Y.
    """
    print("[3/6] Temporal join: CXR studies → prior admissions...")

    merged = cxr[["subject_id", "study_id", "StudyDate"]].merge(
        admissions[["subject_id", "hadm_id", "admittime"]],
        on="subject_id",
        how="inner",
    )

    # Enforce temporal precedence: admission before (or during) the CXR
    merged = merged[merged["admittime"] <= merged["StudyDate"]]

    # Optional lookback window
    if lookback_days is not None:
        cutoff = merged["StudyDate"] - pd.Timedelta(days=lookback_days)
        merged = merged[merged["admittime"] >= cutoff]
        print(f"    Applied {lookback_days}-day lookback window.")

    print(f"    {len(merged)} (study, admission) pairs after temporal filtering.")
    print(f"    Covers {merged['study_id'].nunique()} studies "
          f"and {merged['hadm_id'].nunique()} admissions.")

    return merged


def build_diagnosis_indicators(
    temporal_pairs: pd.DataFrame,
    diagnoses: pd.DataFrame,
    icd_map: Dict,
) -> pd.DataFrame:
    """
    For each (study_id, hadm_id) pair, check if any diagnosis code matches
    each W_pre candidate. Then aggregate to study level (OR across admissions).

    Returns: DataFrame indexed by (subject_id, study_id) with binary columns.
    """
    print("[4/6] Building diagnosis-based W_pre indicators...")

    # Get diagnosis codes for the relevant admissions
    relevant_dx = temporal_pairs[["subject_id", "study_id", "hadm_id"]].merge(
        diagnoses[["subject_id", "hadm_id", "icd_code", "icd_version"]],
        on=["subject_id", "hadm_id"],
        how="inner",
    )
    print(f"    {len(relevant_dx)} diagnosis records to scan.")

    # For each W_pre variable, create a binary indicator
    results = {}
    for var_name, var_info in icd_map.items():
        icd10_prefixes = var_info.get("icd10", [])
        icd9_prefixes = var_info.get("icd9", [])

        mask_10 = (
            (relevant_dx["icd_version"] == 10)
            & relevant_dx["icd_code"].apply(
                lambda c: icd_matches_any_prefix(c, icd10_prefixes)
            )
        ) if icd10_prefixes else pd.Series(False, index=relevant_dx.index)

        mask_9 = (
            (relevant_dx["icd_version"] == 9)
            & relevant_dx["icd_code"].apply(
                lambda c: icd_matches_any_prefix(c, icd9_prefixes)
            )
        ) if icd9_prefixes else pd.Series(False, index=relevant_dx.index)

        matched = relevant_dx[mask_10 | mask_9]

        # Aggregate to study level: 1 if ANY prior admission had this diagnosis
        study_indicator = (
            matched.groupby(["subject_id", "study_id"])
            .size()
            .clip(upper=1)
            .rename(var_name)
        )
        results[var_name] = study_indicator
        n_pos = len(study_indicator)
        print(f"    {var_name}: {n_pos} studies with positive indicator")

    # Combine all indicators
    all_studies = temporal_pairs[["subject_id", "study_id"]].drop_duplicates()
    wpre = all_studies.set_index(["subject_id", "study_id"])

    for var_name, indicator in results.items():
        wpre = wpre.join(indicator, how="left")

    wpre = wpre.fillna(0).astype(int)
    return wpre.reset_index()


def build_procedure_indicators(
    temporal_pairs: pd.DataFrame,
    procedures: pd.DataFrame,
    proc_map: Dict,
) -> pd.DataFrame:
    """
    Same logic as diagnosis indicators, but for procedures.
    """
    print("[5/6] Building procedure-based W_pre indicators...")

    relevant_proc = temporal_pairs[["subject_id", "study_id", "hadm_id"]].merge(
        procedures[["subject_id", "hadm_id", "icd_code", "icd_version"]],
        on=["subject_id", "hadm_id"],
        how="inner",
    )

    results = {}
    for var_name, var_info in proc_map.items():
        icd10_prefixes = var_info.get("icd10", [])
        icd9_prefixes = var_info.get("icd9", [])

        mask_10 = (
            (relevant_proc["icd_version"] == 10)
            & relevant_proc["icd_code"].apply(
                lambda c: icd_matches_any_prefix(c, icd10_prefixes)
            )
        ) if icd10_prefixes else pd.Series(False, index=relevant_proc.index)

        mask_9 = (
            (relevant_proc["icd_version"] == 9)
            & relevant_proc["icd_code"].apply(
                lambda c: icd_matches_any_prefix(c, icd9_prefixes)
            )
        ) if icd9_prefixes else pd.Series(False, index=relevant_proc.index)

        matched = relevant_proc[mask_10 | mask_9]

        study_indicator = (
            matched.groupby(["subject_id", "study_id"])
            .size()
            .clip(upper=1)
            .rename(var_name)
        )
        results[var_name] = study_indicator
        n_pos = len(study_indicator)
        print(f"    {var_name}: {n_pos} studies with positive indicator")

    all_studies = temporal_pairs[["subject_id", "study_id"]].drop_duplicates()
    wproc = all_studies.set_index(["subject_id", "study_id"])

    for var_name, indicator in results.items():
        wproc = wproc.join(indicator, how="left")

    wproc = wproc.fillna(0).astype(int)
    return wproc.reset_index()


def add_demographic_wpre(cxr: pd.DataFrame) -> pd.DataFrame:
    """
    Add patient-level demographic variables that may serve as W_pre.
    
    NOTE: age and sex might be your subgroup variable A, in which case
    do NOT include them in W_pre (they'd be on the wrong side of the DAG).
    This function provides them as options; exclude as needed.
    """
    print("[5b/6] Adding demographic W_pre (optional — exclude if A = sex or age)...")

    patients = pd.read_csv(
        PATIENTS_FILE,
        usecols=["subject_id", "anchor_age", "gender"],
    )

    cxr_demo = cxr[["subject_id", "study_id"]].merge(
        patients, on="subject_id", how="left"
    )

    # Binarize age: elderly (>=65) as a potential upstream variable
    cxr_demo["age_gte_65"] = (cxr_demo["anchor_age"] >= 65).astype(int)
    cxr_demo["age_gte_75"] = (cxr_demo["anchor_age"] >= 75).astype(int)

    # Gender as binary (M=1, F=0) — only include if gender is NOT your subgroup A
    cxr_demo["male"] = (cxr_demo["gender"] == "M").astype(int)

    return cxr_demo[["subject_id", "study_id", "age_gte_65", "age_gte_75", "male"]]


# =============================================================================
# 5. MAIN PIPELINE
# =============================================================================

def main():
    # --- Load data ---
    cxr = load_cxr_cohort()
    admissions, diagnoses, procedures = load_mimic_iv_tables()

    # --- Temporal join ---
    temporal_pairs = temporal_join_cxr_to_admissions(
        cxr, admissions, lookback_days=LOOKBACK_DAYS
    )

    # --- Diagnosis indicators ---
    wpre_dx = build_diagnosis_indicators(temporal_pairs, diagnoses, WPRE_ICD_MAP)

    # --- Procedure indicators ---
    wpre_proc = build_procedure_indicators(temporal_pairs, procedures, WPRE_PROC_MAP)

    # --- Merge diagnosis + procedure indicators ---
    wpre = wpre_dx.merge(wpre_proc, on=["subject_id", "study_id"], how="outer")
    wpre = wpre.fillna(0)

    # --- (Optional) Add demographics ---
    # IMPORTANT: If your subgroup variable A is sex or race, do NOT include
    # sex/race here. Only include demographics that are NOT A.
    # Uncomment the following to add:
    #
    # wpre_demo = add_demographic_wpre(cxr)
    # wpre = wpre.merge(wpre_demo, on=["subject_id", "study_id"], how="left")

    # --- Merge back the CXR labels for convenience ---
    label_cols = ["subject_id", "study_id", "StudyDate",
                  "Cardiomegaly", "Pleural Effusion", "No Finding"]
    label_cols = [c for c in label_cols if c in cxr.columns]
    wpre_full = wpre.merge(cxr[label_cols], on=["subject_id", "study_id"], how="left")

    # --- Summary statistics ---
    print("\n" + "=" * 70)
    print("W_pre MATRIX SUMMARY")
    print("=" * 70)

    wpre_var_cols = [c for c in wpre_full.columns
                     if c not in ["subject_id", "study_id", "StudyDate",
                                  "Cardiomegaly", "Pleural Effusion", "No Finding"]]

    print(f"Total studies with at least one prior admission: {len(wpre_full)}")
    print(f"Total W_pre candidate variables: {len(wpre_var_cols)}")
    print(f"\nPrevalence of each W_pre variable:")
    for col in sorted(wpre_var_cols):
        n_pos = int(wpre_full[col].sum())
        pct = 100.0 * n_pos / len(wpre_full) if len(wpre_full) > 0 else 0
        print(f"  {col:30s}  {n_pos:>7d}  ({pct:5.1f}%)")

    # --- Handle studies with NO prior admissions in MIMIC-IV ---
    # These are CXR studies where the patient has no MIMIC-IV admission record
    # prior to the CXR. They get all-zero W_pre vectors.
    all_cxr_studies = cxr[["subject_id", "study_id"]].drop_duplicates()
    studies_with_wpre = wpre_full[["subject_id", "study_id"]].drop_duplicates()
    missing = all_cxr_studies.merge(
        studies_with_wpre, on=["subject_id", "study_id"],
        how="left", indicator=True
    )
    n_missing = (missing["_merge"] == "left_only").sum()
    print(f"\nStudies with NO prior MIMIC-IV admission: {n_missing} "
          f"(will have all-zero W_pre)")

    # Add missing studies with all-zero W_pre
    if n_missing > 0:
        missing_studies = missing[missing["_merge"] == "left_only"][
            ["subject_id", "study_id"]
        ]
        for col in wpre_var_cols:
            missing_studies[col] = 0
        # Merge labels
        missing_studies = missing_studies.merge(
            cxr[label_cols], on=["subject_id", "study_id"], how="left"
        )
        wpre_full = pd.concat([wpre_full, missing_studies], ignore_index=True)

    # --- Save ---
    wpre_full.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved W_pre matrix to {OUTPUT_FILE}")
    print(f"Final shape: {wpre_full.shape}")
    print(f"Columns: {list(wpre_full.columns)}")


if __name__ == "__main__":
    main()


# =============================================================================
# 6. OPTIONAL: Lab-based W_pre variables
# =============================================================================
# Uncomment and integrate if you want lab-derived binary variables.
# These require loading labevents (large file), so kept separate.
#
# LAB_WPRE = {
#     "elevated_bnp": {
#         "desc": "Elevated BNP (>100 pg/mL) — heart failure biomarker",
#         "itemids": [50963],  # BNP
#         "threshold": 100,
#         "direction": "above",
#     },
#     "elevated_ntprobnp": {
#         "desc": "Elevated NT-proBNP (>300 pg/mL)",
#         "itemids": [50963],  # check exact itemid in your MIMIC version
#         "threshold": 300,
#         "direction": "above",
#     },
#     "elevated_creatinine": {
#         "desc": "Elevated creatinine (>1.5 mg/dL) — renal dysfunction",
#         "itemids": [50912],
#         "threshold": 1.5,
#         "direction": "above",
#     },
#     "low_albumin": {
#         "desc": "Low albumin (<3.5 g/dL) — liver/nutritional",
#         "itemids": [50862],
#         "threshold": 3.5,
#         "direction": "below",
#     },
#     "elevated_wbc": {
#         "desc": "Elevated WBC (>11 K/uL) — infection marker",
#         "itemids": [51301],
#         "threshold": 11,
#         "direction": "above",
#     },
#     "low_hemoglobin": {
#         "desc": "Low hemoglobin (<10 g/dL) — anemia",
#         "itemids": [51222],
#         "threshold": 10,
#         "direction": "below",
#     },
# }
#
# def build_lab_indicators(temporal_pairs, labevents, lab_map):
#     """
#     For each (study_id), check if any lab value in the lookback
#     window crosses the threshold.
#     """
#     # Similar join logic as diagnoses, but with valuenum thresholding.
#     # Left as an exercise — the join key is (subject_id, hadm_id).
#     pass

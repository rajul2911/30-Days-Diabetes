# Diabetes Readmission Prediction - Project Report

**Student Name:** Arpit Panwar 
**Course:** AI in Healthcare  
**Date:** November 3, 2025  
**Project Title:** Predicting 30-Day Hospital Readmission for Diabetic Patients

---

## Executive Summary

This project develops a machine learning-based Clinical Decision Support System (CDSS) to predict 30-day hospital readmissions for diabetic patients. Using the UCI Diabetes 130-US Hospitals dataset (1999-2008) with over 100,000 patient encounters, we trained and compared four classification models. The key innovation is handling severe class imbalance (~11% positive class) using **class weights instead of SMOTE**, avoiding synthetic data generation for better real-world generalization.

**Key Results:**
- **Best Model:** LightGBM with AUC: 0.6791, Recall: 0.5863
- **Financial Impact:** Estimated savings of $23.7M with 651% ROI
- **Clinical Impact:** 23.5% reduction in readmissions through targeted interventions

---

## 1. Data Loading & Exploration

**Objective:** Load and understand the diabetes dataset structure.

**Process:**
- Loaded 101,766 patient encounters with 50 features
- Examined missing values: 49% missing in `weight`, 40% in `payer_code`, 53% in `medical_specialty`
- Analyzed target variable: 11.15% readmitted <30 days (severe class imbalance)

**Key Findings:**
- Highly imbalanced dataset requiring specialized handling
- Significant missing data in non-essential features
- Rich clinical features including diagnoses, medications, and procedures

---

## 2. Data Cleaning & Preparation

**Objective:** Clean and prepare data for modeling.

**Actions Taken:**
- Replaced '?' placeholders with NaN
- Dropped 5 columns with >40% missing values or low relevance
- Removed rows with missing primary diagnosis and invalid gender
- Final dataset: 101,742 encounters

**Results:**
- Clean dataset ready for feature engineering
- Minimal data loss (~3.6% of original)
- All remaining features have <5% missing values

---

## 3. Target Variable Creation

**Objective:** Create binary classification target.

**Approach:**
- Target = 1 if readmitted <30 days, 0 otherwise
- Collapsed 3-class problem (NO, >30, <30) into binary

**Distribution:**
- Class 0 (No readmission <30): 87,021 (88.8%)
- Class 1 (Readmission <30): 11,031 (11.2%)
- **Class imbalance ratio: 7.9:1**

---

## 4. Feature Engineering - ICD-9 Diagnosis Mapping

**Objective:** Convert granular ICD-9 codes into meaningful clinical categories.

**Process:**
- Mapped 3 diagnosis columns (diag_1, diag_2, diag_3) to 18 clinical categories
- Categories include: Circulatory, Respiratory, Diabetes, Digestive, etc.
- Reduced dimensionality while preserving clinical meaning

**Results:**
- Top primary diagnoses: Circulatory (25%), Respiratory (14%), Diabetes (16%)
- Created interpretable features for model training
- Improved feature relevance for clinical decision-making

---

## 5. Preprocessing Pipeline

**Objective:** Transform features for machine learning.

**Pipeline Components:**
- **Numerical features (8):** StandardScaler normalization
- **Categorical features (36):** Imputation + One-Hot Encoding
- Combined using ColumnTransformer

**Results:**
- Original: 44 features
- After encoding: 216 features
- Data ready for multiple model types (sparse/dense formats)

---

## 6. Train-Test Split (No SMOTE)

**Objective:** Split data without synthetic oversampling.

**Approach:**
- 80/20 train-test split with stratification
- **No SMOTE applied** - key difference from traditional approach
- Models use class weights to handle imbalance

**Split Summary:**
- Training: 81,393 samples
- Testing: 20,349 samples
- Training class distribution preserved: 88.8% / 11.2%

---

## 7. Model Training & Evaluation

**Objective:** Train and compare 4 classification models with class weighting.

### Model 1: Logistic Regression (class_weight='balanced')
- **AUC:** 0.6739
- **Recall:** 0.5634
- **Precision:** 0.1822
- **F1-Score:** 0.2754
- Baseline model with balanced class weights

### Model 2: Gaussian Naive Bayes
- **AUC:** 0.5142
- **Recall:** 0.9903
- **Precision:** 0.1143
- **F1-Score:** 0.2050
- Highest recall but lowest precision

### Model 3: Random Forest (class_weight='balanced')
- **AUC:** 0.6615
- **Recall:** 0.2128
- **Precision:** 0.2377
- **F1-Score:** 0.2245
- Strong ensemble performance with feature importance

### Model 4: LightGBM (is_unbalance=True)
- **AUC:** 0.6791 🏆 **BEST**
- **Recall:** 0.5863
- **Precision:** 0.1827
- **F1-Score:** 0.2786
- Best overall performance with gradient boosting

**Winner:** LightGBM selected for further analysis

---

## 8. Model Explainability with SHAP

**Objective:** Understand feature importance and prediction drivers.

**SHAP Analysis Results:**
Top 5 Most Important Features:
1. **number_inpatient** - Prior inpatient visits strongly predict readmission
2. **discharge_disposition_id** - Discharge location impacts risk
3. **number_diagnoses** - Complexity of patient condition
4. **time_in_hospital** - Longer stays correlate with higher risk
5. **number_emergency** - Emergency department utilization history

**Key Insights:**
- Prior healthcare utilization is strongest predictor
- Medication changes and A1C results have moderate impact
- Diagnosis categories contribute to risk stratification
- SHAP values provide actionable insights for clinicians

---

## 9. Cost-Benefit Analysis

**Objective:** Quantify financial and clinical impact of model deployment.

**Assumptions:**
- Average readmission cost: $15,000
- Intervention cost per patient: $500
- False alarm cost: $100
- Intervention effectiveness: 40% reduction

**Financial Results:**
- **Total patients:** 20,349
- **Without model:** $34.05M in readmission costs
- **With model:** $10.34M total costs
- **Net Savings:** $23.7M
- **ROI:** 651%
- **Savings per patient:** $1,165

**Clinical Results:**
- **Readmissions prevented:** 532 (23.5% reduction)
- **Patients receiving intervention:** 7,284
- **Percentage reduction in readmission rate:** 23.5%

---

## 10. Threshold Optimization

**Objective:** Optimize decision thresholds for different clinical scenarios.

**Scenarios Analyzed:**

| Scenario | Threshold | Precision | Recall | F1-Score | Patients Flagged |
|----------|-----------|-----------|--------|----------|------------------|
| **Balanced (Max F1)** | 0.358 | 0.206 | 0.421 | 0.275 | 22.1% |
| **Safety Priority (High Recall ≥85%)** | 0.106 | 0.129 | 0.851 | 0.224 | 72.5% |
| **Resource-Limited (High Precision ≥30%)** | 0.578 | 0.301 | 0.165 | 0.213 | 6.0% |
| **Clinical Optimal (Youden)** | 0.353 | 0.204 | 0.424 | 0.276 | 22.9% |
| **Default** | 0.500 | 0.237 | 0.284 | 0.259 | 13.2% |

**Recommendations:**
- **ICU/High-risk units:** Safety Priority (catches 85% of readmissions)
- **General wards:** Balanced approach (optimal F1-score)
- **Resource-constrained:** High precision threshold (minimize false alarms)

---

## 11. Fairness & Bias Analysis

**Objective:** Ensure equitable performance across demographic groups.

### Performance by Race:
- **Caucasian:** AUC: 0.659, Recall: 0.427 (n=60,726)
- **African American:** AUC: 0.648, Recall: 0.402 (n=15,228)
- **Hispanic:** AUC: 0.681, Recall: 0.412 (n=1,553)
- **Asian:** AUC: 0.714, Recall: 0.375 (n=584)
- **Recall Disparity:** 0.052 (5.2%) ✅ Acceptable (<10%)

### Performance by Gender:
- **Female:** AUC: 0.665, Recall: 0.419
- **Male:** AUC: 0.652, Recall: 0.423
- Minimal gender bias detected

### Performance by Age Group:
- Consistent performance across age groups [30-90]
- Slight variation: elderly patients (>70) show marginally higher recall
- Overall age disparity: <5%

**Fairness Assessment:** ✅ Model demonstrates equitable performance across demographics with acceptable disparity levels.

---

## 12. Clinical Decision Support System (CDSS) Prototype

**Objective:** Demonstrate real-world clinical application.

**CDSS Features:**
- Patient-level risk scores (0-100%)
- Demographic and clinical summaries
- SHAP-based risk factor identification
- Tiered action recommendations:
  - 🔴 **High Risk (≥70%):** Immediate intervention, case manager assignment
  - 🟡 **Moderate Risk (50-69%):** Preventive measures, 14-day follow-up
  - 🟢 **Low Risk (<50%):** Standard discharge protocol

**Prototype Output:**
- Top 5 highest-risk patients identified
- Personalized risk profiles with top contributing factors
- Actionable clinical recommendations for each risk tier

---

## 13. Key Methodological Innovation

**No SMOTE Approach:**
- Traditional approach uses SMOTE to create synthetic minority class samples
- **Our approach:** Use class weights only, no synthetic data
- **Advantages:**
  - Models learn from real data distribution
  - Better generalization to production environments
  - Avoids overfitting to synthetic patterns
  - More trustworthy predictions for clinical use

**Class Weight Implementation:**
- Logistic Regression: `class_weight='balanced'`
- Random Forest: `class_weight='balanced'`
- LightGBM: `is_unbalance=True`
- Gaussian NB: Inherent probabilistic handling

---

## Conclusions

### Technical Achievements:
- Successfully handled severe class imbalance without synthetic data
- Achieved AUC of 0.6791 on real imbalanced test data
- Engineered meaningful features from complex medical codes
- Developed interpretable model with SHAP explanations
- Optimized for multiple clinical scenarios

### Clinical Impact:
- **23.5% reduction in readmissions** through targeted interventions
- **$23.7M estimated savings** for test population
- **Equitable performance** across demographics
- **Actionable insights** for healthcare providers
- **Customizable thresholds** for different hospital settings

### Real-World Applicability:
- Model ready for EHR integration
- Threshold optimization enables deployment flexibility
- SHAP explanations support clinical decision-making
- Fairness analysis ensures ethical deployment
- Cost-benefit analysis justifies implementation

### Limitations:
- Dataset from 1999-2008 may not reflect current practices
- Model performance (AUC 0.68) indicates room for improvement
- Requires external validation on independent datasets
- Need prospective clinical trials for definitive efficacy

### Future Work:
- Real-time EHR integration
- Temporal modeling with patient history
- Multi-center validation studies
- Ensemble methods combining multiple approaches
- Mobile app for patient engagement

---

## Technical Specifications

**Environment:**
- Python 3.x with scikit-learn, LightGBM, SHAP
- Jupyter Notebook environment
- StandardScaler, OneHotEncoder for preprocessing

**Model Hyperparameters (LightGBM):**
- n_estimators: 200
- learning_rate: 0.05
- num_leaves: 31
- is_unbalance: True
- random_state: 42

**Evaluation Metrics:**
- Primary: AUC-ROC (discrimination)
- Secondary: Recall, Precision, F1-Score
- Fairness: Demographic parity, equalized odds

---

## References

1. UCI Machine Learning Repository: Diabetes 130-US Hospitals Dataset
2. Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions (SHAP)
3. Clinical guidelines for diabetes readmission prevention
4. Healthcare cost estimates from CMS and medical literature

---

**Report Prepared By:** Arpit Panwar 
**Date:** November 3, 2025    

---

"""
Synthetic clinical notes for NLP demonstration.

Each note is intentionally rich with named entities across five types:
  DIAGNOSIS, MEDICATION, LAB_VALUE, PROCEDURE, CLINICAL_FINDING

Notes also include negated findings and historical vs. active distinctions
to exercise the negation-detection and context logic in the extractor.
"""

SAMPLE_NOTES = {
    # ------------------------------------------------------------------
    # 1. Discharge Summary — richest note, most entity types
    # ------------------------------------------------------------------
    "Discharge Summary – Acute HF Exacerbation": (
        "Patient is a 72-year-old female admitted for acute decompensated heart failure (I50.9) "
        "with a history of coronary artery disease (I25.10), essential hypertension (I10), and "
        "hyperlipidemia (E78.5). On admission she reported progressive shortness of breath and "
        "bilateral leg swelling for five days. She denied chest pain and denied fever. "
        "Exam notable for bibasilar crackles and 2+ pitting edema to the knees. "
        "BNP on admission 1,840 pg/mL. Creatinine 1.6 mg/dL. eGFR 38. Sodium 132 mEq/L. "
        "Chest X-ray showed pulmonary vascular congestion. Echocardiogram performed showing "
        "ejection fraction 35%. Patient was treated with IV furosemide 80mg twice daily with "
        "good diuretic response. Oral carvedilol 12.5mg twice daily and lisinopril 5mg daily "
        "were continued. Atorvastatin 40mg at bedtime was continued. At discharge, weight "
        "decreased 4.2 kg, creatinine improved to 1.3 mg/dL, and BNP trended down to 620 pg/mL. "
        "Patient ambulating without dyspnea at rest. Discharge instructions given. "
        "Follow-up with cardiology in one week."
    ),

    # ------------------------------------------------------------------
    # 2. Diabetes Progress Note
    # ------------------------------------------------------------------
    "Diabetes Progress Note – Routine Follow-Up": (
        "Patient is a 58-year-old male with a history of type 2 diabetes mellitus (E11.9), "
        "hypertension (I10), and stage 3 chronic kidney disease (N18.3) presenting for routine "
        "three-month follow-up. He reports good medication adherence and denies hypoglycemic "
        "episodes. He denies polyuria or polydipsia. "
        "Current medications: metformin 1000mg twice daily, lisinopril 10mg daily, "
        "atorvastatin 40mg at bedtime, and aspirin 81mg daily. "
        "Vitals: BP 138/86 mmHg, weight 94 kg, BMI 31.2. "
        "HbA1c today 7.8%. Fasting glucose 148 mg/dL. Creatinine stable at 1.8 mg/dL. "
        "eGFR 42. LDL 88 mg/dL. Urine microalbumin/creatinine ratio 45 mg/g. "
        "Foot exam performed — monofilament sensation intact bilaterally, no ulcers noted. "
        "Fundoscopic exam deferred to ophthalmology. Assessment: diabetes suboptimally "
        "controlled. Plan: add semaglutide 0.5mg subcutaneous weekly; recheck HbA1c and "
        "metabolic panel in three months; reinforce low-carbohydrate diet and daily walking."
    ),

    # ------------------------------------------------------------------
    # 3. ED Triage Note
    # ------------------------------------------------------------------
    "ED Triage Note – Chest Pain": (
        "Patient is a 61-year-old male presenting to the emergency department with acute onset "
        "chest pain rated 7/10, described as pressure-like, radiating to the left arm, onset "
        "approximately 90 minutes prior to arrival. Past medical history significant for "
        "coronary artery disease (I25.10) with prior percutaneous coronary intervention in 2019, "
        "atrial fibrillation (I48.91), hypertension (I10), and hyperlipidemia (E78.5). "
        "Current medications: apixaban 5mg twice daily, metoprolol succinate 50mg daily, "
        "rosuvastatin 20mg at bedtime, and nitroglycerin 0.4mg sublingual as needed. "
        "He denies shortness of breath at rest. He denies nausea or vomiting. "
        "Vitals: BP 158/94 mmHg, HR 88 bpm, SpO2 97% on room air. "
        "12-lead ECG performed — sinus rhythm, no ST elevation. "
        "Troponin I 0.04 ng/mL (borderline elevated). "
        "Aspirin 325mg given orally. IV access established. Continuous cardiac monitoring "
        "initiated. Serial troponins and repeat ECG ordered."
    ),

    # ------------------------------------------------------------------
    # 4. Post-Op Note
    # ------------------------------------------------------------------
    "Post-Op Note – Laparoscopic Cholecystectomy": (
        "Patient is a 45-year-old female, post-operative day 1 following laparoscopic "
        "cholecystectomy for acute cholecystitis. History of asthma (J45.909) and depression "
        "(F32.9). She has no history of coronary artery disease and no history of diabetes. "
        "She denies chest pain. She denies shortness of breath. "
        "Current medications: sertraline 50mg daily (home medication, restarted), "
        "ketorolac 15mg IV every 6 hours for pain, acetaminophen 650mg oral every 6 hours, "
        "and albuterol inhaler 2 puffs every 4 hours as needed. "
        "Vitals: BP 118/72 mmHg, HR 76 bpm, Temp 37.1°C, SpO2 98% on room air. "
        "WBC 11.2 x10³/µL (mildly elevated, expected post-op). Hemoglobin 12.4 g/dL. "
        "Abdomen soft, laparoscopic port sites clean and dry, no signs of infection. "
        "Bowel sounds present. Patient tolerating clear liquids. "
        "Plan: advance diet as tolerated, transition to oral analgesics, pulmonary toilet "
        "with incentive spirometry every hour while awake, discharge anticipated tomorrow."
    ),

    # ------------------------------------------------------------------
    # 5. Lab Results and Interpretation Note
    # ------------------------------------------------------------------
    "Lab Results & Interpretation – CKD Monitoring": (
        "Patient is a 64-year-old female with stage 3 chronic kidney disease (N18.3), "
        "type 2 diabetes mellitus (E11.9), and essential hypertension (I10). "
        "Lab results from today's draw: BMP notable for creatinine 2.1 mg/dL (increased from "
        "1.9 mg/dL three months ago), eGFR 29, potassium 5.1 mEq/L, bicarbonate 20 mEq/L. "
        "Phosphorus 4.8 mg/dL. PTH 72 pg/mL. Hemoglobin 10.6 g/dL consistent with anemia of "
        "chronic kidney disease. HbA1c 8.1%. Urine protein/creatinine ratio 320 mg/g, increased "
        "from 240 mg/g. Interpretation: CKD is progressing — reclassify as stage 3b to 4 "
        "transition. Anemia and secondary hyperparathyroidism developing. Current medications "
        "include losartan 50mg daily and amlodipine 5mg daily. Plan: nephrology referral placed. "
        "Increase losartan to 100mg daily. Initiate ferrous sulfate 325mg oral daily for anemia. "
        "Renal diet counseling ordered. Repeat metabolic panel in six weeks."
    ),

    # ------------------------------------------------------------------
    # 6. Primary Care SOAP Note
    # ------------------------------------------------------------------
    "Primary Care SOAP – COPD Exacerbation": (
        "S: Patient is a 69-year-old male with a history of COPD (J44.1) and essential "
        "hypertension (I10) presenting with increased cough productive of yellow-green sputum "
        "for four days and worsening dyspnea on exertion. He denies hemoptysis. "
        "He denies fever or chills. Current medications: tiotropium 18mcg inhaled daily, "
        "budesonide/formoterol 160/4.5mcg 2 puffs twice daily, amlodipine 5mg daily, and "
        "albuterol 2.5mg via nebulizer as needed. "
        "O: Temp 37.6°C, BP 142/88 mmHg, HR 92 bpm, RR 20, SpO2 91% on room air. "
        "FEV1 48% of predicted (prior baseline 55%). Exam: diffuse expiratory wheezes, "
        "prolonged expiration, no accessory muscle use at rest. "
        "Chest X-ray: hyperinflation, no focal consolidation. WBC 13.8 x10³/µL. "
        "A: Acute exacerbation of COPD, likely infectious. Pneumonia (J18.9) excluded by "
        "imaging. "
        "P: Prednisone 40mg oral daily for five days. Azithromycin 500mg oral daily for "
        "five days. Continue and increase albuterol nebulizations to every 4 hours. "
        "Oxygen supplementation to maintain SpO2 above 92%. Follow up in one week."
    ),
}

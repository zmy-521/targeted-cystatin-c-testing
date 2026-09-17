# Cystatin C–creatinine eGFR discordance: code-only reproduction

This repository implements the frozen analysis for **targeted cystatin C testing for large negative eGFR discordance in adults with diabetes**. It distributes Python source code and non-patient specifications only. It does **not** distribute patient records, identifiers, collection dates, raw spreadsheets, individual predictions, SHAP arrays, trained models, or manuscript figures.

**You must supply your own authorized development and external data matching the variable dictionary before running the analysis.** All inputs, private configuration files and outputs must be outside this repository. The program enforces that separation. Do not upload generated outputs alongside this code.

## Frozen analysis

- Development: 1,170 participants, 214 D2 events; primary external: 1,619 participants, 112 events.
- Cohort specification: `modeling-v1.2-single-patient-CBC-QC`. Model specification: `FINAL_MODEL_LOCK_v1.1`.
- D2: unrounded `eGFRcys_2012 / eGFRcr_2021 < 0.70`.
- Creatinine eGFR: CKD-EPI 2021; cystatin C eGFR: CKD-EPI 2012. No race coefficient.
- Final predictor order: **age, sex, eGFRcr_2021, HGB, PLT, BUN, ALB, ALP, GLU**.
- Clinical core: age, sex, eGFRcr_2021; logistic regression with its separately frozen parameters.
- Five-fold stratified CV, shuffle enabled, split seed 42. Model seed remains 42.
- Adjacent-panel stability: paired five-fold CV for split seeds 42 through 51, with no retuning.
- Final probability cutoff: **none**. Testing budgets are fixed rank allocations, not optimized cutoffs.

`config/frozen_analysis.json` is the executable scientific specification. It contains all seven classifiers' selected parameters and the clinical-core parameters, taken from the frozen development fit records. **No grid search is run.** The `historical_search_space` section of `config/frozen_analysis.json` records the originally approved search space for transparency only. Replaying the selected settings reproduces the selected-model OOF calculation without repeating parameter selection or changing the strategy.

Development estimates remain **development/model-selection performance**, not selection-independent internal validation. The original development search and OOF calculation shared folds; SHAP ranking also used the complete development set. This repository preserves that method rather than silently replacing it with nested CV.

## Install

Use Python 3.13.5 (the tested packaging runtime) and an isolated environment **outside the repository**. The frozen lock records package versions but does not record the original Python interpreter version; this package does not invent that provenance. The pinned modeling package versions match the frozen records; plotting and I/O versions match the tested local environment. Do not silently upgrade scikit-learn, XGBoost, SHAP, or NumPy for an exact replay.

```bash
python -m venv ../cysc-env
# Windows: ..\cysc-env\Scripts\activate
# macOS/Linux: source ../cysc-env/bin/activate
python -m pip install -r requirements.txt
```

The plotting code uses Times New Roman when installed, with a serif fallback. Exact typography requires that font to be available locally; no proprietary font files are bundled. Outputs are PNG and LZW TIFF at 300 dpi.

## Prepare private inputs

Copy `config/config.example.yaml` to a private working folder outside this repository. Point it to your data and adjust **field mappings and units only**. Read [Data preparation](#data-preparation-details) before setting the source attestations to true.

The preferred input is a numeric, one-index-observation-per-participant table for each cohort, before or after the frozen basic-CBC inclusion rule. CSV, XLSX and Parquet are supported. Missing values must remain missing, not pre-imputed. The 49 candidate names and canonical units are in the `variable_dictionary` section of `config/frozen_analysis.json`; CysC_mg_L is additionally required for phenotype calculation and is forbidden as a predictor.

For structured specimen-level chemistry/CBC tables, the specimen adapter example below documents an optional adapter. It selects the earliest eligible paired chemistry specimen and links whole auxiliary panels in the preceding seven days through the index day. Identity resolution, diabetes eligibility and source verification must be performed by the local data custodian; the repository does not contain or recreate private identity-linkage tables.

Exact numeric replay also requires the original **frozen row order** and private **tie order**. These are not distributed. See the data-preparation document for why the same seed alone is insufficient.

## Recommended execution order

Run commands from the repository root. Paths below are placeholders for your local private configuration.

```bash
python -B -m cysc --config ../private-work/config.yaml --stage prepare
python -B -m cysc --config ../private-work/config.yaml --stage development
python -B -m cysc --config ../private-work/config.yaml --stage reduction
python -B -m cysc --config ../private-work/config.yaml --stage stability
python -B -m cysc --config ../private-work/config.yaml --stage final
python -B -m cysc --config ../private-work/config.yaml --stage external
python -B -m cysc --config ../private-work/config.yaml --stage shap
python -B -m cysc --config ../private-work/config.yaml --stage targeted
python -B -m cysc --config ../private-work/config.yaml --stage profiles
python -B -m cysc --config ../private-work/config.yaml --stage figures
python -B -m cysc --config ../private-work/config.yaml --stage tables
```

Alternatively, use `--stage all` **once in a fresh output directory**. Finished stages are not overwritten. A changed configuration, code or frozen specification requires a new output directory. The external stage requires a saved model lock; no external fit or preprocessing fit occurs. Development stages cannot run after external evaluation in the same run.

| Stage | Scope and principal outputs |
|---|---|
| prepare | Field mapping, units, CKD-EPI/D1/D2/D3, basic-CBC inclusion, 49-variable missingness/eligibility, fixed 39/28 pools, correlation sources, cohort flow |
| development | Seven fixed-parameter classifiers plus clinical core; paired five-fold OOF; Full-model comparison |
| reduction | XGBoost/RF Full-pool Tree SHAP, all k=4–28 results, mandatory core, candidate tolerance and champion check |
| stability | Fixed 8/9 panels, 10 paired repeats, per-repeat metrics and mean/SD/median/IQR/range |
| final | Final nine-variable XGBoost and core development OOF, full-development fits and local lock |
| external | Locked predictions; paired Bootstrap performance/differences; raw and cross-fitted calibration; full-range raw-probability DCA |
| shap | Final nine-variable Tree SHAP, ranking, native-contribution and additivity checks |
| targeted | 10/20/30/40/50% allocations, 2,000 reranked Bootstrap samples per cohort, random-testing references and tie audit |
| profiles | Fixed top-20% four groups; D1/D3/related phenotype profiles, Wilson/Newcombe/log-RR and median-difference intervals |
| figures | Main Figures 2–6 and Supplementary Figures S1–S3, using the approved panel structures |
| tables | Main Table 1/Table S2 three-line DOCX, and an XLSX workbook of Table 1/S1–S9 aggregate source sheets |

The source workbook is not a complete supplementary manuscript document. Long eligibility and performance tables are supplied as aggregate sheets; narrative supplementary methods are not auto-written.

## Verification without access to patient data

```bash
python -B -m unittest discover -s tests -v
python -B tools/check_public_package.py
```

The integration checks used invented values drawn from probability distributions. The compact repository contains no generated fixtures or synthetic-data generator. `synthetic_test` relaxes only manuscript cohort/pool/candidate fingerprint checks, not the algorithms, hyperparameters, seeds, CV, Bootstrap count or analysis definitions. Synthetic results are software tests, never estimates of clinical performance.

See [Validation](#validation-details) for what was actually tested, and [Frozen methods](#frozen-method-details) for statistical details and interpretation limits. Without access to the authorized frozen inputs, this repository can reproduce the **workflow**, not independently demonstrate equality to every published number.

## Privacy and upload

Only upload this code directory. Do not add the private work folder, fitted models, OOF predictions, calibration probabilities, tie keys, source workbooks or output figures without a separate disclosure review. In particular, scatter plots and SHAP source arrays may expose individual-level measurements.

`tools/check_public_package.py` fails on disallowed file types and recognizable private literals. `.gitignore` adds another barrier, but neither replaces human review of new text. The supplied folder has no bundled data fixtures and no Git history. Uploading to GitHub is left to the repository owner.

NHANES validation, new model tuning, new feature selection, new cutoffs and additional main analyses are outside this frozen primary-analysis package.


## Data preparation details

# Private data preparation contract

## Canonical input

The dictionary enumerates 49 candidate variables, in the original audit order. Missing optional candidate fields become NA and are reported. The development and external tables must use the same canonical units. Supply `age`, `sex`, `SCr_mg_dL`, `CysC_mg_L`, `WBC`, `HGB` and `PLT`. Mandatory sex coding is 0 = female and 1 = male. No automatic recoding of unknown categories occurs.

SCr is mg/dL; hospital SCr in µmol/L is divided by 88.4. CysC is mg/L. The historical `BUN` name denotes the frozen urea-equivalent mmol/L field; do not feed an unconverted BUN mg/dL result. HGB/ALB/TP/GLB are g/L; hematology counts and percentages follow the dictionary. The public source-alias file contains analyte names and conversion factors only, not private source paths or records.

`A/G = ALB / GLB` is recomputed only when GLB > 0 and both values are available. GLB is not backfilled from TP−ALB. Non-HDL-C remains source-reported; no TC−HDL derivation is introduced. Supplied eGFR and label columns are not trusted as model inputs: equations and phenotypes are recomputed without rounding.

## Eligibility and index selection

The data custodian must apply the original cohort-specific stable assay windows, adult age ≥18 years, diabetes eligibility, reliable identity and same-specimen SCr/CysC verification. Unreliably linked registrations cannot be treated as unique patients. The earliest eligible paired observation is the index; same-day ties follow the original specimen sort. Chemistry/lipid features come from that specimen. CBC/HbA1c are selected as whole panels from the latest available record within index minus seven days through index, with stable specimen ordering for ties. Auxiliary records also remain inside the approved source period. A later index must not be selected merely because its CBC is more complete.

The optional specimen adapter expects already resolved private linkage keys, a verified diabetes-eligible flag and specimen-level wide chemistry rows. It is deliberately not a universal parser for unverified hospital exports. Configure period bounds locally; no real participant dates or identity-linkage information are supplied here.

## Invalid measurements

Before modeling, resolve physiologically impossible values and sentinel zeros against original records. If an entire CBC is confirmed invalid, mark its local invalid-panel flag, look for a genuinely valid record within the frozen auxiliary window, and otherwise leave its CBC values NA. The code stops on unresolved nonpositive core/red-cell measurements. It does not trim genuine extremes or assume every differential count of zero is invalid.

Final modeling inclusion requires **valid observed WBC, HGB and PLT simultaneously**. No imputation can rescue this criterion. Neither a raw patient-missingness fraction nor lipid completeness is an additional exclusion. All other eligible predictor missingness is imputed within training folds only. The frozen input must already embody the source-adjudicated single-patient CBC correction; no private identifier for that correction is published or hard-coded.

## Frozen order and tie resolution

Preserve the original modeling table order. Scikit-learn's shuffled stratified split is reproducible conditional on both its seed and its input order. Sorting on a new identifier will change folds even with seed 42.

The original testing-prioritization order is descending raw probability, then the first 64 bits of SHA256 of `42|cohort_namespace|private_study_key`, ascending. The original cohort namespaces are `Wanbei` and `Anyi`. This formula is retained in code; **the private keys and any upstream identity salt are not supplied**. During Bootstrap, occurrence position is the last tie-breaker for repeated draws.

For exact replay, supply a local `tie_key_column` of unique uint64 values encoded as decimal strings, or point `private_study_key_column` at the already-existing private study keys. Do not regenerate study keys from raw identity numbers with a new namespace and expect identical boundary selection. New derived tie keys must stay outside the public repository. The program does not need patient identity in the downstream model matrices.

## Two input modes

1. `index`: preferred for replay. A custodian supplies one eligible index row per participant with correctly linked panels, exact order and resolved QC. All seven attestations must be explicitly true.
2. `specimen`: optional standardized adapter. Supply wide chemistry and auxiliary tables with private linkage columns. The code selects the index first, attaches valid auxiliary panels, then performs basic-CBC inclusion. Frozen study windows and source identity resolution are local responsibilities.

Input files and output locations inside the public code repository are rejected. Generated Parquet, prediction arrays and fitted pipelines are private working artifacts even if direct identifiers have been removed. They must not be uploaded with this package.


## Frozen method details

# Frozen statistical and figure contract

## Phenotypes and variable governance

For sex 0/1 = female/male, CKD-EPI 2021 creatinine uses κ = 0.7/0.9, α = −0.241/−0.302:

`142 × min(SCr/κ,1)^α × max(SCr/κ,1)^−1.200 × 0.9938^age × (1.012 if female)`.

CKD-EPI 2012 cystatin C is:

`133 × min(CysC/0.8,1)^−0.499 × max(CysC/0.8,1)^−1.328 × 0.996^age × (0.932 if female)`.

D2 is a ratio <0.70; D1 is eGFRcys−eGFRcr ≤−15. D3 is eGFRcys <60 only among eGFRcr ≥60. The saved D3 joint indicator is not used with an all-participant prevalence denominator.

The frozen chain is 49 initial candidates → 41 passing missingness ≤25% in both primary cohorts → 40 after SCr structural exclusion → 39 after CO2 comparability exclusion → 28 after explicit clinical redundancy review. LDH, P, HbA1c, TC, HDL-C, LDL-C, TG and Non-HDL-C fail the frozen missingness stage. SCr is structurally redundant with mandatory eGFRcr; CO2 comparability was unresolved. Eleven further exclusions and their original reasons are executable metadata. Correlation pairs at |rho| ≥0.70/0.80 are descriptive evidence, not an automatic pruning algorithm. ALT/AST and mandatory age/eGFRcr remain despite correlation.

Only model-preparation metadata and availability may use external data before lock. External outcomes, discrimination, calibration and utility must never select the variable pool or model. The 39-variable Spearman matrix uses development pairwise-complete measurements, with no imputation. CysC, eGFRcys and all their derivatives are never predictors.

## Development and final model

Seven models: logistic regression, support vector machine, k-nearest neighbors, decision tree, random forest, XGBoost and LightGBM. Pipelines fit median imputation only on training folds. LR/SVM/KNN and the clinical core additionally fit StandardScaler in each training fold. No class weighting, class balancing, synthetic resampling or outcome-based imputation is applied.

The published seven-model settings are replayed without retuning. Primary ranking is average precision (reported as AUPRC), then AUROC, then Brier. Candidate SHAP is fitted on the complete development cohort at the selected settings; XGBoost uses raw-margin/log-odds SHAP, RF uses positive-class probability contributions. Additivity is checked.

For each candidate, rank by descending mean absolute SHAP, retain age/sex/eGFRcr at every k, and add the top remaining features. Preserve original Full-pool column order in every subset. Evaluate every k = 4–28 on identical five-fold splits. The candidate minimum near-optimal k satisfies AUROC and AUPRC drops ≤0.01 and Brier increase ≤0.01 relative to that model's highest-AUPRC row; an exact AUPRC reference tie uses larger k. The entire reduction table is retained.

Champion hierarchy compares reduced candidates: absolute ΔAUPRC ≥0.01 takes priority; otherwise compare AUROC/Brier, then parsimony when all three differences are <0.01. Unresolved tradeoffs do not receive an arbitrary winner. The legally corrected development cohort yielded a single-split minimum k of eight. Paired repeated CV across seeds 42–51 supported retaining ALP; the final nine-variable panel was approved separately. This replay does not reopen that decision based on synthetic or external results.

Final XGBoost: 300 trees, learning_rate 0.03, max_depth 2, min_child_weight 1, subsample 0.8, colsample_bytree 0.8, reg_lambda 1, scale_pos_weight 1, tree_method hist, binary:logistic, eval_metric logloss, seed 42, n_jobs 2. All additional explicitly recorded settings are in frozen_analysis.json. The clinical core uses L2 logistic regression, C=1, liblinear, max_iter=10000, training-only median/scaling.

## Performance, calibration and utility

Performance uses pooled participant-level OOF for development and unchanged locked probabilities for external. AUROC, average precision and Brier receive 2,000 ordinary paired patient Bootstrap draws, seeds 42/43 for development/external. Both models use identical draws; one-class draws are rejected and redrawn. Intervals are percentile 2.5th/97.5th quantiles with linear interpolation. Model selection/fitting is not repeated inside these intervals. Paired differences are XGBoost minus clinical core.

Raw calibration fits `outcome ~ intercept + logit(p)` by binomial GLM; CITL is a separate intercept-only fit with logit(p) as offset. Raw diagnostic probabilities are clipped only for the logit calculation to [1e−8,1−1e−8]. Brier, mean predicted probability and prevalence use original probabilities.

Local recalibration uses five shuffled stratified folds, seed 42. Each training partition estimates alpha/beta on original logit probabilities; it predicts only its held-out partition. Each participant receives exactly one held-out recalibrated value. Positive slopes preserve order **within a fold**, but different fold-specific mappings can change **pooled order**. Therefore raw ROC/PR, DCA, targeting and phenotype groups remain unchanged; no invariant pooled AUROC claim is made. Main calibration uses 10 qcut groups, unconnected points, a diagnostic logistic calibration curve, an ideal line and axes 0–1. The fitted display curve does not overwrite held-out probabilities. Raw calibration remains Figure S3 and supplementary metrics.

DCA: net benefit = TP/n − FP/n × t/(1−t), selected if p ≥ t. Save t = 0 through 0.999 in increments of 0.001; t=1 is undefined. Display the approved main range x=0–0.12, y=−0.02–0.075, with XGBoost, treat-all and treat-none. This is a display range, not an optimized decision cutoff.

Testing budgets: floor(n×budget/100) for 10–50%. Report detected cases, recall, yield, tests per D2 detected and enrichment against random testing. Random expected cases = tested×prevalence; random recall = tested/n; random yield = prevalence. For uncertainty, resample patients 2,000 times within cohort, rerank each draw, and use the same draw for all budgets; seeds 42/43. No-event draws are redrawn. Zero detected-case/infinite testing-efficiency draws trigger review rather than selective deletion.

Figure 6 reuses the exact top-20% membership, with four observed-D2/model-selection groups. Wilson intervals describe group prevalence. The primary comparison is D2-negative/model-positive minus D2-negative/model-negative, with Newcombe absolute risk-difference intervals. Log risk-ratio intervals are secondary and undefined for zero cells; no unapproved continuity correction is introduced. Continuous contrasts use 2,000 independent within-group resamples, fixed membership, median differences and percentile intervals. All five continuous renal outcomes share each draw. D3 denominators are restricted within each group. Related phenotypes share cystatin C measurements and do not constitute independent disease confirmation.

## Figure identities

| Figure | Frozen structure |
|---|---|
| 2 | A seven-model ROC; B/C XGBoost/RF Full SHAP; D two reduction AUROC curves; E XGBoost AUROC/sensitivity/specificity/F1; F nine-variable pooled OOF ROC with individual fold curves |
| 3 | A/B raw external ROC/PR; C raw DCA; D cross-fitted local calibration |
| 4 | A nine-feature SHAP importance; B same-order beeswarm; C nine observed-value dependence plots (3×3) |
| 5 | A capture; B yield with prevalence references; C tests per D2 detected; D enrichment |
| 6 | A four-group composition; B D1; C eligible-denominator D3; D renal profiles with shared across-cohort row z-score colors |
| S1 | 39 pre-redundancy pairwise-complete Spearman heatmap |
| S2 | Seven Full-model PR curves |
| S3 | Raw independent external calibration |

Figure 2E uses each feature count's own development-OOF Youden threshold for descriptive sensitivity/specificity/F1 only. It does not set a final clinical cutoff. Figure 4 dependence plots show observed feature values only; beeswarm uses the actual imputed model input. Continuous dependence LOWESS uses frac=0.35, it=3, displayed from the observed fifth through 95th percentile while retaining every scatter point. Sex is binary with display-only seed-42 jitter. A SHAP zero crossing is not a clinical threshold.

All figures preserve approved panel structures, colors and scientific inputs; fonts may fall back on systems without Times New Roman. The code does not bundle the original rendered images. Full source outputs remain private local files.


## Validation details

# Code-package verification

## Completed checks

- Nine deterministic tests passed: frozen pool sizes/order, equation reference points and phenotype definitions, training-fold-only imputation/scaling, DCA arithmetic, testing budgets, near-optimal rule, cross-fitted recalibration, unit/A:G harmonization, specimen-window/index selection, and repository output isolation (some tests cover multiple items).
- A synthetic integration exercise used 600 invented observations per cohort. It completed seven-model/core CV, all 4–28-feature XGBoost/RF reductions, all ten paired stability repeats, final fitting, locked external prediction, both cohorts' 2,000-draw performance and targeting Bootstraps, cross-fitted recalibration, final SHAP, and four-group phenotype calculations.
- All eight requested figure scripts were exercised on synthetic results. One colorbar API call was corrected, then the figure stage completed. PNG/TIFF dimensions and 300 dpi metadata were checked. These synthetic images and arrays are not distributed.
- Main Table 1/Table S2 DOCX generation and the aggregate S1–S9 source workbook were exercised. The package does not claim a newly typeset, page-reviewed supplementary manuscript; the requested deliverable is source code.
- Classifier settings were extracted through an explicit parameter-field allowlist from frozen fit records, not copied from old exploratory scripts. The final model's nine-feature order, XGBoost settings and core-model settings were cross-checked against the final lock.
- A code-only file allowlist and private-literal scanner are provided and run before handoff. No trained model, source data fixture, patient identity, participant date, per-participant prediction, original figure, source workbook or binary dependency is included.

The tested packaging runtime is Python 3.13.5. The frozen modeling lock records library versions but not its original Python executable version. Exact equality on the original private cohort was **not** re-estimated during packaging. No original model was retrained, retuned or replaced and no clinical external results were used to change settings.

## Expected notices and limits

The pinned scikit-learn runtime emits legacy-API warnings for logistic-regression `penalty`/`l1_ratio` and SVC `probability=True`; LightGBM may report a feature-name warning for imputed arrays. The recorded settings are retained rather than modernized into a different fitting procedure. Training nonconvergence or numerical errors require review.

Synthetic prevalence and clinical distributions intentionally do not match the paper. Some frozen paper-specific plot limits therefore clip synthetic curves (notably DCA and testing yield). This tests the same layout; it is not a clinical result or a proposed replacement figure. Exact private-cohort plotting should be visually reviewed after execution.

The source code preserves the approved analysis, but exact published-number replay additionally depends on the authorized frozen data, source QC, units, row order and tie keys. A generic field mapping cannot infer or recreate private same-specimen/identity linkage. Repository scanning is a practical safeguard, not a mathematical proof that arbitrary future additions are safe to publish.


## Specimen adapter example

Save the following as a private YAML configuration outside the repository.

```yaml
# OPTIONAL INPUT ADAPTER FRAGMENT, not a second analysis protocol.
# Copy these entries into the respective cohort of your private config.
input_mode: specimen
path: ./chemistry_specimens.parquet
columns: {}
factors: {}
tie_key_column: tie_key
linkage:
  identity_resolution_verified: false
  same_specimen_verified: false
  patient_column: private_link_key
  date_column: collection_date
  specimen_column: private_specimen_key
  diabetes_column: diabetes_eligible
  frozen_order_column: frozen_order
  period_start: null  # custodian supplies the approved study window locally
  period_end_exclusive: null
  cbc:
    path: ./cbc_panels.parquet
    patient_column: private_link_key
    date_column: collection_date
    specimen_column: private_specimen_key
    invalid_record_column: invalid_panel
  hba1c:
    path: ./hba1c_panels.parquet
    patient_column: private_link_key
    date_column: collection_date
    specimen_column: private_specimen_key

```

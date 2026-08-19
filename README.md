# Text-Assisted Temporal Action Segmentation

This repository studies whether text can improve **Temporal Action Segmentation
(TAS)** during training while preserving a deployable inference pipeline.

The experiments cover two datasets:

- **Breakfast** — proof-of-concept and controlled teacher–student experiments;
- **Assembly101** — the main architecture, text-alignment, knowledge
  distillation, robustness, and inference-time pseudo-text study.

The project evaluates four ways of exploiting text:

1. a privileged teacher that receives video and ground-truth action text;
2. logit knowledge distillation (KD) into a video-only student;
3. direct alignment of student features with fixed or learnable CLIP
   action-text prototypes;
4. Qwen3.5-generated pseudo-text at inference, including direct teacher input
   and guarded logit fusion.

The selected deployable system does not receive text and does not call the
teacher or a VLM at inference:

```text
training:  video + training-time text supervision
inference: video → visual baseline and KD student → weighted log-probability fusion
```

## Main conclusion

Training-time text supervision provides a **small, metric-dependent signal**
rather than a universal improvement.

Across three Assembly101 seeds, the learnable-text student reaches test
F1@50 `17.53 ± 0.22`, compared with `17.27 ± 0.64` for CE-only (`+0.26`).
The KD student reaches `17.12 ± 0.29` (`−0.15` versus CE-only), so the original
single-seed KD F1@50 gain is not stable across seeds. Text and KD nevertheless
improve test F1@25 over paired CE-only runs on all three seeds.

Inference-time Qwen3.5 pseudo-text does not improve the deployable visual
baseline. Hard pseudo-labels and caption embeddings perform substantially below
the visual-only model, while two guarded pseudo-text fusion searches improve
calibration but fail on the held-out half of validation.

The final locked ensemble combines the notebook-16 visual baseline with the
seed-7 KD student. On full validation it improves F1@50 from `18.97` to `19.49`
(`+0.52`), including a `+0.47` gain on the held-out validation half. This is a
**validation result**; the locked ensemble has not been evaluated on test.

---

## Final Assembly101 configuration

| Component | Selected value |
|---|---|
| Temporal model | MS-TCN |
| Video representation | CLIP ViT-B/16 |
| Student inference input | Video only |
| KD coefficient | `λKD = 0.2` |
| Distillation temperature | `T = 4` |
| Distillation stages | All MS-TCN stages |
| Ensemble members | Notebook-16 visual baseline + seed-7 KD student |
| Ensemble rule | Weighted mean log-probability |
| Ensemble weights | `0.25` visual baseline, `0.75` KD student |
| Full-validation F1@50 | **19.49** |
| Gain over visual baseline | **+0.52 F1@50** |
| Held-out validation gain | **+0.47 F1@50** |
| Ensemble test status | Not evaluated |

---

## Repository structure

```text
notebooks/
  00–11  Breakfast data preparation, visual baselines, teachers, and KD
  12–15  Assembly101 download, annotation conversion, and feature extraction
  16–18  Assembly101 architecture/representation and privileged-teacher ablations
  19–21  CE, KD, learnable text embeddings, and staged hyperparameter search
  22     Core result aggregation, protocol audit, and presentation artifacts
  23     Three-seed CE/text/KD repetition
  24–27  Qwen3.5 pseudo-text extraction, validation, caption rescue, and integrity audit
  28–29  Guarded pseudo-text fusion and transition-aware decoding
  30     Guarded visual-baseline/KD-student ensemble search
```

Large datasets, raw videos, checkpoints, extracted features, predictions, VLM
outputs, and model weights are intentionally not stored in Git.

---

## Notebook pipeline

### Breakfast: notebooks 00–11

| Notebook | Purpose |
|---|---|
| `00_download_breakfast_raw_videos_hf.ipynb` | Downloads the raw Breakfast videos. |
| `01_prepare_breakfast_mstcn_data.ipynb` | Prepares Breakfast data in MS-TCN format. |
| `02_extract_clip_text_embeddings_breakfast.ipynb` | Extracts CLIP embeddings for Breakfast action labels. |
| `03_internal_teacher_student_kd_i3d_clip.ipynb` | Early I3D + CLIP teacher–student proof-of-concept. |
| `04_concat_text_teacher_kd_i3d_clip.ipynb` | Tests a concatenation-based text teacher with I3D features. |
| `05_official_mstcn_i3d_baseline_breakfast.ipynb` | Trains the official-style visual-only MS-TCN baseline. |
| `06_official_style_text_teacher_kd_i3d_clip.ipynb` | Runs official-style text-teacher KD on I3D features. |
| `07_procedurevrl_extract_features_breakfast.ipynb` | Extracts coarse ProcedureVRL output-level features. |
| `08_mstcn_procedurevrl_coarse_baseline_breakfast.ipynb` | Trains a baseline on coarse ProcedureVRL features. |
| `09_procedurevrl_hidden_embeddings_breakfast_COLAB.ipynb` | Extracts 512-dimensional ProcedureVRL hidden embeddings. |
| `10_mstcn_procedurevrl_hidden_baseline_breakfast.ipynb` | Trains the visual-only hidden-feature baseline. |
| `11_procedurevrl_hidden_privileged_text_teacher_kd_breakfast.ipynb` | Trains the privileged teacher, controlled CE student, and video-only KD student. |

### Assembly101: notebooks 12–30

| Notebook | Purpose |
|---|---|
| `12_assembly101_hf_download_and_verify.ipynb` | Downloads and verifies Assembly101 resources. |
| `13_assembly101_annotations_to_mstcn.ipynb` | Converts Assembly101 annotations and official splits to MS-TCN format. |
| `14_assembly101_v1_download_and_smoke_video_crops.ipynb` | Verifies the video source and tests video-crop decoding. |
| `15_assembly101_streaming_procedurevrl_clip_feature_extraction.ipynb` | Extracts aligned ProcedureVRL and CLIP video features. |
| `16_assembly101_visual_only_mstcn_baselines.ipynb` | Compares ProcedureVRL and CLIP using visual-only MS-TCN. |
| `17_assembly101_visual_only_ltcontext_baselines.ipynb` | Repeats the representation comparison with LTContext. |
| `18_assembly101_text_assisted_concat_teacher_baselines.ipynb` | Trains privileged video + ground-truth-text teachers. |
| `19_assembly101_video_only_knowledge_distillation_students.ipynb` | Trains controlled CE-only and initial video-only KD students. |
| `20_assembly101_learnable_text_embedding_students_COLAB.ipynb` | Aligns student features with fixed and learnable CLIP text prototypes. |
| `21_assembly101_text_student_kd_hyperparameter_search_COLAB.ipynb` | Searches KD coefficient, temperature, and stage policy; evaluates the combined text + KD objective. |
| `22_final_ablation_analysis_and_presentation_artifacts_COLAB.ipynb` | Aggregates the core results and creates tables, figures, conclusions, and presentation artifacts. |
| `23_assembly101_three_seed_repetition_COLAB.ipynb` | Repeats CE, text alignment, and KD with seeds 7, 17, and 42 and reports paired/aggregate results. |
| `24_assembly101_qwen35_pseudotext_extraction_COLAB.ipynb` | Uses Qwen3.5-2B to generate 16-step closed-set pseudo-text from validation video frames. |
| `25_assembly101_qwen_pseudotext_teacher_validation_COLAB.ipynb` | Evaluates hard Qwen pseudo-text with the frozen privileged teacher. |
| `26_assembly101_qwen_caption_embedding_ablation_COLAB.ipynb` | Tests direct captions, prompted captions, nearest prototypes, and soft caption mixtures. |
| `27_assembly101_qwen_visual_input_integrity_audit_COLAB.ipynb` | Audits frame ordering, preprocessing, visual sensitivity, and pipeline integrity. |
| `28_assembly101_residual_pseudotext_logit_fusion_COLAB.ipynb` | Tests confidence-gated residual pseudo-text fusion with a calibration/holdout protocol. |
| `29_assembly101_transition_aware_pseudotext_decoding_COLAB.ipynb` | Tests transition-aware decoding combined with pseudo-text residuals. |
| `30_assembly101_three_seed_strategy_ensemble_COLAB.ipynb` | Audits nine student checkpoints and selects a guarded visual/KD ensemble on calibration only. |

---

## Experimental design

### Video representations

| Representation | Shape per sequence | Description |
|---|---:|---|
| ProcedureVRL hidden | `[512, 16]` | Hidden video representation extracted from ProcedureVRL. |
| CLIP ViT-B/16 | `[512, 16]` | CLIP visual features aligned with the CLIP text space. |

### Temporal architectures

- **MS-TCN** — multi-stage temporal convolutional network;
- **LTContext** — temporal model using local/global context operations.

### Training-time text strategies

#### Privileged teacher

The teacher receives video features and ground-truth action-text embeddings at
each temporal position. It is an oracle upper bound and is not deployable.

#### Knowledge distillation

The student receives video only. During training, its logits are matched to the
privileged teacher using temperature-scaled KL divergence.

#### Student-side text alignment

MS-TCN hidden features are projected into the 512-dimensional CLIP text space.
The projected features are trained against action-class text prototypes using a
frame-to-prototype contrastive cross-entropy objective.

Two prototype variants are evaluated:

- **fixed** — frozen normalized CLIP class embeddings;
- **learnable** — normalized CLIP embeddings with a regularized learnable class
  residual.

The text-alignment head is not called during final inference.

### Inference-time pseudo-text

Qwen3.5-2B receives 16 sampled video frames and produces closed-set action
labels and captions. The study tests hard pseudo-label embeddings, direct CLIP
caption embeddings, nearest/soft action prototypes, residual logit fusion, and
transition-aware decoding.

All pseudo-text decisions are made on validation. Calibration-only selection is
followed by a one-shot evaluation on a deterministic held-out half of
validation; failed configurations are not promoted to test.

---

## Assembly101 results

### Architecture × representation ablation

Visual-only test results:

| Architecture | Representation | Accuracy | Edit | F1@10 | F1@25 | F1@50 |
|---|---|---:|---:|---:|---:|---:|
| **MS-TCN** | **CLIP** | 25.86 | **25.01** | <u>28.00</u> | **23.82** | **17.58** |
| MS-TCN | ProcedureVRL hidden | <u>26.16</u> | <u>24.74</u> | **28.36** | <u>23.79</u> | <u>15.72</u> |
| LTContext | CLIP | **27.10** | 22.70 | 25.62 | 21.96 | 13.39 |
| LTContext | ProcedureVRL hidden | **27.10** | 22.80 | 25.11 | 20.05 | 11.92 |

MS-TCN + CLIP is used for the student experiments because it provides the best
visual-only test F1@50 among the evaluated architecture/representation pairs.

### Single-seed strategy comparison

Seed-7 test results are shown below. Bold and underlined values mark the best
and second-best **deployable** values; the oracle is excluded from this ranking.

| Strategy | Configuration | Training text | Inference text | Deployable | Accuracy | Edit | F1@10 | F1@25 | F1@50 |
|---|---|:---:|:---:|:---:|---:|---:|---:|---:|---:|
| Visual baseline | MS-TCN + CLIP | No | No | Yes | 25.86 | <u>25.01</u> | 28.00 | <u>23.82</u> | <u>17.58</u> |
| Controlled CE-only | Same student pipeline, `λKD=0` | No | No | Yes | 25.94 | 24.75 | 27.58 | 23.14 | 16.53 |
| Original KD | `λKD=0.1`, `T=4`, all stages | Yes | No | Yes | **26.38** | 24.64 | 28.40 | 23.75 | **17.73** |
| Text-only | Learnable prototypes, `β=0.01` | Yes | No | Yes | <u>26.27</u> | 24.78 | <u>28.58</u> | 23.76 | 17.56 |
| Expanded KD | `λKD=0.2`, `T=4`, all stages | Yes | No | Yes | 25.67 | **25.10** | 28.22 | **24.11** | 17.36 |
| Text + KD | `β=0.01`, `λKD=0.2`, `T=4`, all stages | Yes | No | Yes | 26.24 | 24.85 | **28.73** | 23.67 | 17.32 |
| Oracle teacher | Video + ground-truth text | Yes | Yes | No | 44.27 | 37.39 | 41.65 | 40.94 | 36.81 |

The original single-seed KD comparison is:

| KD student − CE-only student | Accuracy | Edit | F1@10 | F1@25 | F1@50 |
|---|---:|---:|---:|---:|---:|
| Difference | −0.27 | +0.35 | +0.64 | +0.97 | **+0.83** |

This F1@50 improvement is not treated as a robust multi-seed conclusion.

### Three-seed robustness

Test mean ± sample standard deviation over seeds `7`, `17`, and `42`:

| Strategy | Accuracy | Edit | F1@10 | F1@25 | F1@50 |
|---|---:|---:|---:|---:|---:|
| CE-only | <u>26.26 ± 0.32</u> | <u>25.21 ± 0.41</u> | 28.05 ± 0.41 | 23.88 ± 0.78 | <u>17.27 ± 0.64</u> |
| Text alignment | **26.32 ± 0.45** | **25.23 ± 0.39** | **28.45 ± 0.11** | <u>24.25 ± 0.61</u> | **17.53 ± 0.22** |
| KD | 26.12 ± 0.57 | 25.05 ± 0.17 | <u>28.41 ± 0.27</u> | **24.42 ± 0.70** | 17.12 ± 0.29 |

Paired against CE-only, text alignment improves mean F1@50 by `+0.26`, while
KD changes it by `−0.15`. For F1@25, text alignment and KD improve all `3/3`
paired seeds, with mean gains of `+0.37` and `+0.54`, respectively. Three seeds
are sufficient for a robustness check but not for a formal significance claim.

### Validation-selected KD hyperparameters

| Ablation | Candidates | Selected value | Selected validation F1@50 |
|---|---|---:|---:|
| KD coefficient | `0.01, 0.05, 0.1, 0.2, 0.5, 1.0` | **0.2** | **19.27** |
| Temperature | `1, 2, 4, 8` | **4** | **19.27** |
| Stage policy | `all, final` | **all** | **19.27** |

The configuration-selection score is:

```text
validation F1@25 + 0.01 × validation Edit
```

F1@50 is the primary reported overlap metric, but the historical search itself
used the score above. No test metric was used during hyperparameter selection.

### Fixed vs learnable text prototypes

| Prototype mode | `β` | Validation F1@50 | Selection score |
|---|---:|---:|---:|
| Fixed | 0.01 | 19.04 | 25.946 |
| Fixed | 0.05 | 19.01 | 25.134 |
| **Fixed** | **0.10** | **19.16** | <u>26.059</u> |
| Learnable | 0.01 | 19.03 | **26.063** |
| Learnable | 0.05 | 18.75 | 25.512 |
| Learnable | 0.10 | <u>19.13</u> | 25.550 |

The learnable `β=0.01` strategy is selected by the historical F1@25/Edit score,
whereas fixed `β=0.10` has the best validation F1@50. The results therefore do
not establish a decisive benefit from prototype learnability itself.

### Inference-time Qwen3.5 pseudo-text

Frozen-teacher validation results:

| Text input | Accuracy | Edit | F1@10 | F1@25 | F1@50 |
|---|---:|---:|---:|---:|---:|
| Oracle ground-truth text | **45.21** | **34.92** | **39.73** | **38.25** | **34.14** |
| Visual-only MS-TCN reference | <u>29.95</u> | <u>26.01</u> | <u>29.23</u> | <u>25.90</u> | <u>18.97</u> |
| Direct Qwen caption embedding | 12.08 | 14.68 | 14.73 | 12.52 | 7.47 |
| Zero text | 10.73 | 14.49 | 12.80 | 10.56 | 7.41 |
| Hard Qwen pseudo-label | 6.09 | 7.87 | 6.01 | 4.19 | 2.51 |

Hard pseudo-label accuracy is `4.58%`. Qwen predicts only `32` unique classes
for `154` target classes, and its most frequent class covers `69.58%` of
temporal positions. Notebook 27 finds no frame-order, preprocessing, or visual
input integrity error. The result is most consistent with weak/collapsed
pseudo-text and a teacher that is not robust to noisy inference-time text.

Guarded calibration/holdout decisions:

| Method | Calibration ΔF1@50 | Holdout ΔF1@50 | Full-validation ΔF1@50 | Promote |
|---|---:|---:|---:|:---:|
| Residual pseudo-text fusion | **+1.78** | −0.44 | +0.69 | No |
| Transition + pseudo-text | **+2.13** | −0.59 | +0.80 | No |
| Visual baseline + KD student ensemble | +0.62 | **+0.47** | **+0.52** | **Yes** |

The larger full-validation pseudo-text gains are not promoted because they
reverse on holdout. The visual/KD ensemble is the only guarded follow-up that
improves calibration, holdout, and full validation.

### Locked ensemble validation result

Bold and underlined values mark the best and second-best result.

| Method | Accuracy | Edit | F1@10 | F1@25 | F1@50 |
|---|---:|---:|---:|---:|---:|
| Notebook-16 visual baseline | <u>29.95</u> | <u>26.01</u> | <u>29.23</u> | <u>25.90</u> | <u>18.97</u> |
| **Visual baseline + KD student ensemble** | **30.31** | **26.35** | **29.87** | **26.20** | **19.49** |

The selected score is:

```text
0.25 × log p(notebook-16 visual baseline)
+ 0.75 × log p(seed-7 KD student)
```

The ensemble configuration is selected on 60 calibration sequences and then
evaluated once on 60 held-out validation sequences. It gains `+0.62`, `+0.47`,
and `+0.52` F1@50 on calibration, holdout, and full validation, respectively.
The test split is not accessed in notebook 30.

---

## Breakfast results

All results below use split 1. Bold and underlined values rank deployable
methods only; the privileged oracle is excluded.

| Strategy | Inference input | Accuracy | Edit | F1@10 | F1@25 | F1@50 |
|---|---|---:|---:|---:|---:|---:|
| Historical ProcedureVRL hidden baseline | Video only | **58.04** | <u>55.57</u> | **58.58** | **57.73** | <u>45.76</u> |
| Controlled CE-only student | Video only | 57.42 | 55.61 | 55.76 | 54.29 | 45.34 |
| Video-only KD student | Video only | <u>57.99</u> | **56.46** | <u>56.67</u> | <u>55.42</u> | **46.19** |
| Privileged text teacher | Video + ground-truth text | 98.26 | 96.26 | 96.90 | 96.90 | 96.71 |

The controlled KD student improves over the same-notebook CE-only student by
`+0.85` F1@50. The historical visual baseline remains stronger than the KD
student on F1@25 (`57.73` versus `55.42`), so it is reported as context rather
than evidence that KD is the best model on every metric.

---

---

## Reproducing the experiments

1. Open the notebooks in Google Colab and mount Google Drive.
2. Run the desired pipeline sequentially:
   - Breakfast core pipeline: `00 → 11`;
   - Assembly101 core pipeline: `12 → 23`;
   - Qwen pseudo-text follow-up: `24 → 29`;
   - locked ensemble validation: `30` after `16`, `19–21`, `23`, and `28`.
3. Run notebook `22` after notebooks `16–21` to recreate the core reporting
   artifacts. Notebooks `23–30` additionally write their own tables, figures,
   audits, and `final_summary.json` files under their run directories.

Full training and Qwen notebooks are intended for a GPU runtime. Completed
runs, checkpoints, logits, captions, and summaries are stored on Google Drive
so long stages can be resumed without committing large artifacts to Git.

---

## Reporting artifacts

The reporting notebooks produce:

- completion, provenance, and reconstruction audits;
- architecture × representation and strategy tables;
- hyperparameter and learnable-prototype ablations;
- three-seed means, sample standard deviations, and paired deltas;
- Qwen pseudo-text quality and integrity diagnostics;
- guarded calibration/holdout decisions;
- presentation-ready F1@50 tables and figures;
- machine-readable final summaries.

---

## Known limitations

1. Three student seeds provide a robustness check but not a formal statistical
   significance test.
2. The final ensemble result is validation-only. Its base checkpoints were
   selected using full validation, so the 50% holdout is independent of
   ensemble composition selection but not of historical checkpoint selection.
3. The selected ensemble evaluates two temporal models and therefore costs
   more at inference than a single student.
4. The privileged teacher receives ground-truth action text and is an oracle
   upper bound, not a deployable model.
5. ProcedureVRL and CLIP sequences contain only 16 temporal positions, which
   may under-resolve short actions.
6. The Qwen conclusion is specific to Qwen3.5-2B, the locked prompt, 16 sampled
   frames, and a teacher trained with clean ground-truth text. It does not rule
   out stronger VLMs or noise-robust teacher training.
7. The claims concern controlled improvements and failure analyses, not
   state-of-the-art Assembly101 performance.

---

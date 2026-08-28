# Text-Assisted Temporal Action Segmentation

This repository studies whether text can improve **Temporal Action Segmentation
(TAS)** during training while preserving a deployable inference pipeline.

The experiments cover two datasets:

- **Breakfast** — proof-of-concept, controlled teacher–student experiments, and
  VLM capacity diagnostics with 48 action classes;
- **Assembly101** — the main architecture, text-alignment, knowledge
  distillation, robustness, and inference-time pseudo-text study with 202
  catalog actions.

The project evaluates the following ways of exploiting text:

1. a privileged teacher that receives video and ground-truth action text;
2. logit knowledge distillation (KD) into a video-only student;
3. direct alignment of student features with fixed or learnable CLIP
   action-text prototypes;
4. Qwen3.5-generated pseudo-text at inference, including direct teacher input
   and guarded logit fusion;
5. closed-set Qwen alignment in which the complete action catalog is supplied
   and the model must return an exact action ID for every temporal step;
6. a controlled VLM capacity comparison between Qwen3.5-2B and
   Qwen3.5-35B-A3B on the 48-class Breakfast dataset.

The selected deployable system does not receive text and does not call the
teacher or a VLM at inference:

```text
training:  video + training-time text supervision
inference: video → visual baseline and KD student
                 → weighted log-probability fusion
```

## Main conclusion

Training-time text supervision provides a **small, metric-dependent signal**
rather than a universal improvement.

Across three Assembly101 seeds, the learnable-text student reaches test F1@50
`17.53 ± 0.22`, compared with `17.27 ± 0.64` for CE-only (`+0.26`). The KD
student reaches `17.12 ± 0.29` (`−0.15` versus CE-only), so the original
single-seed KD F1@50 gain is not stable across seeds. Text alignment and KD
nevertheless improve test F1@25 over their paired CE-only runs on all three
seeds.

Inference-time Qwen pseudo-text does not improve the deployable visual model.

On Assembly101, providing Qwen3.5-2B with the complete 202-action catalog
produces validation F1@50 `2.94` with direct top-1 predictions and `3.46` with
fixed temporal smoothing, compared with `18.97` for the visual-only reference.
Only `9` of the `154` validation target classes are predicted, and one class
occupies `84.58%` of all temporal positions.

Reducing the catalog to the 48 Breakfast actions substantially reduces this
collapse. Qwen3.5-2B reaches top-1 accuracy `7.12%` against a `2.08%` random
level, predicts `37` classes, and reduces the dominant-class share to `20.11%`.
However, its direct pseudo-text still reaches only F1@50 `7.43`, below zero
text (`9.22`) and far below the controlled visual-only student (`45.34`).

The final capacity experiment evaluates Qwen3.5-35B-A3B in GPTQ Int4 on an
A100 40 GB. The larger model increases Breakfast pseudo-label accuracy from
`7.12%` to `13.76%`, predicts `43` classes, and reduces the dominant-class share
to `18.18%`. This supports the hypothesis that 2B model capacity was one
important limitation.

Nevertheless, the stronger pseudo-labels still do not improve TAS. The frozen
teacher reaches F1@50 `8.71` with direct Qwen3.5-35B-A3B text and `8.52` with
temporal smoothing, compared with `9.39` for zero text and `45.24` for the
controlled visual-only student. Ground-truth text reaches F1@50 `96.71`.

The combined evidence therefore suggests that:

- VLM capacity materially affects closed-set action recognition;
- the large Assembly101 catalog contributes to prediction collapse;
- neither catalog size nor model capacity alone explains the downstream
  failure;
- the privileged teacher is highly effective with correct text but is not
  robust to noisy inference-time pseudo-text;
- generated action labels remain insufficiently accurate for improving
  segmentation.

The final locked deployable ensemble combines the notebook-16 visual baseline
with the seed-7 KD student. On full Assembly101 validation it improves F1@50
from `18.97` to `19.49` (`+0.52`), including a `+0.47` gain on the held-out
validation half. This is a **validation result**; the locked ensemble has not
been evaluated on the test split.

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

---

## Repository structure

```text
notebooks/
  00–11  Breakfast data preparation, visual baselines, teachers, and KD
  12–15  Assembly101 download, annotation conversion, and feature extraction
  16–18  Assembly101 architecture/representation and privileged-teacher ablations
  19–21  CE, KD, learnable text embeddings, and hyperparameter search
  22     Core result aggregation, protocol audit, and presentation artifacts
  23     Three-seed CE/text/KD repetition
  24–27  Qwen3.5 pseudo-text extraction, validation, caption rescue, and integrity audit
  28–29  Guarded pseudo-text fusion and transition-aware decoding
  30     Guarded visual-baseline/KD-student ensemble search
  31     Assembly101 full-catalog Qwen3.5-2B alignment
  32     Breakfast 48-class Qwen3.5-2B closed-set follow-up
  33     Breakfast Qwen3.5-35B-A3B GPTQ capacity experiment on A100
```

---

## Notebook pipeline

### Breakfast: notebooks 00–11 and 32–33

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
| `32_breakfast_qwen35_2b_catalog_top1_alignment_COLAB.ipynb` | Gives Qwen3.5-2B the complete 48-action catalog, freezes predictions before loading labels, and evaluates direct and temporally smoothed pseudo-text with the notebook-11 checkpoints. |
| `33_breakfast_qwen35_35b_a3b_gptq_catalog_capacity_A100_COLAB.ipynb` | Repeats the fixed Breakfast closed-set protocol with Qwen3.5-35B-A3B GPTQ Int4 on an A100 and evaluates whether increased VLM capacity improves pseudo-labels and downstream TAS. |

### Assembly101: notebooks 12–31

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
| `23_assembly101_three_seed_repetition_COLAB.ipynb` | Repeats CE, text alignment, and KD with seeds 7, 17, and 42 and reports paired and aggregate results. |
| `24_assembly101_qwen35_pseudotext_extraction_COLAB.ipynb` | Uses Qwen3.5-2B to generate 16-step closed-set pseudo-text from validation video frames. |
| `25_assembly101_qwen_pseudotext_teacher_validation_COLAB.ipynb` | Evaluates hard Qwen pseudo-text with the frozen privileged teacher. |
| `26_assembly101_qwen_caption_embedding_ablation_COLAB.ipynb` | Tests direct captions, prompted captions, nearest prototypes, and soft caption mixtures. |
| `27_assembly101_qwen_visual_input_integrity_audit_COLAB.ipynb` | Audits frame ordering, preprocessing, visual sensitivity, and pipeline integrity. |
| `28_assembly101_residual_pseudotext_logit_fusion_COLAB.ipynb` | Tests confidence-gated residual pseudo-text fusion with a calibration/holdout protocol. |
| `29_assembly101_transition_aware_pseudotext_decoding_COLAB.ipynb` | Tests transition-aware decoding combined with pseudo-text residuals. |
| `30_assembly101_three_seed_strategy_ensemble_COLAB.ipynb` | Audits nine student checkpoints and selects a guarded visual/KD ensemble on calibration only. |
| `31_assembly101_qwen_catalog_top1_alignment_COLAB.ipynb` | Gives Qwen3.5-2B the complete 202-action catalog, requires exact step-wise class IDs, and evaluates a predeclared validation-only promotion gate. |

---

## Experimental design

### Video representations

| Representation | Shape per sequence | Description |
|---|---:|---|
| ProcedureVRL hidden | `[512, 16]` | Hidden video representation extracted from ProcedureVRL. |
| CLIP ViT-B/16 | `[512, 16]` | CLIP visual features aligned with the CLIP text space. |

### Temporal architectures

- **MS-TCN** — multi-stage temporal convolutional network;
- **LTContext** — temporal model using local and global context operations.

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

Qwen receives uniformly sampled video frames and the complete action catalog.
It must return one exact catalog action ID for every temporal position.

The study evaluates:

- hard pseudo-label embeddings;
- direct CLIP caption embeddings;
- nearest and soft action prototypes;
- residual logit fusion;
- transition-aware decoding;
- direct catalog top-1 predictions;
- fixed temporal smoothing;
- shuffled pseudo-text controls;
- Qwen3.5-2B versus Qwen3.5-35B-A3B capacity.

On Assembly101, all pseudo-text model-selection decisions are made on
validation. Calibration-only selection is followed by one-shot evaluation on a
deterministic held-out validation half. Failed configurations are not promoted
to the test split.

The Breakfast experiments use all `252` split-1 test videos and the complete
48-action catalog. Each video is represented by `16` uniformly sampled
temporal positions, with four ordered frames supplied for each four-position
generation block.

All Qwen predictions are written to persistent storage and frozen before
ground-truth labels are loaded. Evaluation then uses the frozen notebook-11
teacher and visual checkpoints.

Because the notebook-11 Breakfast checkpoints were previously selected using
the same split-1 test set, notebooks 32 and 33 are explicitly
**exploratory diagnostics**. They are not used for additional tuning or a
formal generalization claim.

### Qwen3.5-35B-A3B capacity configuration

| Component | Value |
|---|---|
| Model | `Qwen/Qwen3.5-35B-A3B-GPTQ-Int4` |
| Model revision | `3af5ca2` |
| Quantization | GPTQ Int4 |
| GPU | NVIDIA A100-SXM4 40 GB |
| Model footprint | 20.86 GiB |
| CUDA allocated after loading | 20.88 GiB |
| Quantized implementation | Marlin |
| Quantized modules | 30,720 |
| Breakfast videos | 252/252 completed |
| Temporal predictions | 4,032 |
| Failed or missing videos | 0 |

The model and protocol gate completed successfully before the full run. The
complete Qwen generation stage finished before any evaluation labels were
loaded.

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
visual-only test F1@50 among the evaluated architecture and representation
pairs.

### Single-seed strategy comparison

Seed-7 test results are shown below. Bold and underlined values mark the best
and second-best **deployable** values; the oracle is excluded from this
ranking.

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
paired seeds, with mean gains of `+0.37` and `+0.54`, respectively.

Three seeds are sufficient for a robustness check but not for a formal
significance claim.

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
| Qwen catalog top-1 + fixed temporal smoothing | 5.73 | 10.27 | 9.28 | 6.56 | 3.46 |
| Qwen direct catalog top-1 | 5.62 | 10.15 | 8.45 | 5.89 | 2.94 |
| Shuffled direct catalog top-1 control | 5.57 | 10.04 | 8.41 | 5.37 | 2.80 |
| Notebook-24 hard Qwen pseudo-label | 6.09 | 7.87 | 6.01 | 4.19 | 2.51 |

Hard pseudo-label accuracy is `4.58%`. Qwen predicts only `32` unique classes
for `154` target classes, and its most frequent class covers `69.58%` of
temporal positions.

Notebook 27 finds no frame-order, preprocessing, or visual-input integrity
error. The result is most consistent with weak or collapsed pseudo-text and a
teacher that is not robust to noisy inference-time text.

#### Full-action-catalog follow-up

Notebook 31 directly tests the proposal of showing Qwen all possible dataset
actions. Notebook 24 already exposed the same catalog inside a combined
caption-and-class prompt; notebook 31 isolates the hypothesis by removing
free-caption generation and requiring one exact ID from the full 202-class
catalog at each of 16 temporal positions.

Predictions are frozen before labels are loaded, and evaluation is
validation-only.

| Variant | Calibration F1@50 | Holdout F1@50 | Full validation F1@50 | Δ vs visual, full |
|---|---:|---:|---:|---:|
| Visual-only MS-TCN | **16.21** | **21.90** | **18.97** | — |
| Direct catalog top-1 | 3.38 | 2.45 | 2.94 | −16.03 |
| Direct top-1 + fixed temporal smoothing | <u>3.52</u> | <u>3.40</u> | <u>3.46</u> | −15.51 |
| Shuffled direct top-1 control | 2.72 | 2.90 | 2.80 | −16.17 |
| Notebook-24 hard top-1 | 2.22 | 2.81 | 2.51 | −16.46 |

The predeclared primary candidate is **direct catalog top-1**, not the smoothed
diagnostic. It improves over the earlier hard pseudo-text by only `+0.43`
F1@50 and remains far below the visual reference.

Its step-wise pseudo-label accuracy is `2.86%`; only `9` of the `154`
validation target classes are predicted, and one class occupies `84.58%` of
positions. The shuffled-control score (`2.80`) is also close to the direct score
(`2.94`).

Therefore, the fixed promotion gate fails, the test split remains untouched,
and this experiment is reported as a negative but informative result.

### Guarded calibration/holdout decisions

| Method | Calibration ΔF1@50 | Holdout ΔF1@50 | Full-validation ΔF1@50 | Promote |
|---|---:|---:|---:|:---:|
| Residual pseudo-text fusion | **+1.78** | −0.44 | +0.69 | No |
| Transition + pseudo-text | **+2.13** | −0.59 | +0.80 | No |
| Visual baseline + KD student ensemble | +0.62 | **+0.47** | **+0.52** | **Yes** |

The larger full-validation pseudo-text gains are not promoted because they
reverse on the held-out validation half. The visual/KD ensemble is the only
guarded follow-up that improves calibration, holdout, and full validation.

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

The Assembly101 test split is not accessed in notebook 30.

---

## Breakfast results

All results below use split 1. Bold and underlined values rank deployable
methods only; the privileged oracle is excluded.

### Core teacher–student results

| Strategy | Inference input | Accuracy | Edit | F1@10 | F1@25 | F1@50 |
|---|---|---:|---:|---:|---:|---:|
| Historical ProcedureVRL hidden baseline | Video only | **58.04** | <u>55.57</u> | **58.58** | **57.73** | <u>45.76</u> |
| Controlled CE-only student | Video only | 57.42 | 55.61 | 55.76 | 54.29 | 45.34 |
| Video-only KD student | Video only | <u>57.99</u> | **56.46** | <u>56.67</u> | <u>55.42</u> | **46.19** |
| Privileged text teacher | Video + ground-truth text | 98.26 | 96.26 | 96.90 | 96.90 | 96.71 |

The controlled KD student improves over the same-notebook CE-only student by
`+0.85` F1@50.

The historical visual baseline remains stronger than the KD student on F1@25
(`57.73` versus `55.42`), so the result is not evidence that KD is the best
model on every metric.

### Breakfast 48-class Qwen3.5-2B follow-up

Notebook 32 tests whether the much smaller Breakfast catalog prevents the
closed-set Qwen collapse observed on Assembly101. All `252/252` split-1 videos
complete, yielding `4,032` frozen temporal predictions.

| Variant | Inference input | Accuracy | Edit | F1@10 | F1@25 | F1@50 |
|---|---|---:|---:|---:|---:|---:|
| Privileged teacher | Video + ground-truth text | 98.26 | 96.26 | 96.90 | 96.90 | 96.71 |
| Controlled visual-only student | Video only | **57.42** | **55.61** | **55.76** | **54.29** | **45.34** |
| Teacher + zero text | Video + zero text | 13.14 | 19.94 | 21.09 | 17.25 | 9.22 |
| Teacher + Qwen temporal | Video + generated text | 13.62 | 21.36 | 18.99 | 14.57 | 7.61 |
| Teacher + Qwen direct top-1 | Video + generated text | 13.00 | 20.72 | 18.29 | 13.79 | 7.43 |
| Teacher + shuffled Qwen control | Video + shuffled generated text | 11.04 | 14.99 | 12.50 | 9.15 | 4.52 |

The generated action IDs are substantially less collapsed than on Assembly101:

| Pseudo-label diagnostic | Assembly101 2B (`202` classes) | Breakfast 2B (`48` classes) |
|---|---:|---:|
| Random top-1 level | 0.50% | 2.08% |
| Qwen top-1 accuracy | 2.86% | **7.12%** |
| Accuracy over random | 5.78× | 3.42× |
| Predicted classes | 9 | **37** |
| Target classes present | 154 | 44 |
| Dominant-class share | 84.58% | **20.11%** |

The cross-dataset comparison is diagnostic rather than a comparison of raw TAS
difficulty and does not causally isolate catalog size because the dataset,
visual appearance, action definitions, and feature pipeline also change.

It is nevertheless consistent with a smaller candidate set producing more
diverse and more accurate closed-set predictions.

Direct Qwen text remains `1.79` F1@50 below zero text and `37.91` below the
controlled visual-only student. Fixed temporal smoothing adds only `+0.18`
F1@50 over direct Qwen text.

Thus, the Assembly101 failure cannot be explained only by the large action
catalog. The generated labels remain too noisy, and the privileged teacher is
not robust to this inference-time text distribution.

### Breakfast Qwen3.5-35B-A3B capacity experiment

Notebook 33 tests Federico's hypothesis that Qwen3.5-2B may be too small for
the closed-set video-action recognition task.

The experiment uses the same Breakfast 48-class catalog, temporal sampling,
prompt structure, frozen teacher, visual checkpoint, and downstream evaluation
protocol. The main changed variable is VLM capacity.

Qwen3.5-35B-A3B GPTQ Int4 successfully fits on an A100 40 GB with a measured
model footprint of `20.86 GiB`. All `252/252` videos and `4,032` temporal
positions complete without missing outputs.

#### Downstream segmentation

| Variant | Inference input | Accuracy | Edit | F1@10 | F1@25 | F1@50 |
|---|---|---:|---:|---:|---:|---:|
| Privileged teacher | Video + ground-truth text | **98.26** | **96.26** | **96.90** | **96.90** | **96.71** |
| Controlled visual-only student | Video only | 57.39 | 55.61 | 55.76 | 54.29 | 45.24 |
| Teacher + zero text | Video + zero text | 13.17 | 19.94 | 21.15 | 17.41 | 9.39 |
| Teacher + Qwen3.5-35B-A3B direct | Video + generated text | 17.21 | 24.02 | 22.76 | 17.42 | 8.71 |
| Teacher + Qwen3.5-35B-A3B temporal | Video + generated text | 17.51 | 24.60 | 23.48 | 17.71 | 8.52 |
| Teacher + shuffled Qwen text | Video + shuffled generated text | 17.14 | 19.26 | 17.76 | 12.76 | 5.69 |

The notebook-33 visual checkpoint reproduction differs from the earlier
notebook-32 reference by only `0.10` F1@50 (`45.24` versus `45.34`). Each table
reports the metrics produced by its own evaluation run; this small difference
does not affect the conclusions.

#### Capacity diagnostics

| Diagnostic | Qwen3.5-2B | Qwen3.5-35B-A3B | Change |
|---|---:|---:|---:|
| Random top-1 level | 2.08% | 2.08% | — |
| Top-1 action accuracy | 7.12% | **13.76%** | **+6.65 pp** |
| Accuracy over random | 3.42× | **6.61×** | +3.19× |
| Predicted classes | 37 | **43** | +6 |
| Target classes present | 44 | 44 | — |
| Predicted target-class coverage | — | **97.73%** | — |
| Dominant-class share | 20.11% | **18.18%** | −1.93 pp |
| Direct Qwen F1@50 | 7.43 | **8.71** | **+1.28** |
| Temporal Qwen F1@50 | 7.61 | **8.52** | +0.91 |
| Zero-text F1@50 | 9.22 | 9.39 | +0.17 |
| Visual-only F1@50 | 45.34 | 45.24 | −0.10 |

Increasing VLM capacity produces a clear improvement in pseudo-label quality:

- top-1 accuracy nearly doubles from `7.12%` to `13.76%`;
- the result is `6.61×` the random level;
- predicted-class coverage increases from `37` to `43` classes;
- the dominant-class share decreases from `20.11%` to `18.18%`;
- direct downstream F1@50 improves by `+1.28`.

However, the primary downstream result remains negative:

```text
Qwen3.5-35B-A3B direct F1@50:  8.71
Qwen3.5-35B-A3B temporal:      8.52
zero text:                     9.39
controlled visual-only:       45.24
oracle ground-truth text:     96.71
```

Direct Qwen text remains `0.68` F1@50 below zero text and `36.53` below the
controlled visual-only student. Temporal smoothing reduces F1@50 by `0.19`
relative to direct Qwen text.

The direct result exceeds the shuffled control by `3.02` F1@50, indicating that
the generated actions contain useful information. That information is still
too noisy or insufficiently aligned with the temporal teacher to improve the
primary F1@50 metric.

The capacity hypothesis is therefore **partially supported**:

- Qwen3.5-2B capacity was a real limitation for action recognition;
- Qwen3.5-35B-A3B produces more accurate and diverse pseudo-labels;
- increased VLM capacity does not solve the downstream segmentation problem;
- model size alone is not sufficient to make inference-time pseudo-text useful.

No further tuning is performed on Breakfast split-1 labels.

---

## Cross-experiment interpretation

The three closed-set catalog experiments provide a controlled progression:

| Dataset and model | Catalog size | Top-1 accuracy | Predicted classes | Dominant share | Direct F1@50 | Visual F1@50 |
|---|---:|---:|---:|---:|---:|---:|
| Assembly101, Qwen3.5-2B | 202 | 2.86% | 9 | 84.58% | 2.94 | 18.97 |
| Breakfast, Qwen3.5-2B | 48 | 7.12% | 37 | 20.11% | 7.43 | 45.34 |
| Breakfast, Qwen3.5-35B-A3B | 48 | **13.76%** | **43** | **18.18%** | **8.71** | 45.24 |

These experiments do not form a complete causal decomposition:

- Assembly101 and Breakfast differ in more than catalog size;
- the Breakfast evaluation uses split-1 checkpoints previously selected on the
  same split;
- Qwen3.5-2B and Qwen3.5-35B-A3B may differ in more than parameter count;
- only one fixed prompt and temporal sampling protocol is tested.

Within those limitations, the evidence is consistent with both catalog size
and VLM capacity affecting pseudo-label quality. Neither improvement is
sufficient for the generated text to outperform zero text or the visual-only
model.

The large oracle gap shows that text itself is not the problem. The unresolved
difficulty is generating temporally correct text and using it robustly under
the distribution shift between ground-truth training text and noisy
inference-time pseudo-text.

---

## Data layout

The notebooks expect data under Google Drive:

```text
/content/drive/MyDrive/mmf_tas_lab_data/
```

Important working directories:

```text
mmf_tas_lab_data/
  zenodo_ms_tcn_data/
    breakfast/
      features/
      groundTruth/
      splits/
      mapping.txt

  breakfast_raw_videos/

  text_assisted_tas/
    breakfast/
      text_embeddings/
      procedurevrl/
      procedurevrl_hidden/
      runs/
        32_breakfast_qwen_catalog_top1_alignment/
        33_breakfast_qwen35b_capacity_test/

    assembly101/
      coarse_mstcn_format/
        streaming_visual_features_v1/
          procedurevrl_hidden/
          clip_vitb16/
          runs/

    final_report_artifacts/
      notebook22/
      notebook22_presentation_artifacts.zip
```

---

## Reproducing the experiments

1. Open the notebooks in Google Colab and mount Google Drive.

2. Run the desired pipeline sequentially:

   - Breakfast core pipeline: `00 → 11`;
   - Breakfast Qwen3.5-2B catalog follow-up: notebook `32` after notebooks `00`,
     `02`, `09`, and `11`;
   - Breakfast Qwen3.5-35B-A3B capacity experiment: notebook `33` after the same
     Breakfast artifacts are available;
   - Assembly101 core pipeline: `12 → 23`;
   - Assembly101 Qwen pseudo-text follow-up: `24 → 29`;
   - locked ensemble validation: notebook `30` after notebooks `16`, `19–21`,
     `23`, and `28`;
   - full-catalog Assembly101 alignment: notebook `31` after the required
     artifacts from notebooks `14`, `18`, `24`, `25`, and `28`.

3. Run notebook `22` after notebooks `16–21` to recreate the core reporting
   artifacts.

Notebooks `23–33` additionally write their own tables, figures, audits, and
machine-readable summary files under their run directories.

Notebooks 32 and 33 write each completed Breakfast Qwen prediction to Google
Drive. Completed videos are detected and skipped when a run is resumed.

### Hardware notes

- Training and Qwen notebooks require a GPU runtime.
- Qwen3.5-2B can run on a smaller Colab GPU.
- The tested Qwen3.5-35B-A3B GPTQ Int4 configuration requires approximately
  `21 GiB` of GPU memory for the loaded model and was validated on an A100
  40 GB.
- The complete 35B Breakfast run processed all 252 videos successfully.
- Completed predictions can be evaluated without loading Qwen again.

Checkpoints, logits, predictions, summaries, and completion manifests are
stored on Google Drive so long stages can be resumed without committing large
artifacts to Git.

---

## Reporting artifacts

The reporting notebooks produce:

- completion, provenance, and reconstruction audits;
- architecture and representation comparisons;
- strategy and hyperparameter tables;
- fixed and learnable prototype ablations;
- three-seed means, sample standard deviations, and paired deltas;
- Qwen pseudo-text quality and visual-input integrity diagnostics;
- full-catalog closed-set alignment and collapse statistics;
- Assembly101 and Breakfast catalog-size comparisons;
- Qwen3.5-2B versus Qwen3.5-35B-A3B capacity comparisons;
- guarded calibration and held-out validation decisions;
- presentation-ready F1@50 tables and figures;
- machine-readable final summaries.

---
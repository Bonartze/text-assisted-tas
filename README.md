# Text-Assisted Temporal Action Segmentation

This repository studies whether text can improve **Temporal Action Segmentation
(TAS)** during training while preserving a strictly **video-only inference
pipeline**.

The experiments cover two datasets:

- **Breakfast** — proof-of-concept and controlled teacher–student experiments;
- **Assembly101** — the main architecture, text-alignment, and knowledge
  distillation study with a train/validation/test protocol.

The project evaluates three ways of exploiting text:

1. a privileged teacher that receives video and ground-truth action text;
2. logit knowledge distillation (KD) into a video-only student;
3. direct alignment of student features with fixed or learnable CLIP action-text
   prototypes.

The final deployable model never receives text and never calls the teacher:

```text
training:  video + training-time text supervision
inference: video → student → action segmentation
```

## Main conclusion

Training-time text supervision provides a **modest positive signal** for a
video-only student.

On Assembly101, the validation-selected KD student improves test F1@25 from
`23.14` for the controlled CE-only student to `24.11` (`+0.97`). A separate
text-aligned student reaches `23.76`. Combining text alignment and KD does not
produce an additional gain.

These differences come from one seed and should not be interpreted as
statistically significant.

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
| Hyperparameter selection | Validation only |
| Test F1@25 | **24.11** |

---

## Repository structure

```text
notebooks/
  00–11  Breakfast data preparation, visual baselines, teachers, and KD
  12–15  Assembly101 download, annotation conversion, and feature extraction
  16–18  Assembly101 architecture/representation and privileged-teacher ablations
  19–21  CE, KD, learnable text embeddings, and staged hyperparameter search
  22     Final aggregation, protocol audit, tables, figures, and presentation artifacts
```

Large datasets, raw videos, checkpoints, extracted features, predictions, and
model weights are intentionally not stored in Git.

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

### Assembly101: notebooks 12–22

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
| `22_final_ablation_analysis_and_presentation_artifacts_COLAB.ipynb` | Aggregates completed results and creates final tables, figures, conclusions, and presentation artifacts. |

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

### Text-assisted strategies

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

---

## Assembly101 results

### Architecture × representation ablation

Visual-only test results:

| Architecture | Representation | Accuracy | Edit | F1@10 | F1@25 | F1@50 |
|---|---|---:|---:|---:|---:|---:|
| **MS-TCN** | **CLIP** | 25.86 | 25.01 | 28.00 | **23.82** | 17.58 |
| MS-TCN | ProcedureVRL hidden | 26.16 | 24.74 | 28.36 | 23.79 | 15.72 |
| LTContext | CLIP | 27.10 | 22.70 | 25.62 | 21.96 | 13.39 |
| LTContext | ProcedureVRL hidden | 27.10 | 22.80 | 25.11 | 20.05 | 11.92 |

MS-TCN + CLIP is used for the final student experiments because it provides the
best visual-only test F1@25 among the evaluated architecture/representation
pairs.

### Strategy comparison

| Strategy | Configuration | Training text | Inference text | Deployable | Accuracy | Edit | F1@10 | F1@25 | F1@50 |
|---|---|:---:|:---:|:---:|---:|---:|---:|---:|---:|
| Visual baseline | MS-TCN + CLIP | No | No | Yes | 25.86 | 25.01 | 28.00 | 23.82 | 17.58 |
| Controlled CE-only | Same student pipeline, `λKD=0` | No | No | Yes | 25.94 | 24.75 | 27.58 | 23.14 | 16.53 |
| Original KD | `λKD=0.1`, `T=4`, all stages | Yes | No | Yes | 26.38 | 24.64 | 28.40 | 23.75 | **17.73** |
| Text-only | Learnable prototypes, `β=0.01` | Yes | No | Yes | 26.27 | 24.78 | 28.58 | 23.76 | 17.56 |
| **Expanded KD** | **`λKD=0.2`, `T=4`, all stages** | **Yes** | **No** | **Yes** | 25.67 | **25.10** | 28.22 | **24.11** | 17.36 |
| Text + KD | `β=0.01`, `λKD=0.2`, `T=4`, all stages | Yes | No | Yes | 26.24 | 24.85 | **28.73** | 23.67 | 17.32 |
| Oracle teacher | Video + ground-truth text | Yes | Yes | No | **44.27** | **37.39** | **41.65** | **40.94** | **36.81** |

The primary controlled Assembly101 comparison is:

| KD student − CE-only student | Accuracy | Edit | F1@10 | F1@25 | F1@50 |
|---|---:|---:|---:|---:|---:|
| Difference | −0.27 | +0.35 | +0.64 | **+0.97** | +0.83 |

KD improves the segment-level metrics, but the gain is metric-dependent:
frame-wise accuracy decreases by `0.27` points.

### Validation-selected KD hyperparameters

| Ablation | Candidates | Selected value | Selected validation F1@25 |
|---|---|---:|---:|
| KD coefficient | `0.01, 0.05, 0.1, 0.2, 0.5, 1.0` | **0.2** | **25.94** |
| Temperature | `1, 2, 4, 8` | **4** | **25.94** |
| Stage policy | `all, final` | **all** | **25.94** |

The selection score is:

```text
validation F1@25 + 0.01 × validation Edit
```

No test metric is used during the Assembly101 hyperparameter search.

### Fixed vs learnable text prototypes

| Prototype mode | `β` | Validation F1@25 | Selection score |
|---|---:|---:|---:|
| Fixed | 0.01 | 25.69 | 25.946 |
| Fixed | 0.05 | 24.87 | 25.134 |
| Fixed | 0.10 | 25.80 | 26.059 |
| **Learnable** | **0.01** | **25.80** | **26.063** |
| Learnable | 0.05 | 25.26 | 25.512 |
| Learnable | 0.10 | 25.29 | 25.550 |

The selected learnable strategy reaches test F1@25 `23.76`, compared with
`23.14` for the controlled CE-only student. However, fixed and learnable
prototypes are effectively tied on validation, so the results do not establish
a decisive benefit from prototype learnability itself.

---

## Breakfast results

All results below use split 1.

| Strategy | Inference input | Accuracy | Edit | F1@10 | F1@25 | F1@50 |
|---|---|---:|---:|---:|---:|---:|
| Historical ProcedureVRL hidden baseline | Video only | **58.04** | 55.57 | **58.58** | **57.73** | 45.76 |
| Controlled CE-only student | Video only | 57.42 | 55.61 | 55.76 | 54.29 | 45.34 |
| Video-only KD student | Video only | 57.99 | **56.46** | 56.67 | 55.42 | **46.19** |
| Privileged text teacher | Video + ground-truth text | 98.26 | 96.26 | 96.90 | 96.90 | 96.71 |

The controlled KD student improves over the same-notebook CE-only student:

| KD student − CE-only student | Accuracy | Edit | F1@10 | F1@25 | F1@50 |
|---|---:|---:|---:|---:|---:|
| Difference | +0.57 | +0.85 | +0.91 | **+1.13** | +0.85 |

The historical visual baseline remains stronger than the KD student on
Breakfast F1@25 (`57.73` vs `55.42`). It is therefore reported as context, not
as evidence that KD is the best Breakfast model.

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

    assembly101/
      coarse_mstcn_format/
        streaming_visual_features_v1/
          procedurevrl_hidden/
          clip_vitb16/
          runs/

    final_report_artifacts/
      notebook22/
        tables/
        figures/
        text/
        final_summary.json
      notebook22_presentation_artifacts.zip
```

---

## Reproducing the experiments

1. Open the notebooks in Google Colab and mount Google Drive.
2. Run notebooks sequentially within the desired pipeline:
   - Breakfast: `00 → 11`;
   - Assembly101: `12 → 21`.
3. Run notebook `22` after all required `final_summary.json` files have status
   `completed`.
4. Use the notebook 22 output tables and figures for reporting; it performs no
   model training.

The full Assembly101 training notebooks are intended for a GPU runtime.
Completed runs, checkpoints, and summaries are stored on Google Drive so that
long experiments can be resumed without committing large artifacts to Git.

---

## Reporting artifacts

Notebook 22 produces:

- a completion and protocol audit;
- architecture × representation tables;
- strategy and hyperparameter ablation tables;
- four presentation-ready figures;
- final conclusions and limitations;
- an eight-slide presentation outline;
- a concise supervisor update;
- a ZIP archive containing the generated artifacts.

---

## Known limitations

1. Final model comparisons currently use one seed; small differences are not
   statistically confirmed.
2. The privileged teacher receives ground-truth action text and is an oracle
   upper bound, not a deployable model.
3. ProcedureVRL and CLIP sequences contain 16 temporal positions, which may
   under-resolve short actions.
4. Breakfast uses a legacy split-1 protocol, and some early I3D/ProcedureVRL
   experiments use different temporal resolutions.
5. The combined text + KD objective was evaluated using separately selected
   loss weights; a larger joint search might produce a different result.
6. The final claims concern controlled improvements over CE-only students, not
   state-of-the-art performance.

---

## Follow-up

The highest-value follow-up is a three-seed repetition of:

1. the controlled CE-only student;
2. the selected text-only student;
3. the selected KD student.

This would estimate variance and determine whether the observed sub-one-point
Assembly101 improvements are reproducible.

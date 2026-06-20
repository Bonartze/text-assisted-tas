# Current Status

This document summarizes the current state of the project based on the three active notebooks:

```text
notebooks/
├── 01_data_preparation.ipynb
├── 02_text_embeddings_breakfast.ipynb
└── 03_mstcn_teacher_student_training.ipynb
```

The current implementation is a working proof-of-concept on the Breakfast dataset. The main idea tested so far is to use action text information during training, while keeping the final student model video-only at inference time.

---

## 1. `01_data_preparation.ipynb`

This notebook prepares and validates the dataset setup.

### What it does

- Sets up the project paths.
- Checks the MS-TCN-style Breakfast data layout.
- Validates feature files, ground-truth label files, split files and class mapping.
- Saves a small data preparation summary.

### Current Breakfast status

| Item | Value |
|---|---:|
| Feature files | 1712 |
| Ground-truth files | 1712 |
| Split files | 8 |
| Classes | 48 |
| Missing features in splits | 0 |
| Missing labels in splits | 0 |
| Sample feature shape | `(2048, 832)` |
| Status | ready |

The notebook also detects a legacy optional 50Salads setup:

| Dataset | Feature files | Ground-truth files | Split files | Classes | Status |
|---|---:|---:|---:|---:|---|
| 50Salads legacy optional | 50 | 50 | 10 | 19 | ready |

### Assessment

Breakfast is correctly prepared and ready for the current experiments. 50Salads is present, but it is currently only optional and is not the main focus of the current project stage.

---

## 2. `02_text_embeddings_breakfast.ipynb`

This notebook prepares the text side of the experiment.

### What it does

- Loads the Breakfast class mapping.
- Converts action labels into readable text.
- Builds CLIP-style prompts such as a video of the action: pour cereals.
- Performs a full scan of all Breakfast feature files.
- Checks consistency between features and ground-truth labels.
- Generates CLIP ViT-B/16 text embeddings for all action classes.
- Saves metadata, prompt files, dataset manifest, and text embedding files.

### Current result

| Item | Value |
|---|---:|
| Feature files | 1712 |
| Ground-truth files | 1712 |
| Common feature/label ids | 1712 |
| Missing feature ids | 0 |
| Missing ground-truth ids | 0 |
| Manifest rows | 1712 |
| Bad feature rows | 0 |
| Feature dimension | 2048 for all 1712 files |
| Classes | 48 |
| Text embedding shape | `(48, 512)` |

The generated text embedding files are:

```text
text_embeddings/
├── breakfast_clip_vitb16_text_embedding_config.json
├── breakfast_clip_vitb16_text_embedding_metadata.csv
└── breakfast_clip_vitb16_text_embeddings.npy
```

### Assessment

The text preprocessing stage is complete for Breakfast. The current CLIP usage is on the text side: action labels are embedded as text prototypes. Visual features are still the existing Breakfast I3D/MS-TCN-style features, not newly extracted ProcedureVRL or CLIP/VideoCLIP visual features yet.

---

## 3. `03_mstcn_teacher_student_training.ipynb`

This notebook implements the current training proof-of-concept.

### What it does

The notebook trains and compares four MS-TCN-style models on Breakfast split 1:

| Model | Description | Text during training | Text during inference |
|---|---|---:|---:|
| `baseline_visual_only` | Visual-only MS-TCN-style baseline | no | no |
| `text_aware_teacher` | Teacher model using visual features and CLIP action text prototypes | yes | yes |
| `student_ce_only` | Video-only student trained with cross entropy only | no | no |
| `student_kd_video_only` | Video-only student trained with knowledge distillation from the text-aware teacher | yes | no |

The important setup point is that `student_kd_video_only` can use text-derived supervision during training, but it remains video-only at inference.

### Full Breakfast split 1 run

| Item | Value |
|---|---:|
| Dataset | Breakfast |
| Split | 1 |
| Train videos | 1460 |
| Test videos | 252 |
| Classes | 48 |
| Visual feature dimension | 2048 |
| Text embedding shape | `(48, 512)` |
| Epochs per model | 10 |
| Model type | MS-TCN-style prototype |

### Frame-wise accuracy results

| Model | Train accuracy | Test accuracy | Text during training | Text during inference |
|---|---:|---:|---:|---:|
| `baseline_visual_only` | 0.6668 | 0.5330 | no | no |
| `text_aware_teacher` | 0.4405 | 0.4043 | yes | yes |
| `student_ce_only` | 0.7058 | 0.5435 | no | no |
| `student_kd_video_only` | 0.5906 | 0.4836 | yes | no |

The best result in the current full split 1 run is the cross-entropy-only video student:

```text
student_ce_only test accuracy = 0.5435
```

The visual-only baseline is close:

```text
baseline_visual_only test accuracy = 0.5330
```

The current KD student does not outperform the CE-only student:

```text
student_kd_video_only test accuracy = 0.4836
```

### Assessment

The full split 1 experiment runs end-to-end successfully. The current implementation proves that the training pipeline works, the Breakfast data are loaded correctly, the CLIP text embeddings are connected to the TAS training code, and the final KD student can be evaluated as a video-only inference model.

However, the current text-aware teacher and KD design do not yet improve performance on the full Breakfast split 1 run. The CE-only video student is currently the strongest model. This suggests that the current text fusion / teacher design needs improvement before making a positive claim about text-guided distillation.

The current result is still useful because it gives a complete baseline and a clear next direction: improve the text-aware teacher or add a simpler concatenation-based teacher baseline before evaluating more broadly.

---

## Current conclusion

The current project stage is a working proof-of-concept, not the final benchmark yet.

The most important result is:

```text
The full Breakfast split 1 training pipeline works end-to-end.
```

The current modeling result is:

```text
The CE-only video student currently performs best.
The current KD/text-guided setup does not yet improve over the video-only baselines.
```

This means the pipeline is ready for the next research iteration, but the text-guided distillation method still needs improvement.

---

## Current limitations

- Only frame-wise accuracy is reported so far.
- Standard TAS metrics are not implemented yet:
  - Edit score
  - F1@10
  - F1@25
  - F1@50
- The current model is an MS-TCN-style implementation, not yet a direct official MS-TCN repository benchmark.
- The current text integration is a prototype.
- The current CLIP usage is text-side only.
- ProcedureVRL / CLIP-like visual feature extraction is not done yet.
- Assembly101 is not integrated yet.
- LTContext is not integrated yet.
- Current results are from Breakfast split 1 only.

---

## Planned next steps

### Immediate next steps

1. Add standard TAS metrics:
   - Edit score
   - F1@10
   - F1@25
   - F1@50

2. Add a simple concatenation-based teacher baseline:
   - visual features + text prototype information during teacher training
   - distill into a video-only student

3. Improve the text-aware teacher design:
   - better text fusion
   - better KD objective
   - possibly tune KD temperature and CE/KD weighting

4. Re-run Breakfast split 1 with the improved teacher/KD setup.

### After that

5. Run all Breakfast splits and report average results.

6. Compare more carefully against MS-TCN-style and/or official MS-TCN baselines.

7. Move toward the larger project scope:
   - ProcedureVRL / CLIP-like visual features
   - Assembly101
   - LTContext
   - comparison of visual-only vs text-guided video-only inference models

---

## Short summary

The project now has a complete Breakfast proof-of-concept:

```text
Data preparation -> CLIP text embeddings -> MS-TCN-style training -> full split 1 comparison
```

The pipeline works, but the current KD/text-guided method is not yet better than the video-only CE baseline. The next step is to improve the teacher/text fusion and add proper TAS evaluation metrics.
# Text-Assisted Temporal Action Segmentation

This repository contains experiments for text-assisted Temporal Action Segmentation (TAS), with a focus on the Breakfast dataset and ProcedureVRL/ProceduralVLR-style video representations.

The main question is:

```text
Can textual supervision be used during training to improve a final video-only TAS model?
```

The current final answer is:

```text
Yes, in a controlled training-time text-supervision setup:
a privileged text teacher improves a video-only KD student over a CE-only video student.
```

The final deployed/evaluated student model uses only video features at inference.

---

## Repository structure

```text
notebooks/
  00_download_breakfast_raw_videos_hf.ipynb
  01_prepare_breakfast_mstcn_data.ipynb
  02_extract_clip_text_embeddings_breakfast.ipynb
  03_internal_teacher_student_kd_i3d_clip.ipynb
  04_concat_text_teacher_kd_i3d_clip.ipynb
  05_official_mstcn_i3d_baseline_breakfast.ipynb
  06_official_style_text_teacher_kd_i3d_clip.ipynb
  07_procedurevrl_extract_features_breakfast.ipynb
  08_mstcn_procedurevrl_coarse_baseline_breakfast.ipynb
  09_procedurevrl_hidden_embeddings_breakfast_COLAB.ipynb
  10_mstcn_procedurevrl_hidden_baseline_breakfast.ipynb
  11_procedurevrl_hidden_privileged_text_teacher_kd_breakfast.ipynb
```

Large data files, raw videos, checkpoints, extracted features, prediction files, and model weights are intentionally not stored in Git.

---

## Notebook pipeline

| Notebook | Purpose |
|---|---|
| `00_download_breakfast_raw_videos_hf.ipynb` | Downloads raw Breakfast videos. |
| `01_prepare_breakfast_mstcn_data.ipynb` | Prepares the Breakfast MS-TCN-format dataset from existing features/labels/splits. |
| `02_extract_clip_text_embeddings_breakfast.ipynb` | Extracts CLIP text embeddings for Breakfast action labels. |
| `03_internal_teacher_student_kd_i3d_clip.ipynb` | Early internal teacher/student KD proof-of-concept using I3D features and CLIP text. |
| `04_concat_text_teacher_kd_i3d_clip.ipynb` | Improved concat text-teacher experiment using I3D + CLIP. |
| `05_official_mstcn_i3d_baseline_breakfast.ipynb` | Official-style MS-TCN visual-only Breakfast baseline on I3D features. |
| `06_official_style_text_teacher_kd_i3d_clip.ipynb` | Official-style text-teacher / video-only student KD experiment on I3D + CLIP. |
| `07_procedurevrl_extract_features_breakfast.ipynb` | Extracts coarse ProcedureVRL output-level features and converts them to MS-TCN format. |
| `08_mstcn_procedurevrl_coarse_baseline_breakfast.ipynb` | Trains an MS-TCN-style baseline on coarse ProcedureVRL features. |
| `09_procedurevrl_hidden_embeddings_breakfast_COLAB.ipynb` | Extracts hidden 512-dimensional ProcedureVRL video embeddings from `model.head`. |
| `10_mstcn_procedurevrl_hidden_baseline_breakfast.ipynb` | Trains the visual-only MS-TCN-style baseline on hidden ProcedureVRL embeddings. |
| `11_procedurevrl_hidden_privileged_text_teacher_kd_breakfast.ipynb` | Trains a privileged text teacher and a final video-only KD student. |

---

## Data layout

The notebooks expect the working data under Google Drive:

```text
/content/drive/MyDrive/mmf_tas_lab_data/
```

Important subdirectories:

```text
mmf_tas_lab_data/
  zenodo_ms_tcn_data/
    breakfast/
      features/
      groundTruth/
      splits/
      mapping.txt

  breakfast_raw_videos/
    videos/

  text_assisted_tas/
    breakfast/
      text_embeddings/
        breakfast_clip_vitb16_text_embeddings.npy
        breakfast_clip_vitb16_text_embedding_metadata.csv

      procedurevrl/
        runs/
          procedurevrl_breakfast_full_split1_split1_views16/

      procedurevrl_hidden/
        runs/
          procedurevrl_hidden_full_split1_split1_views16/
          mstcn_hidden_baseline_split1/
          procedurevrl_hidden_privileged_text_teacher_kd_full_split1_split1/
```

---

## Main results

All Breakfast results below are on split 1.

| Experiment | Features | Model | Inference input | Accuracy | Edit | F1@10 | F1@25 | F1@50 |
|---|---|---|---|---:|---:|---:|---:|---:|
| I3D baseline | I3D | Official MS-TCN visual-only, 30 epochs | video only | 55.38 | 44.97 | 39.05 | 34.80 | 25.25 |
| I3D + CLIP KD proof-of-concept | I3D + CLIP text teacher | Video-only KD student, `λ=0.05` | video only | 70.98 | 58.53 | 50.16 | 46.86 | 38.33 |
| ProcedureVRL coarse baseline | ProcedureVRL output-level features | MS-TCN-style visual-only | video only | 48.91 | 49.15 | 55.46 | 54.38 | 42.61 |
| ProcedureVRL hidden baseline | ProcedureVRL hidden embeddings | MS-TCN-style visual-only | video only | 58.04 | 55.57 | 58.58 | 57.73 | 45.76 |
| Controlled CE-only student | ProcedureVRL hidden embeddings | MS-TCN-style CE-only student | video only | 57.42 | 55.61 | 55.76 | 54.29 | 45.34 |
| Privileged text teacher | ProcedureVRL hidden + ground-truth text | Oracle / privileged teacher | video + ground-truth text | 98.26 | 96.26 | 96.90 | 96.90 | 96.71 |
| Privileged-text KD student | ProcedureVRL hidden embeddings | Video-only KD student, `λ=0.05`, `T=4.0` | video only | 57.99 | 56.46 | 56.67 | 55.42 | 46.19 |

The final controlled comparison is:

| Comparison | Accuracy | Edit | F1@10 | F1@25 | F1@50 |
|---|---:|---:|---:|---:|---:|
| KD student − CE-only student | +0.57 | +0.85 | +0.91 | +1.13 | +0.85 |

This supports the training-time text-supervision hypothesis in a controlled setting: the teacher uses text during training, while the final KD student is video-only at inference.

---

## ProcedureVRL feature variants

Two ProcedureVRL-based representations were extracted:

| Representation | Per-video feature shape | Interpretation |
|---|---:|---|
| Coarse output-level ProcedureVRL features | `[9871, 16]` | Output-level clip features / prediction-like ProcedureVRL outputs. Useful as pipeline validation. |
| Hidden ProcedureVRL embeddings | `[512, 16]` | Hidden video embeddings extracted from `model.head = Linear(768 → 512)`. This is the stronger and cleaner representation for text-assisted experiments. |

The hidden extraction processed:

```text
videos: 1712
train videos: 1460
test videos: 252
hidden shape: [1712, 16, 512]
missing hidden positions: 0
duplicates: 0
NaNs: 0
```

---

## Interpretation

The earlier I3D + CLIP experiments are useful as proof-of-concept, but the visual and text features come from different latent spaces. The ProcedureVRL pipeline addresses this by moving the main experiments to ProcedureVRL hidden video embeddings.

The first fixed text-prototype teacher did not improve over the visual-only hidden baseline. The final notebook therefore uses a stronger privileged teacher: during training it receives ground-truth action text embeddings at every timestep. This teacher is not deployable, but it can provide soft supervision to a video-only student.

The important final model is the KD student:

```text
training:  video features + distillation from privileged text teacher
inference: video features only
```

---

## Known limitations

1. The ProcedureVRL features are temporally coarse: each video is represented by 16 clips.
2. Breakfast results are reported on split 1 only.
3. Several model selections are exploratory and use the test split to select the best checkpoint. For stricter evaluation, a train/validation/test protocol should be used.
4. The privileged teacher is an oracle model because it uses ground-truth action text. It is used only for distillation, not for deployment.

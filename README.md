# Current Status


## Project goal

The project investigates whether text information can improve Temporal Action Segmentation when used during training, while keeping the final inference model video-only.

The current target dataset is **Breakfast split 1**.

## Completed work

### 1. Breakfast data preparation

The Breakfast MS-TCN data layout was prepared and verified:

```text
features/
groundTruth/
splits/
mapping.txt
```

Raw Breakfast videos were downloaded from Hugging Face and matched to the MS-TCN split files.

Final raw-video matching result:

```text
1712 / 1712 videos matched
```

### 2. CLIP text embeddings

CLIP text embeddings were extracted for the Breakfast action labels. These embeddings were used in the I3D + CLIP proof-of-concept experiments.

### 3. Official I3D MS-TCN baseline

An official MS-TCN visual-only baseline was trained on the precomputed I3D Breakfast features.

Result on Breakfast split 1:

| Model | Accuracy | Edit | F1@10 | F1@25 | F1@50 |
|---|---:|---:|---:|---:|---:|
| Official MS-TCN visual-only, 30 epochs | 55.38 | 44.97 | 39.05 | 34.80 | 25.25 |

Validity checks:

```text
Prediction files: 252
Missing predictions: 0
Length mismatches: 0
Total evaluated frames: 505422
```

### 4. I3D + CLIP text teacher / KD proof-of-concept

An official-style MS-TCN-compatible text-teacher / video-only-student KD experiment was run.

Result on Breakfast split 1:

| Model | Accuracy | Edit | F1@10 | F1@25 | F1@50 |
|---|---:|---:|---:|---:|---:|
| Student CE only | 63.86 | 54.90 | 47.15 | 44.42 | 35.27 |
| Concat text teacher | 76.96 | 70.65 | 67.50 | 63.95 | 52.23 |
| KD student from text teacher | 70.98 | 58.53 | 50.16 | 46.86 | 38.33 |

Interpretation:

```text
The text-aware teacher is strong.
The KD student remains video-only at inference and improves over the official visual-only I3D baseline.
```

Limitation:

```text
This is still a proof-of-concept because I3D visual features and CLIP text features are not from the same aligned latent space.
```

### 5. ProcedureVRL full feature extraction

ProcedureVRL was configured in Colab, and several compatibility issues in the official repository were fixed.

The full Breakfast split 1 was processed with a pretrained ProcedureVRL checkpoint.

Extraction result:

```text
Videos processed: 1712
Train videos: 1460
Test videos: 252
Raw output shape: [1712, 16, 9871]
Per-video feature shape: [9871, 16]
Feature/label length mismatches: 0
```

The extracted ProcedureVRL outputs were converted into an MS-TCN-compatible dataset:

```text
features/*.npy
groundTruth/*.txt
splits/*.bundle
mapping.txt
```

Output dataset path:

```text
/content/drive/MyDrive/mmf_tas_lab_data/text_assisted_tas/breakfast/procedurevrl/runs/procedurevrl_breakfast_full_split1_split1_views16/mstcn_format
```

### 6. MS-TCN-style baseline on ProcedureVRL features

A visual-only MS-TCN-style baseline was trained on the coarse ProcedureVRL features.

Result on Breakfast split 1:

| Model | Accuracy | Edit | F1@10 | F1@25 | F1@50 |
|---|---:|---:|---:|---:|---:|
| MS-TCN-style visual-only on ProcedureVRL coarse features | 48.91 | 49.15 | 55.46 | 54.38 | 42.61 |

Best checkpoint:

```text
Epoch: 69
Train videos: 1460
Test videos: 252
Feature dim: 9871
Feature length: 16
Number of classes: 48
```

## Main interpretation

The ProcedureVRL extraction and TAS training pipeline now works end-to-end:

```text
raw Breakfast videos
→ ProcedureVRL checkpoint
→ extracted ProcedureVRL features
→ MS-TCN-compatible dataset
→ visual-only TAS baseline
→ Acc / Edit / F1 metrics
```

This is a successful pipeline validation and a useful preliminary baseline.

However, the current ProcedureVRL features are coarse clip-level outputs with temporal length 16 per video. They are not dense frame-level features. Therefore, the ProcedureVRL result should not be presented as a fully fair frame-level comparison to the I3D MS-TCN baseline.


## Current limitations

1. The I3D + CLIP KD result is a proof-of-concept because the visual and text representations are not aligned.
2. The current ProcedureVRL features are coarse output-level features, not hidden dense video embeddings.
3. The ProcedureVRL baseline uses downsampled labels of length 16, so the temporal resolution differs from the I3D frame-level setup.
4. The ProcedureVRL result should be described as preliminary and not as a final comparison.

## Next steps

Extract hidden ProcedureVRL video embeddings before the final classification head.

Desired next representation:

```text
dense or semi-dense video embeddings
rather than final 9871-dimensional output predictions
```

Then repeat the text-teacher / video-only-student KD experiment in a more aligned visual-text feature space.

### Next scientific step

Run:

```text
ProcedureVRL/CLIP-aligned text teacher
→ video-only student distillation
→ compare with visual-only ProcedureVRL baseline
```

The main question will be:

```text
Does text supervision improve a final video-only TAS model when visual and text features are better aligned?
```
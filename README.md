# Text-Assisted Temporal Action Segmentation

This repository contains a proof-of-concept for text-assisted temporal action segmentation on the Breakfast dataset.

The current experiment uses existing Breakfast I3D/MS-TCN-style visual features and CLIP text embeddings generated from action labels. The main goal of the current stage is to test whether text can be used during training while keeping the final student model video-only at inference time.

Detailed progress and results are documented in:

```text
docs/current_status.md
```

## Current stage

The current implementation focuses on Breakfast split 1 and includes:

* data preparation and validation;
* CLIP ViT-B/16 text embeddings for Breakfast action labels;
* an MS-TCN-style visual-only baseline;
* a text-aware teacher model;
* a video-only student trained with cross entropy;
* a video-only student trained with knowledge distillation from the text-aware teacher.

The current full Breakfast split 1 run completed successfully.

| Model                   | Test accuracy | Text during training | Text during inference |
| ----------------------- | ------------: | -------------------: | --------------------: |
| `baseline_visual_only`  |        0.5330 |                   no |                    no |
| `text_aware_teacher`    |        0.4043 |                  yes |                   yes |
| `student_ce_only`       |        0.5435 |                   no |                    no |
| `student_kd_video_only` |        0.4836 |                  yes |                    no |

The current best model is the CE-only video student. The KD/text-guided setup works technically, but it does not yet improve over the video-only baselines on the full Breakfast split 1 run.

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── configs/
│   ├── README.md
│   ├── local_debug.json
│   ├── scale1_control.json
│   └── full_split1.json
├── docs/
│   ├── current_status.md
│   ├── 01_data_preparation_summary.csv
│   └── 01_project_status.json
└── notebooks/
    ├── 01_data_preparation.ipynb
    ├── 02_text_embeddings_breakfast.ipynb
    └── 03_mstcn_teacher_student_training.ipynb
```

## Data

Large data files are not included in this repository.

The expected local data layout is:

```text
data/
├── zenodo_ms_tcn_data/
│   └── breakfast/
│       ├── features/
│       ├── groundTruth/
│       ├── splits/
│       └── mapping.txt
└── text_assisted_tas/
    └── breakfast/
        └── text_embeddings/
            ├── breakfast_clip_vitb16_text_embedding_config.json
            ├── breakfast_clip_vitb16_text_embedding_metadata.csv
            └── breakfast_clip_vitb16_text_embeddings.npy
```

The `data/` directory is intentionally ignored by Git.

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install PyTorch separately depending on the CUDA setup.

For the local GTX 1060 setup used during development:

```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

Then install the remaining requirements:

```bash
pip install -r requirements.txt
```

Register the environment as a Jupyter kernel:

```bash
python -m ipykernel install --user --name text-assisted-tas --display-name "Python (text-assisted-tas)"
```

## Notebooks

### `01_data_preparation.ipynb`

Validates the Breakfast data structure and checks feature files, ground-truth files, split files, and class mapping.

### `02_text_embeddings_breakfast.ipynb`

Creates CLIP text embeddings for Breakfast action classes and saves the text embedding files used by the training notebook.

### `03_mstcn_teacher_student_training.ipynb`

Runs the MS-TCN-style teacher-student training experiment.

Available run modes:

* `local_debug`: small sanity-check run;
* `scale1_control`: controlled proof-of-concept run;
* `full_split1`: full Breakfast split 1 run.

## Current limitations

The current implementation is a proof-of-concept, not the final benchmark.

Current limitations:

* only frame-wise accuracy is reported;
* Edit score and F1@10/25/50 are not implemented yet;
* the MS-TCN model is an MS-TCN-style prototype, not yet an official MS-TCN repo integration;
* the current CLIP usage is text-side only;
* ProcedureVRL / CLIP-like visual feature extraction is not integrated yet;
* Assembly101 and LTContext are not integrated yet.

## Next steps

Planned next steps:

1. Add standard TAS metrics:

   * Edit score;
   * F1@10;
   * F1@25;
   * F1@50.

2. Add a simple concatenation-based teacher baseline.

3. Improve the text-aware teacher and KD setup.

4. Run all Breakfast splits and report averaged results.

5. Extend the pipeline toward ProcedureVRL / CLIP-like visual features, Assembly101 and LTContext.
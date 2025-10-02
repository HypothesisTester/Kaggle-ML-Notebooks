# Autonomous Motion Prediction (Lyft L5)

Motion forecasting pipeline built for the Lyft Level-5 Kaggle competition. The notebooks walk through profiled data exploration, training BEV models with L5Kit + PyTorch, and preparing Kaggle submissions.

## Tech Stack
- Python, PyTorch, L5Kit for rasterisation and trajectory modelling
- Hydra-driven configs, Albumentations for controlled experimentation
- Catalyst and Kekas for training loops, plus pandas/seaborn for analysis

## Notebook Guide
- `01_eda_and_data_loading.ipynb` — dataset tour: renders semantic/satellite BEV tiles, agent-centric views, and animations to confirm raster alignment; profiles scene/agent metadata (centroid correlations, extent & yaw distributions, ego rotation matrices) and previews Albumentations effects on BEV histograms. Ends with Catalyst/Kekas wiring for quick smoke-test training.
- `02_train_baseline.ipynb` — PyTorch baseline: ResNet backbone widened for raster channels, masked MSE loss with `target_availabilities`, Adam (1e-3) optimiser, periodic checkpoints, and optional Neptune/W&B logging. Intended to scale from debug runs to full-train jobs.
- `03_inference_and_submission.ipynb` — loads saved weights and batches the competition test split to produce `(x, y)` forecasts and a Kaggle-ready `submission.csv`.
- `04_error_analysis_and_visuals.ipynb` — reserved for qualitative diagnostics (agent tracks, heatmaps, miss cases). Notebook stub is empty in this snapshot.

## Workflow Highlights
- **Data/EDA:** Verified BEV raster fidelity via semantic vs satellite overlays and agent-centric renders; quantified spatial distributions (extents, yaw) to guide normalisation; inspected ego rotation correlations to understand motion priors.
- **Modelling & Training:** Experimented with FPN–ResNeXt and ResNet-18 BEV backbones, GPU-trained with masked losses for clean supervision, early stopping, and checkpointing hooks for reproducible reruns.
- **Inference:** Mirrors training config, restores checkpoints, and streams predictions per agent/timestamp.

## Next Steps
- Populate the error-analysis notebook with qualitative miss cases and ADE/FDE metrics.
- Explore multimodal heads or graph-based context for richer trajectory hypotheses.

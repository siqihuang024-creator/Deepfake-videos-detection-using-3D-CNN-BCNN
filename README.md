# 3D-CNN + Bayesian CNN for Deepfake Video Detection

This project is the controlled temporal extension of `E:/PhD/Deepfake video BCN`.
It keeps the paper-faithful Bayesian head from Leyva et al., *Data-agnostic Face
Image Synthesis Detection using Bayesian CNNs*, while replacing the independent
2D frame extractor with a small 3D-CNN.

## Research question

The first matched comparison isolates the practical effect of adding temporal
convolution as closely as possible:

```text
E0 matched control:
frame -> original 2D CNN -> Bayesian FC1/Dropout/FC2 -> mean frame score

E2 temporal model:
8-frame clip -> 3D CNN -> temporal mean -> Bayesian FC1/Dropout/FC2 -> clip score
```

Both experiments use the same identity split, donor-safe DFD evaluation, clip
indices, temporally smoothed face crops, augmentation, optimizer, and model
selection rule. This separates temporal-convolution gains from preprocessing
gains. The historical 2D results in `Deepfake video BCN` remain useful as the
original baseline, but are not the matched-control result.

## Preserved model interface

The 3D extractor uses three blocks with channels `16 -> 24 -> 32`:

```text
Conv3d kernel=(3,5,5), temporal padding=1
AvgPool3d kernel=(1,4,4), stride=(1,2,2)
Sigmoid
```

No temporal pooling is used inside the blocks. After the third block, temporal
mean pooling produces `32 x 22 x 22 = 15488` features. The downstream structure
is unchanged:

```text
BatchNorm2d(32)
Bayesian FC1: 15488 -> 512
Dropout: 0.2
Bayesian FC2: 512 -> 1
```

Only FC1 and FC2 use a mean-field Gaussian posterior. BatchNorm and the feature
extractor are deterministic, matching the repaired baseline implementation.

For the 2D control, eight independent frame outputs are averaged first and one
Gaussian likelihood is evaluated for the clip. Therefore both 2D and 3D use the
same clip-level KL normalization.

## Data protocol

- Joint training uses DFD and CelebDFv3 real videos only.
- Every epoch uses all 613 CelebDFv3 training videos once and resamples DFD to
  613 clips with new temporal positions, for 1,226 balanced real clips per epoch.
- Fake videos never enter backpropagation; they are used only for validation/test metrics.
- DFD validation/test use the donor-safe manifest: neither a fake target nor its donor leaks a DFD training identity.
- Training samples one contiguous 8-frame clip at stride 1 or 2.
- Model selection samples four deterministic positions per video; final evaluation uses eight.
- Horizontal flip is decided once per clip.
- Missing Haar boxes are filled by nearest/interpolated valid boxes before EMA smoothing.
- Video frames are decoded sequentially after one seek per clip.
- Checkpoint selection uses macro dataset AUROC so CelebDFv3 cannot hide weak DFD behavior.
- Labeled validation fakes are used for debug-stage checkpoint selection; this must be disclosed
  or replaced by a fixed-epoch/real-only rule in the final strict data-agnostic experiment.

## Run

The existing environment is `D:/Python/python.exe`. In PyCharm, change `TASK` in
`pycharm_run.py` and press Run.

Build the strict combined manifest:

```powershell
D:\Python\python.exe scripts\build_manifest.py `
  --dfd-root "E:\PhD\Deepfake Video TCN+BCNN\Datasets\DFD-Kaggle" `
  --celeb-root "E:\PhD\Deepfake Video TCN+BCNN\Datasets\CelebDFv3\CelebDFv3" `
  --output-dir artifacts\manifests --seed 42
```

Run architecture and Pyro checks:

```powershell
D:\Python\python.exe smoke_test.py --config configs\combined_3d_bcnn.yaml --forward
D:\Python\python.exe smoke_test.py --config configs\combined_3d_bcnn.yaml --svi-step
```

Run the DFD-only matched comparison first:

```powershell
D:\Python\python.exe train_3d_bcnn.py `
  --config configs\dfd_2d_control.yaml `
  --manifest artifacts\manifests\combined_manifest_dfd_donor_safe.csv

D:\Python\python.exe train_3d_bcnn.py `
  --config configs\dfd_3d_bcnn.yaml `
  --manifest artifacts\manifests\combined_manifest_dfd_donor_safe.csv
```

The DFD-only feasibility run has now been interpreted. The main 3D experiment
uses the combined real-only training pool (264 DFD + 613 CelebDFv3 videos):

```powershell
D:\Python\python.exe train_3d_bcnn.py `
  --config configs\combined_3d_bcnn.yaml `
  --manifest artifacts\manifests\combined_manifest_dfd_donor_safe.csv
```

Its validation pass evaluates DFD and CelebDFv3 separately and selects the
checkpoint using their macro AUROC, so the larger CelebDFv3 domain cannot hide
weak DFD performance.

Evaluate the DFD 3D checkpoint:

```powershell
D:\Python\python.exe evaluate_3d_bcnn.py `
  --config configs\dfd_3d_bcnn.yaml `
  --manifest artifacts\manifests\combined_manifest_dfd_donor_safe.csv `
  --checkpoint artifacts\run_dfd_3d_bcnn_t8_k3\checkpoints\best.pt `
  --split test --export-embeddings
```

Full CelebDFv3 test evaluation is expensive. During debugging, add
`--max-fakes-per-dataset 64`; final reported results should use the complete
test split.

## Outputs and interpretation

Each run writes `best.pt`, `last.pt`, `history.csv/json`, and training curves.
Evaluation writes video scores, per-dataset/per-method metrics, and optional
`15488`-D embeddings.

Track these signals together:

- DFD and CelebDFv3 AUROC separately, plus macro dataset AUROC.
- EER and `tpr_at_target_fpr` (TPR at 5% FPR).
- Real/fake posterior and anomaly-score means.
- Real/fake embedding norm, within-class variance, centroid L2 distance, and cosine distance.
- FPS, clip-level face-detection miss rate, and normalized center/scale box jitter by dataset/class.
- BatchNorm running variance and Bayesian `rho/sigma` diagnostics.

Evaluation reconstructs preprocessing from the checkpoint config. A later YAML
edit cannot silently change clip length, stride, crop margin, or normalization
for an older checkpoint; only device, dataset roots, and report location are runtime overrides.

If 3D embeddings separate real/fake but Bayesian scores do not, the retained
Bayesian head is the next bottleneck. If both stay mixed, the real-only feature
objective or video preprocessing needs attention before trying larger kernels.

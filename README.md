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

## Supervised experiments (S1/S2)

The one-class runs (E0/E2) all sit at chance, which does not separate "the
real-only objective is too hard" from "the feature extractor never learns".
The supervised protocol settles that: with labels available, a still-flat AUROC
points at the Sigmoid extractor and the optimizer, not at the objective.

Supervised training needs labelled fakes, which the one-class manifest parks in
`unused`. Derive the supervised manifest once:

```powershell
D:\Python\python.exe scriptsuild_supervised_manifest.py
```

It promotes a fake into `train` only when **both** its target and its donor are
training identities. The donor rule mirrors the donor-safe val/test rule: a
training fake built from a val/test donor would leak an evaluation identity
into backpropagation. Rows failing it become `excluded_donor_eval`.

| dataset | real train | fake train | excluded (donor outside train) |
| --- | --- | --- | --- |
| DFD | 264 | 1894 | 518 |
| CelebDFv3 | 613 | 30149 | 7186 |

Val/test rows are copied unchanged, so supervised and one-class runs are scored
on identical videos.

Train each domain separately, then self- and cross-test:

```powershell
D:\Python\python.exe train_3d_bcnn.py `
  --config configs\dfd_supervised_3d.yaml `
  --manifest artifacts\manifests\combined_manifest_supervised.csv

D:\Python\python.exe train_3d_bcnn.py `
  --config configs\celeb_supervised_3d.yaml `
  --manifest artifacts\manifests\combined_manifest_supervised.csv
```

```powershell
# self-test: the checkpoint's own domain, checkpoint threshold
D:\Python\python.exe evaluate_3d_bcnn.py `
  --config configs\dfd_supervised_3d.yaml `
  --manifest artifacts\manifests\combined_manifest_supervised.csv `
  --checkpoint artifactsun_dfd_supervised_3d_t8_k3\checkpointsest.pt --split test

# cross-test: the held-out domain, recalibrated operating point
D:\Python\python.exe evaluate_3d_bcnn.py `
  --config configs\dfd_supervised_3d.yaml `
  --manifest artifacts\manifests\combined_manifest_supervised.csv `
  --checkpoint artifactsun_dfd_supervised_3d_t8_k3\checkpointsest.pt `
  --split test --eval-datasets CelebDFv3 --recalibrate-threshold
```

`--recalibrate-threshold` matters because the stored threshold was calibrated
on the training domain's reals. AUROC and EER are threshold-free and stay
comparable either way; accuracy and TPR@FPR are not.

Read the two numbers as a pair: same-dataset AUROC is an upper bound that
includes method-specific artifacts, cross-dataset AUROC is the generalization
result. Supervised training deliberately gives up the data-agnostic claim, so
a large gap between them is the expected finding, not a bug.

### Objectives

`train.objective` selects what the likelihood is anchored to. The detector's
positive class is always "fake", so `score_sign` keeps the anomaly score
ranking fakes high whichever anchor is used.

| objective | trains on | target | score |
| --- | --- | --- | --- |
| `one_class_real` (default) | real only | 1 | `-posterior_loc` |
| `supervised` | real + fake | manifest label | `-posterior_loc` |
| `one_class_fake` | fake only | 1 | `+posterior_loc` |

`train.observation_likelihood` is `gaussian` (paper-faithful) or `bernoulli`.
Supervised configs use `bernoulli`, because {0,1} labels under a unit-variance
Gaussian bury the class signal in observation noise.

`data.train_balance_keys` sets the balance groups (`[dataset, class_name]` for
supervised, so 30k CelebDFv3 fakes cannot drown out 613 reals), and
`data.train_stratify_key: method` round-robins inside each group so all 22
CelebDFv3 forgery methods appear every epoch.


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

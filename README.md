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
  --checkpoint artifacts
un_dfd_supervised_3d_t8_k3\checkpointsest.pt --split test

# cross-test: the held-out domain, recalibrated operating point
D:\Python\python.exe evaluate_3d_bcnn.py `
  --config configs\dfd_supervised_3d.yaml `
  --manifest artifacts\manifests\combined_manifest_supervised.csv `
  --checkpoint artifacts
un_dfd_supervised_3d_t8_k3\checkpointsest.pt `
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


## Running on a rented remote GPU

The repository is synced with `git clone` / `git pull`, so **do not edit the
configs on the remote machine** -- a local edit turns every pull into a
conflict. Override the dataset roots with environment variables instead and the
working tree stays clean.

```bash
cp remote_env.example.sh remote_env.sh   # gitignored
$EDITOR remote_env.sh                    # point both paths at this machine
source remote_env.sh
```

`remote_env.sh` sets two variables that every entry point reads:

| variable | overrides |
| --- | --- |
| `DFD_ROOT` | `data.dataset_roots.DFD` |
| `CELEBDFV3_ROOT` | `data.dataset_roots.CelebDFv3` |
| `NUM_WORKERS` | `data.num_workers` |

Precedence is `--dataset-root NAME=PATH` > environment variable > YAML, so a
one-off run can still override a single dataset:

```bash
python train_3d_bcnn.py --config configs/dfd_supervised_3d.yaml   --manifest artifacts/manifests/combined_manifest_supervised.csv   --dataset-root DFD=/mnt/other/DFD-Kaggle --num-workers 4
```

### Choosing num_workers -- measure, do not guess

More workers is **not** automatically faster, and RAM is not the constraint.
Per-item cost on DFD (1920x1080) breaks down as:

| stage | share of per-item time |
| --- | --- |
| Haar face detection | ~93% |
| video decode (seek + read) | ~6% |
| crop / resize / normalize | <1% |

Haar detection on full-resolution frames is memory-bandwidth bound and OpenCV
already parallelises it internally, so extra worker processes contend for the
same bandwidth while multiplying resident 6 MB frame buffers. Measured on a
20-core box: 2 workers 0.686 s/item, 4 workers 0.610 s/item, 8 workers with
`cv2.setNumThreads(1)` 0.833 s/item, and 20 workers died with an OpenCV
out-of-memory error. Budget roughly 0.6-1.2 GB resident per worker for DFD.

Find the value for the actual machine:

```bash
python scripts/benchmark_loader.py   --config configs/dfd_supervised_3d.yaml   --manifest artifacts/manifests/combined_manifest_supervised.csv   --workers 2 4 8
```

It warms every worker before timing, so the figure is steady-state throughput
rather than process startup. Then `export NUM_WORKERS=<best>`.

Keep the value fixed across runs that are meant to be compared: worker count
changes which worker seeds which item, which perturbs the augmentation RNG.
Every checkpoint records the value actually used.

Verify before starting a long run:

```bash
python scripts/check_dataset_roots.py   --config configs/dfd_supervised_3d.yaml   --manifest artifacts/manifests/combined_manifest_supervised.csv --decode
```

It samples videos from every dataset/split/class group, confirms each file
exists, and (with `--decode`) opens it to check codec support. `train` and
`evaluate` also refuse to start when a root is missing.

### Manifests travel with the repository

`artifacts/manifests/*.csv` is tracked on purpose, even though the rest of
`artifacts/` is ignored. The manifests store paths **relative** to the dataset
roots (`DFD_original sequences/01__exit_phone_room.mp4`), so they are portable
as they are, and rebuilding them remotely is a hazard: `split_identities`
derives the train/val/test identities from whatever video files it finds, so a
remote copy missing a single file would silently produce a different split and
make the two machines' results incomparable. Clone, source the env file, train.

### Linux notes

- Directory names are case-sensitive there and were not on Windows. The layout
  must keep `DFD_original sequences` (with the space),
  `DFD_manipulated_sequences/DFD_manipulated_sequences`, `REAL/`, and `FAKE/`
  exactly as the manifest records them.
- `data.num_workers: 2` is conservative; raise it to match the rented CPU.
  Video decoding is the throughput bottleneck.
- Checkpoints store the training machine's roots, but evaluation always takes
  roots from the runtime config, so a Windows-trained checkpoint evaluates on
  Linux unchanged.

## Capacity diagnostic (`--overfit-subset`)

Both the one-class runs and the supervised run plateau at chance, which does
not say whether the model *cannot* learn or merely *did not*. `--overfit-subset
N` answers that in minutes instead of hours: train on N videos per dataset per
class and score those same videos, with augmentation off, clip positions
pinned, and the decoded clips cached in RAM.

```bash
python train_3d_bcnn.py   --config configs/dfd_supervised_3d.yaml   --manifest artifacts/manifests/combined_manifest_supervised.csv   --overfit-subset 6 --max-epochs 20 --run-suffix baseline
```

Outputs go to `<run_dir>_<suffix>_overfit/`, never over a real run. The reported
AUROC measures capacity, not detection performance -- it is scored on the
training videos.

### Sweeping hypotheses

Architecture and optimiser are command-line overrides, so one config serves
every variant, and the values used still reach the checkpoint:

| flag | effect |
| --- | --- |
| `--activation sigmoid\|relu\|leaky_relu\|tanh` | `model.activation` |
| `--spatial-output-size 22\|7\|4` | adaptive pool after conv3: feature_dim 15488 / 1568 / 512 |
| `--hidden-dim` | Bayesian FC1 width |
| `--optimizer sgd\|adam` | `train.optimizer` |
| `--learning-rate` | `train.learning_rate` |
| `--run-suffix` | keeps sweep variants in separate directories |

`spatial_output_size` shrinks the Bayesian head's input without altering its
structure: `FC1 -> Dropout -> FC2` is untouched. `feature_dim` is derived from
`conv_channels` and `spatial_output_size`; a config that declares a
contradicting value is rejected rather than silently ignored.

Compare the runs with:

```bash
python scripts/compare_runs.py artifacts/run_*_overfit --label A B C
```

### First result

12 DFD videos, 20 epochs, seed 42:

| variant | training-loss drop | AUROC on train |
| --- | --- | --- |
| baseline (sigmoid, 15488, SGD 1e-4) | 0.03% | 0.750 |
| ReLU only | 0.04% | 0.750 |
| spatial_output_size=7 only | 0.27% | 0.722 |
| **Adam 1e-3 only** | **5.85%** | **0.917** |
| ReLU + pool7 + Adam | 6.83% | 0.778 |

The optimiser is the binding constraint, not the activation. Sigmoid gradients
are small but consistent in direction, and Adam's per-parameter normalisation
turns them into full-size steps, whereas SGD at a fixed 1e-4 crawls. Absolute
losses are not comparable across different `feature_dim` (the KL term over FC1
scales with it); the relative drop is.

Caveats: 12 videos give AUROC a granularity of 1/36, 20 epochs is short, and
this is one seed. Treat it as a direction to test at scale, not a result.


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

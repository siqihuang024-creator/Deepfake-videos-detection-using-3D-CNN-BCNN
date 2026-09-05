# Stage A experiment log

Running record of every Stage A pre-training run and diagnostic, so a result is
not lost between sessions and a hypothesis is not re-tested by accident.

The protocol these runs implement is `docs/stage_a_protocol.html`, agreed with
the supervisor on 2026-09-03. Its settled decisions:

- Stage A trains **separately on each dataset**; no joint training for now.
- **DFD**: whole frame at native resolution, decimated — keep every other pixel
  on both axes, no interpolation anywhere. 1920x1080 becomes 960x540.
- **CelebDFv3**: tracked face box, expanded, aspect ratio preserved.
- The split stays **strictly identity-disjoint** with the donor rule (a fake
  enters a split only if its target *and* its donor are both in it). The cost
  in discarded data is accepted and not reported.
- The DFD face-crop control is deferred, then reinstated as a comparison to be
  run once the whole-frame line has a conclusion.

Manifest: `artifacts/manifests/combined_manifest_stage_a.csv`.
Label convention: **1 = real, 0 = fake** (`build_manifest.py:82`).

| dataset | split | real | fake |
|---|---|---|---|
| DFD | train | 264 | 1894 |
| DFD | val | 45 | 17 |
| DFD | test | 54 | 54 |
| DFD | excluded_donor | 0 | 1103 |
| CelebDFv3 | train | 554 | 28150 |
| CelebDFv3 | val | 166 | 5075 |
| CelebDFv3 | test | 170 | 5433 |
| CelebDFv3 | excluded_donor | 0 | 14538 |

Totals reconcile with the published datasets: DFD 363 real / 3068 fake,
CelebDFv3 890 real (590 celebrity + 300 YouTube). DFD training covers 20
identities. The donor rule discards 36% of DFD fakes and 27% of CelebDFv3's.

---

## 2026-09-04 / 05 — DFD, whole frame, decimate step 2

All runs use `configs/stage_a_dfd.yaml`, clip length 8, batch 4, Adam 1e-4,
`lr_gamma` 0.95 unless stated, ReLU + BatchNorm, `conv_channels [16,24,32]`,
`spatial_output_size` 22, `feature_dim` 15488.

### Run 1 — main run, wrong architecture
Killed after one epoch. The config named neither `activation` nor `norm`, and
`build_model` falls back to `sigmoid` / `none` (`train_3d_bcnn.py:74,77`) — the
pre-v9 architecture, whose three conv layers behind sigmoid pass almost no
gradient. Opening epoch train BCE **0.9250**, above ln 2. Fixed by writing both
keys into the config (commit `6857889`).

### Run 2 — main run, ReLU + BatchNorm
`artifacts/run_stage_a_dfd_decimate_pretrain/`. 15 epochs before being stopped.

| epoch | train BCE | val AUROC |
|---|---|---|
| 1 | 1.5400 | 0.4745 |
| 2 | 0.8420 | 0.5660 |
| 11 | 0.6980 | 0.3948 |
| 12 | 0.6988 | 0.5098 |
| 13 | 0.6964 | 0.4366 |
| 14 | 0.6963 | 0.6078 |
| 15 | 0.6943 | 0.4275 |

**Pinned at ln 2 = 0.6931 from epoch 11** — the constant-output solution.
Validation AUROC oscillates around chance on 62 videos. ~10 min/epoch.
`best.pt` is epoch 9 (val AUROC 0.6157, the noise peak).

### Diagnostic — 4-identity subset (71 videos: 55 real, 16 fake)
`--train-identities 4 --samples-per-group 64 --max-epochs 40`. Plateau is the
mean of the last 20 epochs.

| variant | plateau | min | note |
|---|---|---|---|
| baseline | **0.5241** | 0.4074 | reference |
| `--spatial-output-size 44` | 0.9572 | 0.6800 | not converged; head grew 7.9M → 31.7M with the learning rate unchanged, so this is inconclusive, not a refutation |
| `--pool-type max` | **0.5161** | 0.3824 | no difference from baseline |
| constant LR (`lr_gamma 1.0`) | — | — | see below |
| s44 + constant LR | — | — | stopped at 36/80 |

Not a strict overfitting test: clip start, stride, a 0.5 horizontal flip and
dropout 0.2 all resample every epoch, so the model never sees the same input
twice. A true fixed-clip test exists as `train_3d_bcnn.py --overfit-subset`
and has not been run.

### Constant learning rate (run C), 37 of 80 epochs

| epochs | baseline (decay 0.95) | constant 1e-4 | difference |
|---|---|---|---|
| 1-10 | 1.1469 | 1.3781 | +0.2312 |
| 11-20 | 0.7390 | 0.8935 | +0.1545 |
| 21-30 | 0.5307 | 0.7111 | +0.1804 |
| 31-37 | 0.5225 | 0.6134 | +0.0908 |

Constant LR is **worse at every stage**, by 3-7 standard errors of the block
mean. The decay was helping, not starving the run. **Keep `lr_gamma: 0.95`.**

Note the design flaw in run D: it was meant to give `spatial_output_size 44` a
fairer learning rate, but a *constant* 1e-4 leaves the late-run rate **higher**
than the decayed schedule it was compared against, so it never tested the
hypothesis it was built for.

---

## Feature diagnostics

`scripts/feature_diagnostics.py`. Stage A's head is
`fc1 -> dropout -> out` with **no activation between the linear layers**
(`model.py:253-255`), so it composes to a single linear functional — Stage A is
already a linear probe, and its success criterion is whether a linear
classifier separates real from fake on **unseen identities**.

Two instrument bugs found and fixed before any number was trusted:

1. A probe fitted and scored on the same videos is vacuous at 60 samples in
   15488 dimensions — an untrained extractor scored **0.9422**. Replaced by
   ridge regression solved in the dual with folds split by identity. Validated
   on synthetic data: pure noise gives 0.50 held out against 1.00 in sample;
   injected signal gives 1.00 held out.
2. `spread_over_norm` divided a per-dimension standard deviation by a
   whole-vector L2 norm, understating it by sqrt(15488) and inventing an
   apparent collapse. Corrected to `relative_variation`.
3. Sampling took the first 30 reals and first 30 fakes from a sorted manifest,
   which drew **two** identities and left the fold split with one class to
   train on. Replaced by round-robin over identities: the same 60-video budget
   now covers all 20 DFD training identities with both classes in each.

### Results

| | v9 | v9arch | run 2 |
|---|---|---|---|
| dataset | CelebDFv3 | **DFD** | DFD |
| input | face crop | **face crop** | whole frame |
| checkpoint | `..._celeb_..._v9_bn` ep 38 | `..._dfd_..._v9arch` ep 4 | `..._decimate_pretrain` ep 9 |
| recorded end-to-end val AUROC | 0.7773 | 0.6083 | 0.6157 |
| identities in the probe | 30 | **4** | 20 |
| **held-out probe, trained** | **0.7322** | **0.5956** | **0.4778** |
| held-out probe, random init | 0.5422 | 0.5833 | 0.4878 |
| **change from training** | **+0.1900** | **+0.0122** | **-0.0100** |
| separation_ratio trained / random | 0.684 / 0.486 | 0.239 / 0.206 | 0.240 / 0.226 |
| relative_variation | 0.937 | 0.943 | 0.970 |

The middle column was added to break the confound in the first comparison: v9
differed from the whole-frame run in **both** dataset and input mode. DFD put
through the same face-crop pipeline that works on CelebDFv3 moves the probe by
+0.012 -- an order of magnitude below v9's +0.190, and the same size as the
whole-frame run's -0.010. **The failure follows the dataset, not the input
mode**, which matches the cross-dataset matrix: only CelebDFv3 self-test beats
chance.

Two caveats on that column. The checkpoint is epoch 4, so how long the run
actually trained has to be confirmed before it counts as a fair test. And its
validation split holds only 4 identities, which makes the fold structure thin
and the interval wide -- "no detectable signal", not "zero".

The probe reads **0.7322 on v9 against its true 0.7773**, so it tracks real
performance rather than merely reporting a positive. On DFD whole-frame it
reads chance, and training moved it by -0.01: **that run learned nothing**.

Interval note: 30 vs 30 gives a Hanley–McNeil standard error near 0.075, so
0.478 carries a 95% interval of about [0.33, 0.63]. The honest statement is
"no detectable signal", not "exactly zero".

---

## Hypotheses settled so far

| hypothesis | status | evidence |
|---|---|---|
| Learning-rate decay starved the run | **eliminated** | constant LR worse at every stage |
| Feature collapse | **eliminated** | `relative_variation` 0.97 |
| Features separable, optimisation fails | **eliminated** | held-out probe at chance |
| The representation carries no class information | **established** | trained ≈ random init on both metrics |
| Whole frame vs DFD-the-dataset | **resolved: the dataset** | DFD + face crop moves the probe +0.012 against v9's +0.190 |
| Identity scarcity explains DFD | eliminated earlier | v11 cut CelebDFv3 to 20 identities and lost 0.03 |
| Face detection explains DFD | eliminated earlier | the Haar/MTCNN/RetinaFace comparison |
| Clip-level label noise | **open** | a video-level "fake" label on an 8-frame window with no visible manipulation puts a floor on BCE |
| Decode corruption | **open** | run 2's log carried `moov atom not found`, `partial file`, `Invalid NAL unit size` and 30 s read timeouts while reporting `skipped=0`; the loader hides failures by repeating the previous frame (`data.py:358-365`). `scripts/scan_decode_health.py` measures the per-class rate and has not been run |

## Code facts worth not rediscovering

- `pretrain_extractor.py` takes `--max-epochs`, not `--epochs`, and its default
  of 30 silently overrides the config's `epochs`.
- `run_suffix` (default `pretrain`) is appended to the run directory name, so
  `artifacts/run_stage_a_dfd_decimate/` in a config becomes
  `artifacts/run_stage_a_dfd_decimate_pretrain/` on disk.
- `early_stopping_patience` is **not implemented** in `pretrain_extractor.py`;
  a run always completes `--max-epochs`.
- Checkpoint selection is already by validation AUROC, not by training loss.
- `pretrain_extractor.py` saves the extractor under `extractor`;
  `train_3d_bcnn.py` saves it under `feature_extractor`.
- Dataset roots come from `DFD_ROOT` / `CELEBDFV3_ROOT`; `source remote_env.sh`
  is required or the configs' Windows paths are used and the run aborts.

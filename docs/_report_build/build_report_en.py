# -*- coding: utf-8 -*-
"""English edition of the experiment report.

Same figures, same tables, same numbers as docs/experiment_report_zh.html; the
prose and every text node inside the carried-over figures are translated.
Terminology follows the deepfake-detection literature: dataset, split, feature
extractor, linear probe, identity-disjoint, anomaly score.
"""
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SA = REPO / "scratch_archive"
sys.path.insert(0, str(SA))
from svg_en import SVG_TEXT, SVG_TWEAKS  # noqa: E402

STYLE = SA.joinpath("_style.txt").read_text(encoding="utf-8")
SCRIPT = SA.joinpath("_script.txt").read_text(encoding="utf-8")
# the hover tooltips carry their own labels
for _zh, _en in (("训练 BCE", "train BCE"), ("验证 AUROC", "val AUROC"),
                 ("验证 AUC", "val AUC")):
    SCRIPT = SCRIPT.replace(_zh, _en)
BOXES = SA.joinpath("_figboxes.txt").read_text(encoding="utf-8").split("\n<<<SPLIT>>>\n")
assert len(BOXES) == 8


def translate(block):
    for old, new in SVG_TWEAKS.items():
        block = block.replace(old, new)
    for old, new in sorted(SVG_TEXT.items(), key=lambda kv: -len(kv[0])):
        block = block.replace(old, new)
    return block


BOXES = [translate(b) for b in BOXES]
EXP1_LOSS, EXP1_AUC, EXP1_ROC, TSNE, EXP3_A, EXP3_B, EXP3_ROC, EXP3_METHOD = range(8)


def fig(box, number, caption):
    return ('<figure>\n  ' + BOXES[box].rstrip() +
            '\n  <figcaption><b>Figure {}.</b> {}</figcaption>\n</figure>'.format(number, caption))


ARCH = '''<figure>
  <div class="figbox">
    <svg viewBox="0 0 720 512" role="img" aria-label="Architecture and two-stage procedure: a three-stage 3D convolutional extractor produces a 15488-dimensional feature; Stage A trains that extractor behind a deterministic head, then hands the frozen weights to Stage B, which trains only a Bayesian head and outputs an anomaly score.">
      <defs>
        <marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="currentColor"/>
        </marker>
        <marker id="ahk" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="var(--s2)"/>
        </marker>
      </defs>

      <text x="20" y="24" font-family="Source Serif 4, serif" font-size="12.5" font-weight="600" fill="currentColor">Feature extractor</text>
      <text x="700" y="24" font-family="IBM Plex Mono, monospace" font-size="9.5" fill="var(--faint)" text-anchor="end">trained in Stage A · frozen in Stage B</text>

      <g fill="none" stroke="currentColor" stroke-width="1.2" opacity="0.85">
        <rect x="20" y="36" width="88" height="58" rx="3"/>
        <rect x="120" y="36" width="88" height="58" rx="3"/>
        <rect x="220" y="36" width="88" height="58" rx="3"/>
        <rect x="320" y="36" width="88" height="58" rx="3"/>
        <rect x="420" y="36" width="88" height="58" rx="3"/>
        <rect x="520" y="36" width="88" height="58" rx="3"/>
        <rect x="620" y="36" width="88" height="58" rx="3"/>
      </g>
      <g font-family="Source Serif 4, serif" font-size="11" fill="currentColor" text-anchor="middle">
        <text x="64" y="59">Input clip</text><text x="164" y="59">Conv stage 1</text><text x="264" y="59">Conv stage 2</text>
        <text x="364" y="59">Conv stage 3</text><text x="464" y="59">Temporal mean</text><text x="564" y="59">Adaptive pool</text>
        <text x="664" y="59">Feature vector</text>
      </g>
      <g font-family="IBM Plex Mono, monospace" font-size="9" fill="var(--muted)" text-anchor="middle">
        <text x="64" y="76">3×8×256×256</text><text x="164" y="76">16×8×125×125</text><text x="264" y="76">24×8×59×59</text>
        <text x="364" y="76">32×8×26×26</text><text x="464" y="76">32×26×26</text><text x="564" y="76">32×22×22</text>
        <text x="664" y="76">15488</text>
      </g>
      <g font-family="IBM Plex Mono, monospace" font-size="8.5" fill="var(--faint)" text-anchor="middle">
        <text x="64" y="89">face crop</text><text x="164" y="89">3→16</text><text x="264" y="89">16→24</text>
        <text x="364" y="89">24→32</text><text x="464" y="89">over 8 steps</text><text x="564" y="89">+BatchNorm2d</text>
        <text x="664" y="89">flatten</text>
      </g>
      <g stroke="currentColor" stroke-width="1.2" marker-end="url(#ah)" opacity="0.7">
        <line x1="108" y1="65" x2="118" y2="65"/><line x1="208" y1="65" x2="218" y2="65"/>
        <line x1="308" y1="65" x2="318" y2="65"/><line x1="408" y1="65" x2="418" y2="65"/>
        <line x1="508" y1="65" x2="518" y2="65"/><line x1="608" y1="65" x2="618" y2="65"/>
      </g>
      <path d="M 120 104 L 120 112 L 408 112 L 408 104" fill="none" stroke="var(--faint)" stroke-width="1"/>
      <text x="264" y="128" font-family="IBM Plex Mono, monospace" font-size="9.5" fill="var(--faint)" text-anchor="middle">each stage: Conv3d k=3×5×5 → AvgPool3d 4×4 stride 2 → BatchNorm3d → ReLU</text>

      <path d="M 664 94 L 664 146 L 366 146" fill="none" stroke="currentColor" stroke-width="1.2" opacity="0.7"/>
      <line x1="366" y1="146" x2="366" y2="168" stroke="currentColor" stroke-width="1.2" marker-end="url(#ah)" opacity="0.7"/>

      <rect x="60" y="172" width="600" height="116" rx="3" fill="none" stroke="var(--s1)" stroke-width="1.4"/>
      <text x="78" y="194" font-family="Source Serif 4, serif" font-size="12.5" font-weight="600" fill="var(--s1)">Stage A · supervised pre-training</text>
      <text x="642" y="194" font-family="IBM Plex Mono, monospace" font-size="9.5" fill="var(--s1)" text-anchor="end">extractor trainable</text>
      <g font-family="IBM Plex Mono, monospace" font-size="10" fill="currentColor">
        <text x="78" y="220">Linear  15488 → 512</text>
        <text x="78" y="238">Dropout p = 0.2</text>
        <text x="78" y="256">Linear  512 → 1</text>
      </g>
      <text x="78" y="277" font-family="Source Serif 4, serif" font-size="10.5" fill="var(--faint)">no activation between the two layers</text>
      <g font-family="Source Serif 4, serif" font-size="11" fill="var(--muted)">
        <text x="330" y="220">Objective: binary cross-entropy, real/fake labels</text>
        <text x="330" y="238">Adam, lr 1×10⁻⁴, exponential decay γ = 0.95</text>
        <text x="330" y="256">1000 clips sampled per class per epoch</text>
        <text x="330" y="277">Checkpoint: highest validation AUROC</text>
      </g>

      <line x1="366" y1="288" x2="366" y2="334" stroke="var(--s2)" stroke-width="2" marker-end="url(#ahk)"/>
      <text x="382" y="303" font-family="Source Serif 4, serif" font-size="11.5" font-weight="600" fill="var(--s2)">extractor weights kept and frozen</text>
      <text x="382" y="320" font-family="Source Serif 4, serif" font-size="11" fill="var(--muted)">BatchNorm statistics included · Stage A head discarded</text>
      <text x="350" y="303" font-family="Source Serif 4, serif" font-size="11" fill="var(--faint)" text-anchor="end">hand-off</text>

      <rect x="60" y="338" width="600" height="116" rx="3" fill="none" stroke="var(--s3)" stroke-width="1.4"/>
      <text x="78" y="360" font-family="Source Serif 4, serif" font-size="12.5" font-weight="600" fill="var(--s3)">Stage B · one-class anomaly detection</text>
      <text x="642" y="360" font-family="IBM Plex Mono, monospace" font-size="9.5" fill="var(--s3)" text-anchor="end">extractor frozen</text>
      <g font-family="IBM Plex Mono, monospace" font-size="10" fill="currentColor">
        <text x="78" y="386">FC1 ~ N(μ, σ)  15488 → 512</text>
        <text x="78" y="404">Dropout p = 0.2</text>
        <text x="78" y="422">FC2 ~ N(μ, σ)  512 → 1</text>
      </g>
      <text x="78" y="443" font-family="Source Serif 4, serif" font-size="10.5" fill="var(--faint)">prior N(0, 0.1²) · KL weight 10⁻³</text>
      <g font-family="Source Serif 4, serif" font-size="11" fill="var(--muted)">
        <text x="330" y="386">Objective: TraceGraph ELBO, Pyro SVI</text>
        <text x="330" y="404">SGD, trained on real videos only</text>
        <text x="330" y="422">Anomaly score = negative posterior predictive</text>
        <text x="330" y="443">Checkpoint: highest validation macro AUROC</text>
      </g>

      <line x1="366" y1="454" x2="366" y2="480" stroke="currentColor" stroke-width="1.2" marker-end="url(#ah)" opacity="0.7"/>
      <text x="366" y="500" font-family="Source Serif 4, serif" font-size="11.5" fill="currentColor" text-anchor="middle">video-level anomaly score → AUROC · EER · TPR@5%FPR</text>
    </svg>
  </div>
  <figcaption><b>Figure 1.</b> Network structure and the two-stage procedure, executed top to bottom. The feature extractor is three stages of 3D convolution; within each stage the order is convolution, average pooling, batch normalisation, activation. After the third stage the tensor is <b>averaged along the temporal axis</b> &mdash; that stage emits one feature map per time step, eight in all, and the mean collapses them into one, so an 8-frame clip yields exactly one feature vector. The result is adaptively average-pooled to 22×22, passed through one further BatchNorm2d and flattened to 15488 dimensions. Tensor sizes are given for a 256×256 face-crop input; a 540×960 whole-frame input follows the same path to the same 15488 dimensions, the size difference being absorbed by the adaptive pooling. <b>Stage A</b> trains the extractor together with a deterministic head under real/fake labels; that head is then <b>discarded and the extractor weights are frozen and handed to Stage B</b>, which trains only a mean-field Bayesian head of identical shape, on real videos alone, and emits a video-level anomaly score.</figcaption>
</figure>'''

DOC = '''<title>3D-CNN Bayesian Detection Experiments</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=Noto+Serif+SC:wght@400;500;700&family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;1,8..60,400&display=swap">
''' + STYLE + '''

<div class="page">

<header>
  <h1>Deepfake Video Detection with a 3D-CNN Feature Extractor and a Bayesian One-Class Head</h1>
  <p class="sub">Experimental method, the purpose of each step, and the measurements obtained</p>
  <p class="meta">Experiments run 2026-09-04 — 2026-09-07 · 45 runs · 2180 epochs</p>
</header>

<div class="abstract">
  <div class="lead">Scope of this document</div>
  <p>This document records the <b>method, the purpose of each step, and the measurements</b> for two experiments: the network structure, the split protocol, the preprocessing, the training configuration, and the curves, tables and metrics each step produced. It does not interpret those measurements or draw conclusions from them. Every number is computed from the files under <span class="m">results/</span> rather than transcribed.</p>
</div>

<h2><span class="n">1</span>Architecture</h2>

<p>The feature extractor follows the fine-to-coarse structure of Leyva et al. (2024), inflated along the temporal axis to 3D. Two classification heads of identical shape sit behind it, one for each training stage.</p>

''' + ARCH + '''

<div class="tw">
<table>
  <caption><b>Table 1.</b> Layer-by-layer output sizes and parameter counts for a 3×8×256×256 face-crop input. The 3D convolutions pad the temporal dimension by 1, so the clip length stays at 8 throughout; the temporal receptive field is 7 frames.</caption>
  <thead><tr><th>Layer</th><th>Operator</th><th class="n">Output size</th><th class="n">Parameters</th></tr></thead>
  <tbody>
    <tr><td>Input</td><td>RGB clip</td><td class="n">3×8×256×256</td><td class="n">—</td></tr>
    <tr><td>Conv stage 1</td><td>Conv3d 3→16, k=3×5×5</td><td class="n">16×8×252×252</td><td class="n">3,616</td></tr>
    <tr><td></td><td>AvgPool3d 1×4×4, stride 1×2×2</td><td class="n">16×8×125×125</td><td class="n">—</td></tr>
    <tr><td></td><td>BatchNorm3d → ReLU</td><td class="n">16×8×125×125</td><td class="n">32</td></tr>
    <tr><td>Conv stage 2</td><td>Conv3d 16→24, k=3×5×5</td><td class="n">24×8×121×121</td><td class="n">28,824</td></tr>
    <tr><td></td><td>AvgPool3d → BatchNorm3d → ReLU</td><td class="n">24×8×59×59</td><td class="n">48</td></tr>
    <tr><td>Conv stage 3</td><td>Conv3d 24→32, k=3×5×5</td><td class="n">32×8×55×55</td><td class="n">57,632</td></tr>
    <tr><td></td><td>AvgPool3d → BatchNorm3d → ReLU</td><td class="n">32×8×26×26</td><td class="n">64</td></tr>
    <tr><td>Temporal pooling</td><td>mean along the temporal axis (8 steps into 1)</td><td class="n">32×26×26</td><td class="n">—</td></tr>
    <tr><td>Spatial pooling</td><td>AdaptiveAvgPool2d(22)</td><td class="n">32×22×22</td><td class="n">—</td></tr>
    <tr><td>Output norm</td><td>BatchNorm2d(32) → flatten</td><td class="n">15488</td><td class="n">64</td></tr>
    <tr class="head"><td>Extractor total</td><td></td><td class="n">15488</td><td class="n">90,280</td></tr>
    <tr><td>Stage A head</td><td>Linear 15488→512 → Dropout → Linear 512→1</td><td class="n">1</td><td class="n">7,930,881</td></tr>
    <tr><td>Stage B head</td><td>same shape, mean-field Bayesian</td><td class="n">1</td><td class="n">7,930,881 × 2</td></tr>
  </tbody>
</table>
</div>

<h2><span class="n">2</span>Method</h2>

<h3>2.1 Split protocol</h3>

<p><b>Purpose:</b> to ensure that no identity in the test set appears in training, and that the source real video behind a forgery is not in the training set either.</p>

<p>Splits are identity-disjoint, with an additional <b>donor constraint</b>: a forged video enters a subset only if <em>both</em> its target identity and its donor identity fall in that subset. DFD forgeries are named <span class="m">&lt;target&gt;_&lt;source&gt;__&lt;scene&gt;__&lt;id&gt;</span>, so both identities are recoverable from the filename. Splitting by video alone would leave 73.2% of the DFD test forgeries with their own source real video in the training set. The constraint discards 1103 forged videos in DFD (36%) and 14538 in CelebDF++ (27.3%). Table 2 reports how much overlap a target-only split leaves behind.</p>

<div class="tw">
<table>
  <caption><b>Table 2.</b> Donor overlap left behind by a target-only split. The first column is the number of forgeries a subset would receive if each one simply followed its target identity. <b>Not every manipulation involves a second identity</b> — CelebDF++'s TalkingFace family drives a single identity from audio and has no donor face, so the donor rule never applies to it, which is why fewer than six in ten of its forgeries carry a donor at all. The donor-in-train percentage is taken over the forgeries that carry a donor; over every target-only forgery it is 37.8% for the CelebDF++ test subset. The last column is the protocol's <b>cost in data, not a leakage rate</b>: some discarded forgeries had their donor in validation rather than training.</caption>
  <thead><tr><th>Dataset · subset</th><th class="n">Target-only forgeries</th><th class="n">Carrying a donor</th><th class="n">Donor seen in training</th><th class="n">Discarded by the rule</th></tr></thead>
  <tbody>
    <tr><td>DFD · validation</td><td class="n">250</td><td class="n">250 (100%)</td><td class="n">199 (79.6%)</td><td class="n">233 (93.2%)</td></tr>
    <tr class="head"><td>DFD · test</td><td class="n">406</td><td class="n">406 (100%)</td><td class="n">313 (77.1%)</td><td class="n">352 (86.7%)</td></tr>
    <tr><td>CelebDF++ · validation</td><td class="n">8462</td><td class="n">4311 (50.9%)</td><td class="n">2554 (59.2%)</td><td class="n">3387 (40.0%)</td></tr>
    <tr class="head"><td>CelebDF++ · test</td><td class="n">10255</td><td class="n">5975 (58.3%)</td><td class="n">3879 (64.9%)</td><td class="n">4822 (47.0%)</td></tr>
  </tbody>
</table>
</div>

<h3>2.2 Preprocessing</h3>

<p><b>Purpose:</b> to turn variable-length videos into fixed-length clips. The two input modes are the independent variable of the experimental design.</p>

<p><b>Whole-frame mode (experiment 1):</b> native resolution is kept and every second pixel is taken along both spatial axes (<span class="m">frame[::2, ::2]</span>), so 1920×1080 becomes 960×540 at an unchanged aspect ratio. The operation involves no interpolation.</p>

<p><b>Face-crop mode (experiment 3):</b> a dlib HOG detector locates the face, 81 landmarks are regressed, the frame is rotated about the box centre so that the eye line is level, the detection box is expanded by a factor of ×2.0, cropped, and resized to 256×256, with out-of-bounds regions filled by edge replication. Detection is run once offline before training and cached per video (<span class="m">frame_indices</span>, <span class="m">boxes</span>, <span class="m">landmarks</span>, <span class="m">detect_stride</span>); training and evaluation read only those coordinates. Detection stride is 4 frames, with intermediate frames linearly interpolated.</p>

<p>Clip length is 8 frames in both modes.</p>

<h3>2.3 Two-stage training</h3>

<p><b>Purpose of Stage A:</b> to train the feature extractor to separate real from fake under real/fake labels, with deterministic gradients; the classification head is discarded afterwards and only the extractor weights are carried forward.</p>

<p><b>Purpose of Stage B:</b> to freeze the extractor and train a Bayesian head on real videos alone, so that it assigns a high posterior to real videos and a low one to inputs that depart from that distribution. The anomaly score is the negative of the posterior predictive.</p>

<div class="tw">
<table>
  <caption><b>Table 3.</b> Training configuration for the two stages. Both experiments share these settings; they differ only in dataset, input mode, and early-stopping patience (Table 4).</caption>
  <thead><tr><th>Item</th><th>Stage A</th><th>Stage B</th></tr></thead>
  <tbody>
    <tr><td>Objective</td><td>binary cross-entropy</td><td>TraceGraph ELBO (Pyro SVI)</td></tr>
    <tr><td>Optimiser</td><td>Adam</td><td>SGD</td></tr>
    <tr><td>Learning rate</td><td>1×10⁻⁴, exponential decay γ = 0.95</td><td>1×10⁻⁴, exponential decay γ = 0.95</td></tr>
    <tr><td>Training sample</td><td>both classes, 1000 clips per class per epoch</td><td>real videos only</td></tr>
    <tr><td>Batch size</td><td>4</td><td>4</td></tr>
    <tr><td>Epoch budget</td><td>60</td><td>50</td></tr>
    <tr><td>Extractor</td><td>trainable</td><td>frozen, BatchNorm statistics included</td></tr>
    <tr><td>Checkpoint selection</td><td>highest validation AUROC</td><td>highest validation macro AUROC</td></tr>
  </tbody>
</table>
</div>

<h3>2.4 Evaluation metrics</h3>

<p><b>Purpose:</b> to report readings on the held-out test subset that are comparable with the literature.</p>

<p>Each video contributes 8 evenly spaced clips and its score is their mean, so every metric is <b>video-level</b>. The primary metric is <b>AUROC</b>; AP, accuracy, balanced accuracy, EER and TPR@5%FPR are reported alongside it. The decision threshold is calibrated on the real videos of the validation split at a 5% false positive rate and stored with the checkpoint.</p>

<p>Confidence intervals for AUROC come from an <b>identity-clustered bootstrap</b> (2000 draws, resampling identities rather than videos). On CelebDF++ one real clip yields a median of 84 forgeries that share its background, wardrobe, lighting and camera motion; resampling videos would treat those as independent observations.</p>

<h3>2.5 Experimental design</h3>

<div class="tw">
<table>
  <caption><b>Table 4.</b> The two experiments recorded here. Network structure, input resolution, clip length, optimiser and learning-rate schedule are identical between them.</caption>
  <thead><tr><th>Experiment</th><th>Dataset</th><th>Input mode</th><th class="n">Stage A patience</th><th>Status</th></tr></thead>
  <tbody>
    <tr><td>Experiment 1</td><td>DFD</td><td>whole frame, pixel decimation</td><td class="n">12</td><td>complete</td></tr>
    <tr><td>Experiment 3</td><td>CelebDF++</td><td>face crop ×2.0</td><td class="n">8</td><td>complete</td></tr>
  </tbody>
</table>
</div>

<h2><span class="n">3</span>Experiment 1: DFD, whole-frame input</h2>

<p><b>Purpose:</b> to measure the end-to-end performance of the two-stage detector on DFD without any face detection.</p>

<div class="tw">
<table>
  <caption><b>Table 5.</b> DFD split statistics. Label convention: 1 = real, 0 = fake. DFD contains 363 original videos and 3068 forgeries, recorded by 28 actors.</caption>
  <thead><tr><th>Subset</th><th class="n">Real</th><th class="n">Fake</th><th class="n">Identities</th><th>Use</th></tr></thead>
  <tbody>
    <tr><td>Train</td><td class="n">264</td><td class="n">1894</td><td class="n">20</td><td>Stage A / Stage B</td></tr>
    <tr><td>Validation</td><td class="n">45</td><td class="n">17</td><td class="n">4</td><td>checkpoint selection</td></tr>
    <tr><td>Test</td><td class="n">54</td><td class="n">54</td><td class="n">4</td><td>final evaluation, held out throughout</td></tr>
    <tr><td class="dim">Discarded by donor constraint</td><td class="n dim">0</td><td class="n dim">1103</td><td class="n dim">—</td><td class="dim">used in no stage</td></tr>
  </tbody>
</table>
</div>

''' + fig(EXP1_LOSS, 2,
          'Stage A training cross-entropy over 60 epochs, from 1.5778 to 0.6579. The dashed line marks <b>ln 2 = 0.6931</b>, the cross-entropy of a trivial predictor that outputs 0.5 for every input on class-balanced data; Stage A samples 1000 clips per class per epoch, so the classes are balanced and the reference applies. 46 of the 60 epochs sit below it. Hover to read any epoch.') + '''

''' + fig(EXP1_AUC, 3,
          'Validation AUROC for the same run, on a validation subset of 62 videos across 4 identities. First epoch 0.4928, last 0.4248, highest 0.6039 (epoch 14), mean 0.4477 over the run; from epoch 24 onward, 1 of 37 epochs exceeds 0.50.') + '''

<div class="tw">
<table>
  <caption><b>Table 6.</b> Experiment 1 test results, 54 real / 54 fake, video-level. The threshold is calibrated on the validation split at 5% FPR. Stage B early-stopped at epoch 19, with a best validation AUROC of 0.6627 and a run mean of 0.6259.</caption>
  <thead><tr><th>Metric</th><th class="n">Value</th><th>Note</th></tr></thead>
  <tbody>
    <tr class="head"><td>Video-level AUROC</td><td class="n key">0.4877</td><td>identity-clustered 95% bootstrap interval <span class="m">[0.458, 0.531]</span>, 4 identity clusters</td></tr>
    <tr><td>Average precision</td><td class="n">0.4883</td><td>balanced classes, base rate 0.50</td></tr>
    <tr><td>Accuracy</td><td class="n">0.4815</td><td>at the calibrated threshold</td></tr>
    <tr><td>Balanced accuracy</td><td class="n">0.4815</td><td></td></tr>
    <tr><td>EER</td><td class="n">0.5093</td><td></td></tr>
    <tr><td>TPR @ 5% FPR</td><td class="n">0.0000</td><td></td></tr>
    <tr><td>Confusion matrix</td><td class="n">[[52, 2], [54, 0]]</td><td>[real: TN, FP] / [fake: FN, TP]</td></tr>
    <tr><td>Mean posterior, real / fake</td><td class="n">0.8330 / 0.8478</td><td></td></tr>
  </tbody>
</table>
</div>

''' + fig(EXP1_ROC, 4,
          'Experiment 1 test set, drawn from 108 per-video anomaly scores. <b>(a)</b> Score distributions; real mean −0.8330, fake mean −0.8478. The dashed black line is the calibrated threshold. <b>(b)</b> ROC curve, AUC 0.4877, identity-clustered 95% bootstrap interval [0.458, 0.531].') + '''

<h2><span class="n">4</span>Experiment 3: CelebDF++, face-crop input</h2>

<p><b>Purpose:</b> to measure the same pipeline on a 2025 benchmark covering 22 forgery methods. The methods in CelebDF++ fall into three families: face swapping (FaceSwap), face reenactment (FaceReenact), and audio-driven talking-head generation (TalkingFace).</p>

<div class="tw">
<table>
  <caption><b>Table 7.</b> CelebDF++ split statistics. The test subset has a real-to-fake ratio of roughly 1:32.</caption>
  <thead><tr><th>Subset</th><th class="n">Real</th><th class="n">Fake</th><th class="n">Identities</th><th>Use</th></tr></thead>
  <tbody>
    <tr><td>Train</td><td class="n">554</td><td class="n">28150</td><td class="n">245</td><td>Stage A / Stage B</td></tr>
    <tr><td>Validation</td><td class="n">166</td><td class="n">5075</td><td class="n">57</td><td>checkpoint selection</td></tr>
    <tr><td>Test</td><td class="n">170</td><td class="n">5433</td><td class="n">57</td><td>final evaluation, held out throughout</td></tr>
    <tr><td class="dim">Discarded by donor constraint</td><td class="n dim">0</td><td class="n dim">14538</td><td class="n dim">—</td><td class="dim">used in no stage</td></tr>
  </tbody>
</table>
</div>

<p>Face detection rates: train 99.3% real / 99.2% fake; validation 99.2% / 99.4%; test 98.1% / 99.1%. Median detection-box width on the test split: 138 px for real videos, 156 px for forgeries.</p>

''' + fig(EXP3_A, 5,
          'Experiment 3, Stage A, over 60 epochs. <b>(a)</b> Training cross-entropy from 1.5420 to 0.5618, minimum 0.5279; the dashed ln 2 line carries the same meaning as in Figure 2. <b>(b)</b> Validation AUROC from 0.5864 to 0.6918, highest 0.6937 (epoch 57), mean 0.6572 over the run. The validation subset holds 5241 videos across 57 identities.') + '''

''' + fig(EXP3_B, 6,
          'Experiment 3, Stage B, over 17 epochs (early-stopping patience 8). Validation AUROC peaks at 0.6569 (epoch 9), with a run mean of 0.6198 and a last epoch of 0.6255. The reading of the same checkpoint on the test set is marked separately.') + '''

<div class="tw">
<table>
  <caption><b>Table 8.</b> Experiment 3 test results, 170 real / 5433 fake, of which 3 videos were skipped for a missing detection or a decode failure, leaving 169 real / 5431 fake scored. The threshold is the one stored with the checkpoint. The base rate of the test subset is 0.9698.</caption>
  <thead><tr><th>Metric</th><th class="n">Value</th><th>Note</th></tr></thead>
  <tbody>
    <tr class="head"><td>Video-level AUROC</td><td class="n key">0.5284</td><td>identity-clustered 95% bootstrap interval <span class="m">[0.502, 0.572]</span>, 57 identity clusters</td></tr>
    <tr><td>EER</td><td class="n">0.5019</td><td></td></tr>
    <tr><td>Balanced accuracy</td><td class="n">0.5094</td><td></td></tr>
    <tr><td>TPR @ 5% FPR</td><td class="n">0.0755</td><td></td></tr>
    <tr><td>Accuracy</td><td class="n">0.1041</td><td>base rate 0.9698</td></tr>
    <tr><td>Average precision</td><td class="n">0.9733</td><td>base rate 0.9698</td></tr>
    <tr><td>Confusion matrix</td><td class="n">[[159, 10], [5007, 424]]</td><td>[real: TN, FP] / [fake: FN, TP]</td></tr>
    <tr><td>Mean anomaly score, real / fake</td><td class="n">−0.8564 / −0.8173</td><td>standard deviations 0.2716 / 0.2261</td></tr>
    <tr><td>Mean posterior predictive s.d.</td><td class="n">0.2086</td><td>16 Monte Carlo samples</td></tr>
  </tbody>
</table>
</div>

''' + fig(EXP3_ROC, 7,
          'Experiment 3 test set, drawn from 5600 per-video anomaly scores. <b>(a)</b> Score distributions; because the real-to-fake ratio is 1:32 the vertical axis is density rather than count. <b>(b)</b> ROC curve, AUC 0.5284, recomputed independently from the same score file and in agreement with Table 8.') + '''

''' + fig(EXP3_METHOD, 8,
          'Test-set AUROC split by forgery method, 22 methods arranged by family. Each TalkingFace method contributes 590–625 forged videos, each FaceReenact method 67, and each FaceSwap method 69 (201 for Celeb-DF-v2). All three families are scored against the same 169 real videos.') + '''

<h2><span class="n">5</span>Feature representation measurements</h2>

<p><b>Purpose:</b> what Stage A delivers is an extractor, not a classifier, and its classification head carries no activation between its two linear layers, which makes it a linear functional. The question "did Stage A succeed?" is therefore measured directly: <b>can a linear classifier separate real from fake given the extractor output, on identities it has never seen?</b></p>

<p><b>Method:</b> the extractor is frozen and its features are fitted with ridge regression (closed-form dual solution) under five-fold cross-validation <em>split by identity</em>, and the held-out-fold AUROC is reported. The same measurement is repeated on randomly initialised extractors to establish the range a matched but untrained network produces — this reference is not 0.5, because random convolutions and pooling already encode low-level image statistics. The t-SNE projection reduces the features to 30 dimensions by PCA first; on that plane a 5-NN classifier is run leave-one-identity-out.</p>

''' + fig(TSNE, 9,
          't-SNE projection of extractor features for 60 DFD training videos (30 real / 30 fake, 20 identities). t-SNE maps the 15488-dimensional feature of each video to a point on a plane, placing vectors that were close in the original space close on the plane; it is used for inspection only and enters no calculation. Green dots are real videos, orange crosses are forgeries. The leave-one-identity-out 5-NN accuracy on this plane is 0.375.') + '''

<div class="tw">
<table>
  <caption><b>Table 9.</b> Held-out-identity linear probe readings, 60 videos per measurement. <span class="m">Random init</span> is a matched but untrained extractor, one draw per row; <span class="m">difference</span> is the gap between the two. Under the donor-safe protocol this work has no extractor that can serve as a positive control; see the note below.</caption>
  <thead><tr><th>Measured on</th><th class="n">Checkpoint epoch</th><th class="n">Random init</th><th class="n">Trained</th><th class="n">Difference</th></tr></thead>
  <tbody>
    <tr class="head"><td>Experiment 1 · DFD, whole frame</td><td class="n">60</td><td class="n">0.4900</td><td class="n">0.5033</td><td class="n">+0.0133</td></tr>
    <tr class="head"><td>Experiment 3 · CelebDF++, face crop</td><td class="n">57</td><td class="n">0.7552</td><td class="n">0.7149</td><td class="n">−0.0402</td></tr>
  </tbody>
</table>
</div>


<div class="note">
  <b>Historical diagnostic under a pre-correction split.</b> One further extractor reads 0.7322 on the frozen probe against a random-init reference of 0.5778 (a difference of +0.1544), with an end-to-end test AUROC of 0.7773. <b>It was trained before the donor-safe constraint was enforced on the CelebDF++ split</b>, so its training data contained donor identities that later fell in the test split; the same configuration produces no such difference on the donor-safe split, which is experiment 3. It shows that this probe can read a clear difference when one is present, and why the donor constraint matters, but it is <b>not evidence that supervised pre-training produces transferable representations under the clean protocol</b>, and so is not listed in the table above.
</div>

<h2><span class="n">6</span>Metadata baselines</h2>

<p><b>Purpose:</b> to measure how far the two classes can be separated from container and encoding metadata alone, without decoding a single pixel. Such information carries no forgery evidence, so what it yields is a floor for how separable the dataset itself is.</p>

<p><b>Method:</b> each video is reduced to one scalar (frame geometry, duration, file bitrate, bits per pixel), the videos are ranked by that scalar, and the ranking is scored as if it were a detector. AUROC is symmetric in direction — a reading of 0.2174 ranks as strongly as one of 0.7826 — so what matters is the distance from 0.50, not the sign. Intervals come from an identity-clustered bootstrap over 2000 draws.</p>

<div class="tw">
<table>
  <caption><b>Table 10.</b> Single-scalar metadata controls on the two test subsets, with 95% identity-clustered intervals. <b>File bitrate reads 0.2174 on CelebDF++, an orientation-adjusted 0.7826</b>; the same control reads 0.5243 on DFD. The four geometry controls are exactly 0.5000 on DFD because both classes there are 1920×1080 throughout. Crop-scale controls require the per-video face-box cache and are not included here.</caption>
  <thead><tr><th>Control</th><th>CelebDF++ test</th><th>DFD test</th></tr></thead>
  <tbody>
    <tr class="head"><td>File bitrate (MB/s)</td><td>0.2174 &nbsp;<span class="m">[0.187, 0.240]</span></td><td>0.5243 &nbsp;<span class="m">[0.454, 0.616]</span></td></tr>
    <tr class="head"><td>Bits per pixel</td><td>0.6108 &nbsp;<span class="m">[0.568, 0.645]</span></td><td>0.5243 &nbsp;<span class="m">[0.454, 0.616]</span></td></tr>
    <tr><td>Frame width</td><td>0.0759 &nbsp;<span class="m">[0.050, 0.098]</span></td><td>0.5000 &nbsp;<span class="m">[0.500, 0.500]</span></td></tr>
    <tr><td>Frame pixels</td><td>0.0775 &nbsp;<span class="m">[0.050, 0.102]</span></td><td>0.5000 &nbsp;<span class="m">[0.500, 0.500]</span></td></tr>
    <tr><td>Aspect ratio</td><td>0.1240 &nbsp;<span class="m">[0.084, 0.168]</span></td><td>0.5000 &nbsp;<span class="m">[0.500, 0.500]</span></td></tr>
    <tr><td>Frame height</td><td>0.3940 &nbsp;<span class="m">[0.377, 0.410]</span></td><td>0.5000 &nbsp;<span class="m">[0.500, 0.500]</span></td></tr>
    <tr><td>Frame count</td><td>0.1051 &nbsp;<span class="m">[0.068, 0.138]</span></td><td>0.3923 &nbsp;<span class="m">[0.325, 0.472]</span></td></tr>
  </tbody>
</table>
</div>

<p>To check whether the model output carries a monotonic association with the strongest of these, we computed the Spearman rank correlation between the anomaly score and both compression measures over the same 5600 test videos of experiment 3, recomputing each measure’s own AUROC on that same subset.</p>

<div class="tw">
<table>
  <caption><b>Table 11.</b> Spearman rank correlation between the model anomaly score and the metadata measures, experiment 3 test set. The model's own AUROC on the same subset is 0.5284. A rank correlation detects monotonic association only; it does not rule out a non-linear dependence or an interaction with other quantities.</caption>
  <thead><tr><th>Subset</th><th class="n">Videos</th><th class="n">ρ with bitrate</th><th class="n">ρ with bits per pixel</th></tr></thead>
  <tbody>
    <tr><td>All</td><td class="n">5600</td><td class="n">−0.0218</td><td class="n">−0.0049</td></tr>
    <tr><td>Real only</td><td class="n">169</td><td class="n">−0.0866</td><td class="n">−0.0890</td></tr>
    <tr><td>Fake only</td><td class="n">5431</td><td class="n">−0.0192</td><td class="n">−0.0040</td></tr>
  </tbody>
</table>
</div>

<h2><span class="n">7</span>Sensitivity of the interval to the resampling scheme</h2>

<p><b>Purpose:</b> a bootstrap interval for AUROC depends on the resampling scheme. This section measures what two schemes give on the same set of scores, as a sensitivity analysis and <b>not as a significance test</b>.</p>

<p><b>Method:</b> the per-video test scores of each experiment are bootstrapped twice, 2000 draws each under the same seed, both through <span class="m">clustered_auroc_interval</span>: once with the single video as the group, once with the target identity. <b>Neither arm stratifies by class</b>, so the two differ by more than the resampling unit — under a 169:5431 split, declining to stratify widens an interval on its own. The stratified case/control scheme conventional in ROC analysis is not implemented here.</p>

<p><b>Effective clusters:</b> what sets the resolution is not the number of identities but the number carrying the scarcer class. The CelebDF++ test split holds 57 target identities and 5431 forgeries, but those forgeries come from <b>twelve</b> of them (322–570 videos each); of DFD's four test identities, <b>three</b> carry forgeries.</p>

<div class="tw">
<table>
  <caption><b>Table 12.</b> The same test scores under two resampling schemes, 95% percentile intervals. Uncertainty estimates are sensitive to the scheme; given the small and class-asymmetric number of identity clusters — three in DFD, twelve fake-bearing identities in CelebDF++ — <b>we do not interpret the nominal coverage as evidence of above-chance performance</b>.</caption>
  <thead><tr><th>Run</th><th class="n">AUROC</th><th>Per video (unstratified)</th><th>Per identity (unstratified)</th><th class="n">Identities · with fakes</th></tr></thead>
  <tbody>
    <tr><td>Experiment 1 · DFD, whole frame</td><td class="n">0.4877</td><td><span class="m">[0.376, 0.600]</span> width 0.223, covers 0.50</td><td><span class="m">[0.458, 0.531]</span> width 0.073, covers 0.50</td><td class="n">4 · 3</td></tr>
    <tr><td>Experiment 3 · CelebDF++, face crop</td><td class="n">0.5284</td><td><span class="m">[0.484, 0.573]</span> width 0.089, covers 0.50</td><td><span class="m">[0.502, 0.572]</span> width 0.071, does not</td><td class="n">57 · 12</td></tr>
  </tbody>
</table>
</div>

<footer>
  Sources: <span class="m">results/curves.csv</span>, <span class="m">results/summary.csv</span>, <span class="m">results/run_stage_a_dfd_decimate/reports/</span> (108 per-video scores for experiment 1), <span class="m">results/run_stage_a_celebdfv3_face_stageb_celeb/</span> (experiment 3 metrics, the per-method breakdown, and 5600 per-video scores), <span class="m">results/run_stage_a_celebdfv3_face_pretrain/history.csv</span>, <span class="m">results/diagnostics/feature_probe_*.json</span>, <span class="m">results/diagnostics/shortcut_controls_celeb_face.json</span> (Table 10), <span class="m">scripts/video_size_reports/video_sizes.csv</span> (Tables 9 and 10). Architecture and parameter counts are read from an instantiated model in <span class="m">src/video_bcnn/model.py</span>. The full running log is <span class="m">docs/stage_a_experiment_log.md</span>.
</footer>

</div>

''' + SCRIPT + '''
'''

out = REPO / "docs/experiment_report_en.html"
out.write_text(DOC, encoding="utf-8")
left = re.findall(r"[一-鿿]+", DOC)
print("wrote {}  {} chars".format(out, len(DOC)))
print("remaining Chinese runs:", len(left), left[:12])

# -*- coding: utf-8 -*-
"""Add the metadata-baseline and resampling-unit sections to both editions.

Both are measurements the paper's contributions rest on, and both were argued
rather than shown. They are stated here as method, purpose and numbers, with no
reading of what they mean, in keeping with the rest of the document.

Also discloses that the scale row of table 8 was trained before the donor
constraint reached the CelebDF++ split, so it cannot serve as a clean control.
"""
from pathlib import Path

S = Path(__file__).resolve().parent

ZH = '''
<h2><span class="n">6</span>元数据基线</h2>

<p><b>目的：</b>检查在不解码任何像素的前提下，仅凭容器与编码层面的元信息能把两类分开到什么程度。这类信息不含任何伪造证据，因此它给出的是「数据集本身有多可分」的下界参照。</p>

<p><b>方法：</b>对每条视频取一个标量（画面宽高、时长、文件码率、每像素比特数），按该标量排序，把这个排序当作检测器打分。AUROC 关于方向对称——读数 0.2174 与 0.7826 的判别力相同——因此看的是它离 0.50 的距离，而非方向。区间由按身份聚类的自助重采样给出（2000 次）。</p>

<div class="tw">
<table>
  <caption><b>表 9.</b> 单标量元数据控制项在两个测试子集上的 AUROC 与 95% 身份聚类区间。<b>码率一项在 CelebDF++ 上读 0.2174，等效判别力 0.7826</b>；同一项在 DFD 上为 0.5243。DFD 的四个几何量精确等于 0.5000，因为该数据集两类视频均为 1920×1080。裁剪尺度相关的控制项需要逐视频的人脸框缓存，未包含在本表内。</caption>
  <thead><tr><th>控制项</th><th>CelebDF++ 测试集</th><th>DFD 测试集</th></tr></thead>
  <tbody>
    <tr class="head"><td>文件码率 (MB/s)</td><td>0.2174 &nbsp;<span class="m">[0.187, 0.240]</span></td><td>0.5243 &nbsp;<span class="m">[0.454, 0.616]</span></td></tr>
    <tr class="head"><td>每像素比特数</td><td>0.6108 &nbsp;<span class="m">[0.568, 0.645]</span></td><td>0.5243 &nbsp;<span class="m">[0.454, 0.616]</span></td></tr>
    <tr><td>画面宽度</td><td>0.0759 &nbsp;<span class="m">[0.050, 0.098]</span></td><td>0.5000 &nbsp;<span class="m">[0.500, 0.500]</span></td></tr>
    <tr><td>画面像素数</td><td>0.0775 &nbsp;<span class="m">[0.050, 0.102]</span></td><td>0.5000 &nbsp;<span class="m">[0.500, 0.500]</span></td></tr>
    <tr><td>宽高比</td><td>0.1240 &nbsp;<span class="m">[0.084, 0.168]</span></td><td>0.5000 &nbsp;<span class="m">[0.500, 0.500]</span></td></tr>
    <tr><td>画面高度</td><td>0.3940 &nbsp;<span class="m">[0.377, 0.410]</span></td><td>0.5000 &nbsp;<span class="m">[0.500, 0.500]</span></td></tr>
    <tr><td>帧数</td><td>0.1051 &nbsp;<span class="m">[0.068, 0.138]</span></td><td>0.3923 &nbsp;<span class="m">[0.325, 0.472]</span></td></tr>
  </tbody>
</table>
</div>

<p>为检查模型输出与其中最强的一项之间是否存在单调关联，我们在实验三的同一批 5600 条测试视频上，计算了模型异常分数与码率、每像素比特数的 Spearman 秩相关。</p>

<div class="tw">
<table>
  <caption><b>表 10.</b> 模型异常分数与元数据的 Spearman 秩相关，实验三测试集。同一子集上模型自身的 AUROC 为 0.5284。秩相关只能检出单调关联，无法排除非线性依赖或与其他量的交互。</caption>
  <thead><tr><th>子集</th><th class="n">视频数</th><th class="n">ρ 与码率</th><th class="n">ρ 与每像素比特数</th></tr></thead>
  <tbody>
    <tr><td>全部</td><td class="n">5600</td><td class="n">−0.0218</td><td class="n">−0.0049</td></tr>
    <tr><td>仅真实</td><td class="n">169</td><td class="n">−0.0866</td><td class="n">−0.0890</td></tr>
    <tr><td>仅伪造</td><td class="n">5431</td><td class="n">−0.0192</td><td class="n">−0.0040</td></tr>
  </tbody>
</table>
</div>

<h2><span class="n">7</span>重采样单位对区间的影响</h2>

<p><b>目的：</b>AUROC 的自助区间取决于重采样的单位。一条真实片段在 CelebDF++ 上平均衍生 84 条伪造视频，它们共享背景、服装、光照与镜头运动；按视频重采样把它们当作独立观测，按身份重采样则把它们作为一个整体抽取。本节测量这两种选择在同一批分数上给出的区间差异。</p>

<p><b>方法：</b>对每个实验的逐视频测试分数做两次自助重采样，各 2000 次，随机种子相同。第一次以单条视频为单位，第二次以目标身份为单位。只落到单一类别的抽样被跳过而非计分。</p>

<div class="tw">
<table>
  <caption><b>表 11.</b> 同一批测试分数在两种重采样单位下的 95% 区间。<b>在实验三上，逐视频区间覆盖 0.50，身份聚类区间不覆盖</b>。实验一的身份聚类区间建立在 4 个簇之上，可产生的不同重采样极少，应作为指示性而非精确结果读取。</caption>
  <thead><tr><th>运行</th><th class="n">AUROC</th><th>逐视频 95% 区间</th><th>身份聚类 95% 区间</th><th class="n">身份簇数</th></tr></thead>
  <tbody>
    <tr><td>实验一 · DFD，整帧</td><td class="n">0.4877</td><td><span class="m">[0.376, 0.600]</span> 宽 0.223</td><td><span class="m">[0.458, 0.531]</span> 宽 0.073</td><td class="n">4</td></tr>
    <tr><td>实验三 · CelebDF++，人脸裁剪</td><td class="n">0.5284</td><td><span class="m">[0.484, 0.573]</span> 宽 0.089</td><td><span class="m">[0.502, 0.572]</span> 宽 0.071</td><td class="n">57</td></tr>
  </tbody>
</table>
</div>
'''

EN = '''
<h2><span class="n">6</span>Metadata baselines</h2>

<p><b>Purpose:</b> to measure how far the two classes can be separated from container and encoding metadata alone, without decoding a single pixel. Such information carries no forgery evidence, so what it yields is a floor for how separable the dataset itself is.</p>

<p><b>Method:</b> each video is reduced to one scalar (frame geometry, duration, file bitrate, bits per pixel), the videos are ranked by that scalar, and the ranking is scored as if it were a detector. AUROC is symmetric in direction — a reading of 0.2174 ranks as strongly as one of 0.7826 — so what matters is the distance from 0.50, not the sign. Intervals come from an identity-clustered bootstrap over 2000 draws.</p>

<div class="tw">
<table>
  <caption><b>Table 9.</b> Single-scalar metadata controls on the two test subsets, with 95% identity-clustered intervals. <b>File bitrate reads 0.2174 on CelebDF++, an orientation-adjusted 0.7826</b>; the same control reads 0.5243 on DFD. The four geometry controls are exactly 0.5000 on DFD because both classes there are 1920×1080 throughout. Crop-scale controls require the per-video face-box cache and are not included here.</caption>
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

<p>To check whether the model output carries a monotonic association with the strongest of these, we computed the Spearman rank correlation between the anomaly score and both compression measures over the same 5600 test videos of experiment 3.</p>

<div class="tw">
<table>
  <caption><b>Table 10.</b> Spearman rank correlation between the model anomaly score and the metadata measures, experiment 3 test set. The model's own AUROC on the same subset is 0.5284. A rank correlation detects monotonic association only; it does not rule out a non-linear dependence or an interaction with other quantities.</caption>
  <thead><tr><th>Subset</th><th class="n">Videos</th><th class="n">ρ with bitrate</th><th class="n">ρ with bits per pixel</th></tr></thead>
  <tbody>
    <tr><td>All</td><td class="n">5600</td><td class="n">−0.0218</td><td class="n">−0.0049</td></tr>
    <tr><td>Real only</td><td class="n">169</td><td class="n">−0.0866</td><td class="n">−0.0890</td></tr>
    <tr><td>Fake only</td><td class="n">5431</td><td class="n">−0.0192</td><td class="n">−0.0040</td></tr>
  </tbody>
</table>
</div>

<h2><span class="n">7</span>The resampling unit and the interval</h2>

<p><b>Purpose:</b> a bootstrap interval for AUROC depends on what is resampled. One real clip in CelebDF++ yields a median of 84 forgeries that share its background, wardrobe, lighting and camera motion; resampling videos treats those as independent observations, while resampling identities draws them as a block. This section measures what the two choices give on the same set of scores.</p>

<p><b>Method:</b> the per-video test scores of each experiment are bootstrapped twice, 2000 draws each under the same seed — once with the single video as the unit, once with the target identity. Draws that end up with a single class are skipped rather than scored.</p>

<div class="tw">
<table>
  <caption><b>Table 11.</b> The same test scores under two resampling units, 95% intervals. <b>On experiment 3 the per-video interval covers 0.50 and the identity-clustered interval does not.</b> Experiment 1's identity-clustered interval rests on 4 clusters, which admit very few distinct resamples, and should be read as indicative rather than exact.</caption>
  <thead><tr><th>Run</th><th class="n">AUROC</th><th>Per-video 95% CI</th><th>Per-identity 95% CI</th><th class="n">Clusters</th></tr></thead>
  <tbody>
    <tr><td>Experiment 1 · DFD, whole frame</td><td class="n">0.4877</td><td><span class="m">[0.376, 0.600]</span> width 0.223</td><td><span class="m">[0.458, 0.531]</span> width 0.073</td><td class="n">4</td></tr>
    <tr><td>Experiment 3 · CelebDF++, face crop</td><td class="n">0.5284</td><td><span class="m">[0.484, 0.573]</span> width 0.089</td><td><span class="m">[0.502, 0.572]</span> width 0.071</td><td class="n">57</td></tr>
  </tbody>
</table>
</div>
'''


def edit(path, pairs):
    p = S / path
    t = p.read_text(encoding="utf-8")
    for old, new, tag in pairs:
        assert t.count(old) == 1, "{} / {} -> {}".format(path, tag, t.count(old))
        t = t.replace(old, new)
    p.write_text(t, encoding="utf-8")
    print("patched", path)


edit("_doc.py", [
    ("<footer>", ZH + "\n<footer>", "sections 6-7"),
    ('<td class="dim">量表：一个端到端测试 AUROC 为 0.7773 的提取器</td>',
     '<td class="dim">量表（历史诊断）：一个端到端测试 AUROC 为 0.7773 的提取器</td>',
     "scale row label"),
    ('<b>第一行是量表</b>：它测的是一个端到端测试 AUROC 达 0.7773 的提取器，用于显示当提取器确实学到了可线性分离的信息时，该探针会读出多大的增益。',
     '<b>第一行是量表</b>：它测的是一个端到端测试 AUROC 达 0.7773 的提取器，用于显示该探针会读出多大的增益。'
     '<b>该提取器训练于供体约束施加到 CelebDF++ 划分之前</b>，其训练数据包含后来被划入测试集的供体身份；'
     '同一配置在供体安全划分上（实验三）不再产生此增益。此行为历史诊断，不作为干净的正对照。',
     "scale row caption"),
    ("<span class=\"m\">results/diagnostics/</span>。",
     "<span class=\"m\">results/diagnostics/</span>（探针与控制项，含 "
     "<span class=\"m\">shortcut_controls_celeb_face.json</span>，见表 9）、"
     "<span class=\"m\">scripts/video_size_reports/video_sizes.csv</span>（表 9、表 10）。",
     "footer sources"),
])

edit("build_report_en.py", [
    ("<footer>", EN + "\n<footer>", "sections 6-7"),
    ('<td class="dim">Scale: an extractor with an end-to-end test AUROC of 0.7773</td>',
     '<td class="dim">Scale (historical diagnostic): an extractor with an end-to-end test AUROC of 0.7773</td>',
     "scale row label"),
    ('<b>The first row is the scale</b>: it measures an extractor whose end-to-end test AUROC is 0.7773, and shows how large a gain this probe reads when an extractor has in fact learned something linearly separable.',
     '<b>The first row is the scale</b>: it measures an extractor whose end-to-end test AUROC is 0.7773 '
     'and shows how large a gain this probe reads. <b>That extractor was trained before the donor '
     'constraint reached the CelebDF++ split</b>, so its training data contained donor identities that '
     'later fell in the test split; the same configuration produces no such gain on the donor-safe '
     'split (experiment 3). The row is a historical diagnostic, not a clean positive control.',
     "scale row caption"),
    ("<span class=\"m\">results/diagnostics/feature_probe_*.json</span>.",
     "<span class=\"m\">results/diagnostics/feature_probe_*.json</span>, "
     "<span class=\"m\">results/diagnostics/shortcut_controls_celeb_face.json</span> (Table 9), "
     "<span class=\"m\">scripts/video_size_reports/video_sizes.csv</span> (Tables 9 and 10).",
     "footer sources"),
])

head = (S / "_head.py").read_text(encoding="utf-8")
arch = (S / "_arch.py").read_text(encoding="utf-8").split("\n", 1)[1]
(S / "build_report.py").write_text(
    head + arch + "\n\n" + (S / "_doc.py").read_text(encoding="utf-8"), encoding="utf-8")
print("reassembled")

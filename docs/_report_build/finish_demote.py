# -*- coding: utf-8 -*-
"""Finish the demotion: table 8 loses its leaky scale row, English gets section 7."""
from pathlib import Path

S = Path(__file__).resolve().parent
NL = chr(10)

EN_SEC7 = '''<h2><span class="n">7</span>Sensitivity of the interval to the resampling scheme</h2>

<p><b>Purpose:</b> a bootstrap interval for AUROC depends on the resampling scheme. This section measures what two schemes give on the same set of scores, as a sensitivity analysis and <b>not as a significance test</b>.</p>

<p><b>Method:</b> the per-video test scores of each experiment are bootstrapped twice, 2000 draws each under the same seed, both through <span class="m">clustered_auroc_interval</span>: once with the single video as the group, once with the target identity. <b>Neither arm stratifies by class</b>, so the two differ by more than the resampling unit — under a 169:5431 split, declining to stratify widens an interval on its own. The stratified case/control scheme conventional in ROC analysis is not implemented here.</p>

<p><b>Effective clusters:</b> what sets the resolution is not the number of identities but the number carrying the scarcer class. The CelebDF++ test split holds 57 target identities and 5431 forgeries, but those forgeries come from <b>twelve</b> of them (322–570 videos each); of DFD's four test identities, <b>three</b> carry forgeries.</p>

<div class="tw">
<table>
  <caption><b>Table 11.</b> The same test scores under two resampling schemes, 95% percentile intervals. Uncertainty estimates are sensitive to the scheme; given the small and class-asymmetric number of identity clusters — three in DFD, twelve fake-bearing identities in CelebDF++ — <b>we do not interpret the nominal coverage as evidence of above-chance performance</b>.</caption>
  <thead><tr><th>Run</th><th class="n">AUROC</th><th>Per video (unstratified)</th><th>Per identity (unstratified)</th><th class="n">Identities · with fakes</th></tr></thead>
  <tbody>
    <tr><td>Experiment 1 · DFD, whole frame</td><td class="n">0.4877</td><td><span class="m">[0.376, 0.600]</span> width 0.223, covers 0.50</td><td><span class="m">[0.458, 0.531]</span> width 0.073, covers 0.50</td><td class="n">4 · 3</td></tr>
    <tr><td>Experiment 3 · CelebDF++, face crop</td><td class="n">0.5284</td><td><span class="m">[0.484, 0.573]</span> width 0.089, covers 0.50</td><td><span class="m">[0.502, 0.572]</span> width 0.071, does not</td><td class="n">57 · 12</td></tr>
  </tbody>
</table>
</div>
'''

ZH_NOTE = '''<div class="note">
  <b>历史诊断，非干净对照。</b>另有一个提取器，其冻结探针读数为 0.7322、随机初始化参照 0.5778（差值 +0.1544），端到端测试 AUROC 为 0.7773。<b>它训练于供体约束施加到 CelebDF++ 划分之前</b>，训练数据包含后来被划入测试集的供体身份；同一配置在供体安全划分上（即实验三）不再产生该增益。它说明该探针在存在泄漏时能读出明显差值，也说明供体约束为何必要，但<b>不能作为「监督预训练在干净协议下能产生可迁移表征」的证据</b>，因此不列入上表。
</div>
'''

EN_NOTE = '''<div class="note">
  <b>Historical diagnostic under a pre-correction split.</b> One further extractor reads 0.7322 on the frozen probe against a random-init reference of 0.5778 (a difference of +0.1544), with an end-to-end test AUROC of 0.7773. <b>It was trained before the donor-safe constraint was enforced on the CelebDF++ split</b>, so its training data contained donor identities that later fell in the test split; the same configuration produces no such difference on the donor-safe split, which is experiment 3. It shows that this probe can read a clear difference when one is present, and why the donor constraint matters, but it is <b>not evidence that supervised pre-training produces transferable representations under the clean protocol</b>, and so is not listed in the table above.
</div>
'''


def sub(text, old, new, tag):
    assert text.count(old) == 1, "{} -> {}".format(tag, text.count(old))
    return text.replace(old, new)


# ---------------------------------------------------------------- Chinese ---
p = S / "_doc.py"
t = p.read_text(encoding="utf-8")
t = sub(t, '    <tr><td class="dim">量表（历史诊断）：一个端到端测试 AUROC 为 0.7773 的提取器</td>'
           '<td class="n dim">38</td><td class="n dim">0.5778</td><td class="n dim">0.7322</td>'
           '<td class="n dim">+0.1544</td></tr>' + NL, "", "zh scale row")
t = sub(t, '<b>第一行是量表</b>：它测的是一个端到端测试 AUROC 达 0.7773 的提取器，用于显示该探针会读出多大的增益。'
           '<b>该提取器训练于供体约束施加到 CelebDF++ 划分之前</b>，其训练数据包含后来被划入测试集的供体身份；'
           '同一配置在供体安全划分上（实验三）不再产生此增益。此行为历史诊断，不作为干净的正对照。',
        '在供体安全协议下，本工作没有可用作正对照的提取器，见下方说明。', "zh caption")
t = sub(t, '<h2><span class="n">6</span>元数据基线</h2>',
        ZH_NOTE + NL + '<h2><span class="n">6</span>元数据基线</h2>', "zh note")
t = sub(t, '计算了模型异常分数与码率、每像素比特数的 Spearman 秩相关。',
        '计算了模型异常分数与码率、每像素比特数的 Spearman 秩相关，并在同一子集上重算了这两项自身的 AUROC。',
        "zh table 10 lead-in")
p.write_text(t, encoding="utf-8")
print("patched _doc.py")

# ---------------------------------------------------------------- English ---
p = S / "build_report_en.py"
t = p.read_text(encoding="utf-8")
start = t.index('<h2><span class="n">7</span>The resampling unit and the interval</h2>')
end = t.index(NL + "<footer>")
t = t[:start] + EN_SEC7 + t[end:]
t = sub(t, '    <tr><td class="dim">Scale (historical diagnostic): an extractor with an '
           'end-to-end test AUROC of 0.7773</td><td class="n dim">38</td>'
           '<td class="n dim">0.5778</td><td class="n dim">0.7322</td>'
           '<td class="n dim">+0.1544</td></tr>' + NL, "", "en scale row")
t = sub(t, '<b>The first row is the scale</b>: it measures an extractor whose end-to-end test '
           'AUROC is 0.7773 and shows how large a gain this probe reads. <b>That extractor was '
           'trained before the donor constraint reached the CelebDF++ split</b>, so its training '
           'data contained donor identities that later fell in the test split; the same '
           'configuration produces no such gain on the donor-safe split (experiment 3). The row '
           'is a historical diagnostic, not a clean positive control.',
        'Under the donor-safe protocol this work has no extractor that can serve as a positive '
        'control; see the note below.', "en caption")
t = sub(t, '<h2><span class="n">6</span>Metadata baselines</h2>',
        EN_NOTE + NL + '<h2><span class="n">6</span>Metadata baselines</h2>', "en note")
t = sub(t, 'over the same 5600 test videos of experiment 3.',
        'over the same 5600 test videos of experiment 3, recomputing each measure’s own AUROC on '
        'that same subset.', "en table 10 lead-in")
p.write_text(t, encoding="utf-8")
print("patched build_report_en.py")

head = (S / "_head.py").read_text(encoding="utf-8")
arch = (S / "_arch.py").read_text(encoding="utf-8").split(NL, 1)[1]
(S / "build_report.py").write_text(
    head + arch + NL + NL + (S / "_doc.py").read_text(encoding="utf-8"), encoding="utf-8")
print("reassembled")

# -*- coding: utf-8 -*-
"""Put the split audit in section 2.1, where the protocol is defined.

The donor constraint was described but never counted. What it costs and what it
catches are the quantities that justify it, and they belong beside the rule.
Everything from table 2 onward shifts by one.
"""
import re
from pathlib import Path

S = Path(__file__).resolve().parent
NL = chr(10)

ZH_TABLE = '''
<div class="tw">
<table>
  <caption><b>表 2.</b> 仅按目标身份划分时残留的供体重叠。「target-only 伪造数」是每条伪造视频跟随其目标身份时该子集会得到的数量。<b>并非所有伪造方法都涉及第二个身份</b>——CelebDF++ 的 TalkingFace 家族由音频驱动单一身份，没有供体人脸，供体规则对其不适用，故「有供体」一列在该数据集上不足六成。「供体在训练集」的百分比以有供体的伪造视频为分母；若以全部 target-only 伪造视频为分母，CelebDF++ 测试子集为 37.8%。「被规则排除」是协议的<b>数据代价而非泄漏率</b>：被排除的视频中，部分供体位于验证集而非训练集。</caption>
  <thead><tr><th>数据集 · 子集</th><th class="n">target-only 伪造数</th><th class="n">其中有供体</th><th class="n">供体在训练集</th><th class="n">被规则排除</th></tr></thead>
  <tbody>
    <tr><td>DFD · 验证</td><td class="n">250</td><td class="n">250 (100%)</td><td class="n">199 (79.6%)</td><td class="n">233 (93.2%)</td></tr>
    <tr class="head"><td>DFD · 测试</td><td class="n">406</td><td class="n">406 (100%)</td><td class="n">313 (77.1%)</td><td class="n">352 (86.7%)</td></tr>
    <tr><td>CelebDF++ · 验证</td><td class="n">8462</td><td class="n">4311 (50.9%)</td><td class="n">2554 (59.2%)</td><td class="n">3387 (40.0%)</td></tr>
    <tr class="head"><td>CelebDF++ · 测试</td><td class="n">10255</td><td class="n">5975 (58.3%)</td><td class="n">3879 (64.9%)</td><td class="n">4822 (47.0%)</td></tr>
  </tbody>
</table>
</div>
'''

EN_TABLE = '''
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
'''

ZH_LEAD = ('该约束在 DFD 剔除 1103 条伪造视频（占 36%），在 CelebDF++ 剔除 14538 条（占 27.3%）。'
           '表 2 给出仅隔离目标身份时残留的重叠规模。')
EN_LEAD = ('The constraint discards 1103 forged videos in DFD (36%) and 14538 in CelebDF++ '
           '(27.3%). Table 2 reports how much overlap a target-only split leaves behind.')


def renumber(text, word):
    """Shift every table number from 2 upward by one, captions and references."""
    def shift(match):
        number = int(match.group(1))
        return "{} {}".format(word, number + 1 if number >= 2 else number)
    return re.sub(word + r" (\d+)", shift, text)


for path, word, table, old_tail, new_tail, anchor in (
        ("_doc.py", "表", ZH_TABLE,
         "该约束在 DFD 剔除 1103 条伪造视频（占 36%），在 CelebDF++ 剔除 14538 条（占 27.3%）。",
         ZH_LEAD, "<h3>2.2 "),
        ("build_report_en.py", "Table", EN_TABLE,
         "The constraint discards 1103 forged videos in DFD (36%) and 14538 in CelebDF++ (27.3%).",
         EN_LEAD, "<h3>2.2 ")):
    p = S / path
    t = p.read_text(encoding="utf-8")
    before = len(re.findall(word + r" \d+", t))
    t = renumber(t, word)
    assert t.count(old_tail) == 1, "{}: lead-in -> {}".format(path, t.count(old_tail))
    t = t.replace(old_tail, new_tail)
    index = t.index(anchor)
    t = t[:index] + table.lstrip(NL) + NL + t[index:]
    after = len(re.findall(word + r" \d+", t))
    p.write_text(t, encoding="utf-8")
    print("{}: {} table references before, {} after".format(path, before, after))

head = (S / "_head.py").read_text(encoding="utf-8")
arch = (S / "_arch.py").read_text(encoding="utf-8").split(NL, 1)[1]
(S / "build_report.py").write_text(
    head + arch + NL + NL + (S / "_doc.py").read_text(encoding="utf-8"), encoding="utf-8")
print("reassembled")

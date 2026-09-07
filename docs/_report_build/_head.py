# -*- coding: utf-8 -*-
"""Rebuild the report as method, purpose and measured data only.

The previous draft argued a conclusion. This one states what was built, why each
step was run, and what came out, and leaves the reading of it to the reader.
Figures and tables carry over; every caption is rewritten to describe rather
than interpret. The architecture figure is new.
"""
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SA = REPO / "scratch_archive"
BOXES = SA.joinpath("_figboxes.txt").read_text(encoding="utf-8").split("\n<<<SPLIT>>>\n")
STYLE = SA.joinpath("_style.txt").read_text(encoding="utf-8")
SCRIPT = SA.joinpath("_script.txt").read_text(encoding="utf-8")
assert len(BOXES) == 8, len(BOXES)

# figbox index -> figure number in the new document
EXP1_LOSS, EXP1_AUC, EXP1_ROC, TSNE, EXP3_A, EXP3_B, EXP3_ROC, EXP3_METHOD = range(8)


def fig(box, number, caption):
    return ('<figure>\n  ' + BOXES[box].rstrip() +
            '\n  <figcaption><b>图 {}.</b> {}</figcaption>\n</figure>'.format(number, caption))



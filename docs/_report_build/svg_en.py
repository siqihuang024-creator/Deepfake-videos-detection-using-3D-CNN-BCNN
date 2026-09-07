# -*- coding: utf-8 -*-
"""Chinese -> English for every text node inside the carried-over figures."""

SVG_TEXT = {
    # figure 2 / 3 -- experiment 1 stage A
    "训练交叉熵": "Training cross-entropy",
    "阶段 A 训练交叉熵随 epoch 变化：从 1.578 下降到 0.658，60 个 epoch 中有 46 个低于 ln 2。":
        "Stage A training cross-entropy by epoch: from 1.578 down to 0.658, with 46 of 60 epochs below ln 2.",
    "随机水平 0.50": "Chance level 0.50",
    "验证 AUC（62 条视频）": "Validation AUC (62 videos)",
    "阶段 A 验证 AUC 随 epoch 变化：均值 0.4477，自第 24 个 epoch 起 37 次中仅 1 次高于 0.5。":
        "Stage A validation AUC by epoch: mean 0.4477; from epoch 24 onward, 1 of 37 epochs exceeds 0.5.",
    # figure 4 -- experiment 1 scores and ROC
    "(a) 视频级异常分数分布": "(a) Video-level anomaly score distribution",
    "阈值": "Threshold",
    "异常分数（越大越可疑）": "Anomaly score (higher is more suspect)",
    "真实 (54)": "Real (54)",
    "伪造 (54)": "Fake (54)",
    "(b) ROC 曲线": "(b) ROC curve",
    "随机": "Chance",
    "假阳率": "False positive rate",
    "真阳率": "True positive rate",
    "左：真实与伪造视频的异常分数直方图，两个分布几乎完全重叠；右：ROC 曲线贴着对角线，AUC 0.4877。":
        "Left: anomaly score histograms for real and fake videos, almost entirely overlapping. "
        "Right: the ROC curve follows the diagonal, AUC 0.4877.",
    # figure 9 -- t-SNE and probe,
    "真实 (30)": "Real (30)",
    "伪造 (30)": "Fake (30)",
    "5-NN 按身份留一 0.375": "5-NN, leave-one-identity-out 0.375",
    "60 条训练视频的提取器特征经 t-SNE 投影到平面，真实与伪造完全混杂，未形成可分结构。":
        "Extractor features from 60 training videos projected to a plane by t-SNE; real and fake are fully "
        "intermixed and form no separable structure.",
    # figure 5 -- experiment 3 stage A
    "(a) 训练交叉熵": "(a) Training cross-entropy",
    "(b) 验证 AUROC（5241 条视频，57 身份）": "(b) Validation AUROC (5241 videos, 57 identities)",
    "随机水平": "Chance level",
    "实验三阶段 A：左为训练交叉熵自 1.542 降至 0.562，右为验证 AUROC 自 0.586 升至 0.692。":
        "Experiment 3, Stage A: left, training cross-entropy from 1.542 to 0.562; "
        "right, validation AUROC from 0.586 to 0.692.",
    # figure 6 -- experiment 3 stage B
    "验证最佳 0.6569": "Best validation 0.6569",
    "测试 0.5284": "Test 0.5284",
    "阶段 B epoch": "Stage B epoch",
    "实验三阶段 B 的 17 个验证 epoch，最高 0.6569，而测试集读数为 0.5284。":
        "The 17 validation epochs of experiment 3, Stage B, peaking at 0.6569, against a test reading of 0.5284.",
    # figure 7 -- experiment 3 scores and ROC
    "(a) 视频级异常分数分布（密度）": "(a) Video-level anomaly score distribution (density)",
    "真实 (169)": "Real (169)",
    "伪造 (5431)": "Fake (5431)",
    "左：真实与伪造视频异常分数的密度直方图，两个分布几乎完全重叠；右：ROC 曲线贴着对角线，AUC 0.5284。":
        "Left: density histograms of anomaly scores for real and fake videos, almost entirely overlapping. "
        "Right: the ROC curve follows the diagonal, AUC 0.5284.",
    # figure 8 -- per-method breakdown
    "7 种方法 · 4278 条伪造视频": "7 methods · 4278 fake videos",
    "7 种方法 · 469 条伪造视频": "7 methods · 469 fake videos",
    "8 种方法 · 684 条伪造视频": "8 methods · 684 fake videos",
    "测试集 video-level AUROC": "Test-set video-level AUROC",
    "按伪造方法拆分的测试集 AUROC，TalkingFace 家族整体高于随机，FaceSwap 家族整体低于随机。":
        "Test-set AUROC split by forgery method: the TalkingFace family sits above chance, "
        "the FaceSwap family below it.",
}

# the three probe row labels are wider in English; drop them a point so they
# clear the plot area they sit beside
SVG_TWEAKS = {
    '<g font-family="Source Serif 4, serif" font-size="11" fill="var(--ink)" text-anchor="start">':
        '<g font-family="Source Serif 4, serif" font-size="9.5" fill="var(--ink)" text-anchor="start">',
}

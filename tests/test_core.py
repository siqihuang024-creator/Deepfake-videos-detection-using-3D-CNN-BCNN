"""Fast tests that do not decode datasets or execute full 224x224 convolutions."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pyro
import torch
from pyro.infer import SVI, TraceGraph_ELBO
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from video_bcnn.data import VideoClipDataset
from video_bcnn.experiment import DatasetBalancedEpochSampler
from video_bcnn.metrics import calibrate_threshold, detection_metrics
from video_bcnn.model import Stable2DFeatureExtractor, Stable3DFeatureExtractor, VideoBayesianCNN
from evaluate_3d_bcnn import checkpoint_evaluation_config


class ModelContractTests(unittest.TestCase):
    def test_feature_contract_and_receptive_field(self):
        extractor = Stable3DFeatureExtractor(3)
        self.assertEqual(extractor.feature_dim, 15488)
        self.assertEqual(extractor.temporal_receptive_field, 7)
        self.assertEqual(extractor.conv1.kernel_size, (3, 5, 5))
        self.assertEqual(extractor.pool1.kernel_size, (1, 4, 4))

    def test_even_temporal_kernel_is_rejected(self):
        with self.assertRaises(ValueError):
            Stable3DFeatureExtractor(2)

    def test_wider_extractor_preserves_bayesian_feature_interface(self):
        extractor = Stable3DFeatureExtractor(5, (24, 32, 32))
        self.assertEqual(extractor.feature_dim, 15488)
        self.assertEqual(extractor.temporal_receptive_field, 13)
        self.assertEqual(extractor.conv_channels, (24, 32, 32))

    def test_wider_extractor_rejects_changed_output_channels(self):
        with self.assertRaises(ValueError):
            Stable3DFeatureExtractor(5, (24, 32, 48))

    def test_matched_2d_control_keeps_the_same_feature_interface(self):
        extractor = Stable2DFeatureExtractor()
        self.assertEqual(extractor.feature_dim, 15488)
        self.assertEqual(extractor.input_mode, "frame")
        self.assertEqual(extractor.conv1.kernel_size, (5, 5))


class ClipIndexTests(unittest.TestCase):
    def make_dataset(self, training):
        dataset = VideoClipDataset.__new__(VideoClipDataset)
        dataset.training = training
        dataset.clip_length = 8
        dataset.train_clip_strides = (1,)
        dataset.eval_clip_stride = 2
        dataset.clips_per_video = 3
        return dataset

    def test_eval_indices_are_contiguous_at_configured_stride(self):
        clips = self.make_dataset(False).clip_indices(100)
        self.assertEqual(len(clips), 3)
        for clip in clips:
            self.assertEqual(len(clip), 8)
            self.assertTrue(all(right - left == 2 for left, right in zip(clip, clip[1:])))

    def test_short_video_repeats_last_frame(self):
        clip = self.make_dataset(False).clip_indices(3)[0]
        self.assertEqual(clip[-1], 2)
        self.assertTrue(all(0 <= index <= 2 for index in clip))

    def test_missing_first_box_uses_nearest_face_not_center_square(self):
        frames = [np.zeros((100, 200, 3), dtype=np.uint8) for _ in range(3)]
        face = np.asarray([50.0, 20.0, 40.0, 40.0])
        boxes = VideoClipDataset._interpolate_missing_boxes([None, face, None], frames)
        np.testing.assert_allclose(boxes[0], face)
        np.testing.assert_allclose(boxes[2], face)


class MetricTests(unittest.TestCase):
    def test_fake_anomaly_direction(self):
        labels = np.asarray([1, 1, 0, 0])
        scores = np.asarray([0.1, 0.2, 0.8, 0.9])
        threshold = calibrate_threshold(scores[labels == 1], 0.05)
        metrics = detection_metrics(labels, scores, threshold)
        self.assertAlmostEqual(metrics["auroc"], 1.0)
        self.assertAlmostEqual(metrics["tpr_at_target_fpr"], 1.0)


class ClipLikelihoodTests(unittest.TestCase):
    class TinyExtractor(nn.Module):
        feature_dim = 2
        input_mode = "frame"

        def __init__(self):
            super().__init__()
            self.projection = nn.Linear(2, 2)

        def forward(self, values):
            return self.projection(values)

    def test_eight_frame_outputs_form_one_clip_observation(self):
        pyro.clear_param_store()
        model = VideoBayesianCNN(self.TinyExtractor(), kl_weight=0.001)
        svi = SVI(model.model, model.guide, pyro.optim.SGD({"lr": 1e-5}), loss=TraceGraph_ELBO())
        loss = svi.step(torch.randn(8, 2), torch.ones(1), 10, 8)
        self.assertTrue(np.isfinite(loss))


class BalancedSamplerTests(unittest.TestCase):
    def test_max_mode_uses_all_large_domain_and_resamples_small_domain(self):
        records = (
            [{"dataset": "DFD"} for _ in range(2)]
            + [{"dataset": "CelebDFv3"} for _ in range(3)]
        )
        sampler = DatasetBalancedEpochSampler(records, seed=42, samples_per_dataset="max")
        indices = list(iter(sampler))
        self.assertEqual(len(indices), 6)
        self.assertEqual(sum(index < 2 for index in indices), 3)
        self.assertEqual(sum(index >= 2 for index in indices), 3)
        self.assertEqual(set(index for index in indices if index >= 2), {2, 3, 4})


class EvaluationConfigTests(unittest.TestCase):
    def test_checkpoint_preprocessing_wins_over_changed_yaml(self):
        saved = {
            "device": "cuda",
            "data": {"clip_length": 8, "dataset_roots": {"DFD": "old"}},
            "train": {"report_dir": "old-report"},
        }
        runtime = {
            "device": "cpu",
            "data": {"clip_length": 16, "dataset_roots": {"DFD": "new"}},
            "train": {"report_dir": "new-report"},
        }
        result = checkpoint_evaluation_config(saved, runtime)
        self.assertEqual(result["data"]["clip_length"], 8)
        self.assertEqual(result["data"]["dataset_roots"]["DFD"], "new")
        self.assertEqual(result["train"]["report_dir"], "new-report")


if __name__ == "__main__":
    unittest.main()

"""Temporal 3D-CNN feature extraction with a paper-faithful Bayesian head."""

from .model import Stable2DFeatureExtractor, Stable3DFeatureExtractor, VideoBayesianCNN

__all__ = ["Stable2DFeatureExtractor", "Stable3DFeatureExtractor", "VideoBayesianCNN"]

"""3D-CNN video features with the original mean-field Bayesian FC head."""

import torch
import torch.nn.functional as F
from torch import nn

import pyro
import pyro.distributions as dist


# Sigmoid is the paper-faithful choice, but three stacked sigmoids cap the
# gradient reaching conv1 at 0.25**3, which measured as a ~1e-7 single-step
# update. The alternatives exist so that ceiling can be tested rather than
# assumed.
ACTIVATIONS = {
    "sigmoid": nn.Sigmoid,
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "tanh": nn.Tanh,
}


def build_activation(name):
    if name not in ACTIVATIONS:
        raise ValueError(
            "Unknown activation {!r}; expected one of {}.".format(name, sorted(ACTIVATIONS))
        )
    return ACTIVATIONS[name]()


def feature_dimension(channels, spatial_output_size):
    return int(channels) * int(spatial_output_size) * int(spatial_output_size)


class Stable2DFeatureExtractor(nn.Module):
    """Original paper feature extractor used with the matched clip preprocessing."""

    input_mode = "frame"

    def __init__(self, conv_channels=(16, 24, 32), activation="sigmoid",
                 spatial_output_size=22):
        super().__init__()
        conv_channels = tuple(int(value) for value in conv_channels)
        if len(conv_channels) != 3 or any(value < 1 for value in conv_channels):
            raise ValueError("conv_channels must contain three positive integers.")
        if int(spatial_output_size) < 1:
            raise ValueError("spatial_output_size must be positive.")
        self.conv_channels = conv_channels
        self.activation_name = activation
        self.spatial_output_size = int(spatial_output_size)
        self.feature_dim = feature_dimension(conv_channels[-1], spatial_output_size)
        self.conv1 = nn.Conv2d(3, conv_channels[0], kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(conv_channels[0], conv_channels[1], kernel_size=5, stride=1, padding=0)
        self.conv3 = nn.Conv2d(conv_channels[1], conv_channels[2], kernel_size=5, stride=1, padding=0)
        self.pool1 = nn.AvgPool2d(4, stride=2)
        self.pool2 = nn.AvgPool2d(4, stride=2)
        self.pool3 = nn.AvgPool2d(4, stride=2)
        self.activation = build_activation(activation)
        # Identity when the size already matches, so the default is unchanged.
        self.spatial_pool = nn.AdaptiveAvgPool2d(self.spatial_output_size)
        self.batch_norm = nn.BatchNorm2d(conv_channels[-1])
        self._reset_convolutions()

    def _reset_convolutions(self):
        for layer in (self.conv1, self.conv2, self.conv3):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, images):
        if images.dim() != 4 or images.shape[1] != 3 or tuple(images.shape[-2:]) != (224, 224):
            raise ValueError("The controlled 2D BCNN requires RGB 224x224 frames.")
        values = self.activation(self.pool1(self.conv1(images)))
        values = self.activation(self.pool2(self.conv2(values)))
        values = self.activation(self.pool3(self.conv3(values)))
        values = self.spatial_pool(values)
        return self.batch_norm(values).flatten(1)


class Stable3DFeatureExtractor(nn.Module):
    """Inflate only the paper CNN's temporal axis and preserve its spatial interface."""

    input_mode = "clip"

    def __init__(self, temporal_kernel_size=3, conv_channels=(16, 24, 32),
                 activation="sigmoid", spatial_output_size=22):
        super().__init__()
        temporal_kernel_size = int(temporal_kernel_size)
        if temporal_kernel_size < 1 or temporal_kernel_size % 2 == 0:
            raise ValueError("temporal_kernel_size must be a positive odd integer.")
        conv_channels = tuple(int(value) for value in conv_channels)
        if len(conv_channels) != 3 or any(value < 1 for value in conv_channels):
            raise ValueError("conv_channels must contain three positive integers.")
        if int(spatial_output_size) < 1:
            raise ValueError("spatial_output_size must be positive.")
        temporal_padding = temporal_kernel_size // 2
        kernel = (temporal_kernel_size, 5, 5)
        padding = (temporal_padding, 0, 0)
        self.temporal_kernel_size = temporal_kernel_size
        self.conv_channels = conv_channels
        self.activation_name = activation
        self.spatial_output_size = int(spatial_output_size)
        # 32 x 22 x 22 = 15488 reproduces the paper interface; a smaller
        # spatial_output_size shrinks the Bayesian FC1 without touching the head.
        self.feature_dim = feature_dimension(conv_channels[-1], spatial_output_size)
        self.conv1 = nn.Conv3d(3, conv_channels[0], kernel_size=kernel, stride=1, padding=padding)
        self.conv2 = nn.Conv3d(
            conv_channels[0], conv_channels[1], kernel_size=kernel, stride=1, padding=padding
        )
        self.conv3 = nn.Conv3d(
            conv_channels[1], conv_channels[2], kernel_size=kernel, stride=1, padding=padding
        )
        self.pool1 = nn.AvgPool3d((1, 4, 4), stride=(1, 2, 2))
        self.pool2 = nn.AvgPool3d((1, 4, 4), stride=(1, 2, 2))
        self.pool3 = nn.AvgPool3d((1, 4, 4), stride=(1, 2, 2))
        self.activation = build_activation(activation)
        # Identity when the size already matches, so the default is unchanged.
        self.spatial_pool = nn.AdaptiveAvgPool2d(self.spatial_output_size)
        # Temporal pooling precedes the original deterministic BatchNorm2d.
        self.batch_norm = nn.BatchNorm2d(conv_channels[-1])
        self._reset_convolutions()

    def _reset_convolutions(self):
        for layer in (self.conv1, self.conv2, self.conv3):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    @property
    def temporal_receptive_field(self):
        return 1 + 3 * (self.temporal_kernel_size - 1)

    def forward(self, clips):
        if clips.dim() != 5:
            raise ValueError("Expected clips shaped [batch, channels, time, height, width].")
        if clips.shape[1] != 3 or tuple(clips.shape[-2:]) != (224, 224):
            raise ValueError("The controlled 3D BCNN requires RGB 224x224 clips.")
        values = self.activation(self.pool1(self.conv1(clips)))
        values = self.activation(self.pool2(self.conv2(values)))
        values = self.activation(self.pool3(self.conv3(values)))
        values = self.spatial_pool(values.mean(dim=2))
        expected = (self.conv_channels[-1], self.spatial_output_size, self.spatial_output_size)
        if tuple(values.shape[1:]) != expected:
            raise RuntimeError("Unexpected extractor output shape: {}".format(tuple(values.shape)))
        return self.batch_norm(values).flatten(1)


class VideoBayesianCNN:
    """Mean-field posterior over FC1/FC2; the 3D feature extractor is deterministic."""

    LIKELIHOODS = ("gaussian", "bernoulli")

    def __init__(self, feature_extractor, dropout=0.2, prior_std=0.1,
                 observation_std=1.0, rho_init=-5.0, kl_weight=0.001,
                 likelihood="gaussian", hidden_dim=512):
        self.feature_extractor = feature_extractor
        self.dropout = float(dropout)
        self.prior_std = float(prior_std)
        self.observation_std = float(observation_std)
        self.rho_init = float(rho_init)
        self.kl_weight = float(kl_weight)
        if likelihood not in self.LIKELIHOODS:
            raise ValueError(
                "Unknown likelihood {!r}; expected one of {}.".format(
                    likelihood, list(self.LIKELIHOODS)
                )
            )
        self.likelihood = likelihood
        self.hidden_dim = int(hidden_dim)
        if self.hidden_dim < 1:
            raise ValueError("hidden_dim must be positive.")
        device = next(feature_extractor.parameters()).device
        self.fc1_init = nn.Linear(int(feature_extractor.feature_dim), self.hidden_dim).to(device)
        self.out_init = nn.Linear(self.hidden_dim, 1).to(device)

    @staticmethod
    def _event_normal(loc, scale):
        return dist.Normal(loc, scale).to_event(loc.dim())

    def _observation_distribution(self, predictions):
        """Paper-faithful Gaussian regression, or a Bernoulli head for labels.

        Supervised training has genuine {0, 1} targets, where a Gaussian with
        observation_std=1.0 buries the class signal under unit noise. The
        Bernoulli option reads the same output as a logit instead. Ranking
        metrics are unaffected because the logit is monotone in the probability.
        """
        if self.likelihood == "bernoulli":
            return dist.Bernoulli(logits=predictions)
        return dist.Normal(predictions, self.observation_std)

    def _prior_sample(self, name, initial_parameter):
        return pyro.sample(
            name,
            self._event_normal(
                torch.zeros_like(initial_parameter),
                torch.full_like(initial_parameter, self.prior_std),
            ),
        )

    def _posterior_sample(self, name, initial_parameter):
        loc = pyro.param("{}_loc".format(name), initial_parameter.detach().clone())
        rho = pyro.param("{}_rho".format(name), torch.full_like(initial_parameter, self.rho_init))
        return pyro.sample(name, self._event_normal(loc, F.softplus(rho) + 1e-6))

    def _sample_prior_weights(self):
        return {
            "fc1_weight": self._prior_sample("fc1_weight", self.fc1_init.weight),
            "fc1_bias": self._prior_sample("fc1_bias", self.fc1_init.bias),
            "out_weight": self._prior_sample("out_weight", self.out_init.weight),
            "out_bias": self._prior_sample("out_bias", self.out_init.bias),
        }

    def _sample_posterior_weights(self):
        return {
            "fc1_weight": self._posterior_sample("fc1_weight", self.fc1_init.weight),
            "fc1_bias": self._posterior_sample("fc1_bias", self.fc1_init.bias),
            "out_weight": self._posterior_sample("out_weight", self.out_init.weight),
            "out_bias": self._posterior_sample("out_bias", self.out_init.bias),
        }

    @staticmethod
    def _forward_features_with_weights(features, weights, dropout, training):
        hidden = F.linear(features, weights["fc1_weight"], weights["fc1_bias"])
        hidden = F.dropout(hidden, p=dropout, training=bool(training))
        return F.linear(hidden, weights["out_weight"], weights["out_bias"]).squeeze(-1)

    def _forward_with_weights(self, clips, weights, training):
        features = self.feature_extractor(clips)
        return self._forward_features_with_weights(features, weights, self.dropout, training)

    def model(self, clips, targets=None, num_train_clips=1, units_per_observation=1):
        pyro.module("video_features", self.feature_extractor, update_module_params=True)
        with pyro.poutine.scale(scale=self.kl_weight / float(max(1, num_train_clips))):
            weights = self._sample_prior_weights()
        predictions = self._forward_with_weights(clips, weights, training=True)
        units_per_observation = int(units_per_observation)
        if units_per_observation > 1:
            if predictions.numel() % units_per_observation:
                raise ValueError("Model outputs cannot be grouped into clip observations.")
            predictions = predictions.reshape(-1, units_per_observation).mean(dim=1)
        if targets is not None and predictions.shape != targets.shape:
            raise ValueError(
                "Prediction/target shape mismatch: {} versus {}.".format(
                    tuple(predictions.shape), tuple(targets.shape)
                )
            )
        with pyro.plate("video_bcnn_data", predictions.shape[0]):
            with pyro.poutine.scale(scale=1.0 / float(predictions.shape[0])):
                pyro.sample("observations", self._observation_distribution(predictions), obs=targets)

    def guide(self, clips, targets=None, num_train_clips=1, units_per_observation=1):
        del targets, units_per_observation
        pyro.module("video_features", self.feature_extractor, update_module_params=True)
        with pyro.poutine.scale(scale=self.kl_weight / float(max(1, num_train_clips))):
            self._sample_posterior_weights()

    def _posterior_loc_weights(self):
        store = pyro.get_param_store()
        return {
            "fc1_weight": store["fc1_weight_loc"],
            "fc1_bias": store["fc1_bias_loc"],
            "out_weight": store["out_weight_loc"],
            "out_bias": store["out_bias_loc"],
        }

    @torch.no_grad()
    def posterior_loc_from_features(self, features):
        return self._forward_features_with_weights(
            features, self._posterior_loc_weights(), self.dropout, training=False
        )

    @torch.no_grad()
    def posterior_from_features(self, features, mc_samples):
        store = pyro.get_param_store()
        location = self._posterior_loc_weights()
        scales = {
            name: F.softplus(store["{}_rho".format(name)]) + 1e-6
            for name in location
        }
        samples = []
        for _ in range(int(mc_samples)):
            weights = {
                name: location[name] + scales[name] * torch.randn_like(location[name])
                for name in location
            }
            samples.append(self._forward_features_with_weights(features, weights, self.dropout, False))
        values = torch.stack(samples, dim=0)
        return values.mean(dim=0), values.std(dim=0, unbiased=False)

    @torch.no_grad()
    def diagnostics(self):
        store = pyro.get_param_store()
        result = {
            "batch_norm_num_batches_tracked": int(self.feature_extractor.batch_norm.num_batches_tracked.item()),
            "batch_norm_running_mean_abs": float(self.feature_extractor.batch_norm.running_mean.abs().mean().item()),
            "batch_norm_running_var_mean": float(self.feature_extractor.batch_norm.running_var.mean().item()),
            "input_mode": self.feature_extractor.input_mode,
            "likelihood": self.likelihood,
            "activation": getattr(self.feature_extractor, "activation_name", "sigmoid"),
            "spatial_output_size": int(getattr(self.feature_extractor, "spatial_output_size", 22)),
            "feature_dim": int(self.feature_extractor.feature_dim),
            "hidden_dim": self.hidden_dim,
        }
        if self.feature_extractor.input_mode == "clip":
            result["temporal_kernel_size"] = int(self.feature_extractor.temporal_kernel_size)
            result["temporal_receptive_field"] = int(self.feature_extractor.temporal_receptive_field)
            result["conv_channels"] = list(self.feature_extractor.conv_channels)
        for name in ("fc1_weight", "fc1_bias", "out_weight", "out_bias"):
            rho = store["{}_rho".format(name)]
            sigma = F.softplus(rho) + 1e-6
            result["{}_rho_mean".format(name)] = float(rho.mean().item())
            result["{}_sigma_mean".format(name)] = float(sigma.mean().item())
        return result

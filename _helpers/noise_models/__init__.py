"""
Noise Models Package

This package provides a registry-based architecture for custom noise models.
Noise model factories are auto-registered when this package is imported.

Usage:
    from _helpers.noise_models import noise_model_registry, NoiseModelWrapper

    # Check if a noise model type is available
    if "arrhenius" in noise_model_registry:
        wrapper = NoiseModelWrapper(config)
        noise_model, backend = wrapper.build()

    # List all available noise model types
    print(noise_model_registry.list_factories())
"""

from _helpers.noise_models.base import (
    NoiseModelFactory,
    NoiseModelRegistry,
    CustomNoiseModelBackend,
    noise_model_registry,
)
from _helpers.noise_models.noise_model_wrapper import NoiseModelWrapper

# Import concrete implementations to trigger auto-registration
from _helpers.noise_models.arrhenius_noise_model import ArrheniusNoiseModel
from _helpers.noise_models.simple_noise_model import SimpleNoiseModel
from _helpers.noise_models.random_noise_model import RandomNoiseModel

__all__ = [
    "NoiseModelFactory",
    "NoiseModelRegistry",
    "CustomNoiseModelBackend",
    "noise_model_registry",
    "NoiseModelWrapper",
    "ArrheniusNoiseModel",
    "SimpleNoiseModel",
    "RandomNoiseModel",
]

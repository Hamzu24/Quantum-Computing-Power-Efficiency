from typing import Tuple, Any
from qiskit_aer.noise import NoiseModel
from _helpers.noise_models.base import noise_model_registry
from _helpers.helpers import get_control_parameters


class NoiseModelWrapper:
    def __init__(self, config: dict):
        self.config = config
        self.noise_model_type = config.get("type")

        if self.noise_model_type is None:
            raise ValueError("Config must have a 'type' field")

        if self.noise_model_type not in noise_model_registry:
            raise ValueError(
                f"Unknown noise model type: {self.noise_model_type}. "
                f"Available types: {noise_model_registry.list_factories()}"
            )

        # Resolve control parameters from config if present
        if config.get("control_parameters"):
            self.resolved_control_parameters = get_control_parameters(config)
        else:
            self.resolved_control_parameters = {}

        # Get the factory class and instantiate it
        factory_class = noise_model_registry.get(self.noise_model_type)
        self.factory = factory_class(config, self.resolved_control_parameters)

    def build(self) -> Tuple[NoiseModel, Any]:
        """
        Returns:
            Tuple of (NoiseModel, backend_class) where backend_class
            is typically a CustomNoiseModelBackend instance for custom
            noise models.
        """
        self.factory.validate()
        return self.factory.build()

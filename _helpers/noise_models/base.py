from typing import Protocol, ClassVar, Tuple, List, Any, Type, Dict
from qiskit_aer.noise import NoiseModel
from _helpers.constants import (
    DefaultBasisGatesNoiseless, DefaultBasisGates1qb, DefaultBasisGates2qb
)


class NoiseModelFactory(Protocol):
    registry_name: ClassVar[str]
    required_parameters: ClassVar[List[str]]
    optional_parameters: ClassVar[dict[str, Any]]

    def __init__(self, config: dict, resolved_control_parameters: dict): ...

    def validate(self) -> None:
        """Validate that all required parameters are present in the config."""
        ...

    def build(self) -> Tuple[NoiseModel, Any]:
        """Build and return the noise model and backend marker class."""
        ...

    def build_pauli(self) -> Tuple[Dict, Any]:
        """
        Build Pauli noise config for Stim stabilizer simulation.

        Returns:
            Tuple of (pauli_config, backend_marker) where pauli_config has structure:
                {
                    '1q': {'p_x': float, 'p_y': float, 'p_z': float},
                    '2q': {'p_x': float, 'p_y': float, 'p_z': float},
                    'measurement': {'p_flip': float}
                }

        Raises:
            NotImplementedError: If noise model cannot be converted to Pauli channels.
        """
        ...

    @classmethod
    def get_basis_gates(cls) -> List[str]:
        """Return the basis gates supported by this noise model."""
        ...


class NoiseModelRegistry:
    """Registry for custom noise model factories."""

    def __init__(self):
        self._factories: dict[str, Type[NoiseModelFactory]] = {}

    def register(self, cls: Type[NoiseModelFactory]) -> Type[NoiseModelFactory]:
        """
        Register a noise model factory class.

        Can be used as a decorator:
            @noise_model_registry.register
            class MyNoiseModel:
                registry_name = "my_noise_model"
                ...
        """
        self._factories[cls.registry_name] = cls
        return cls

    def get(self, name: str) -> Type[NoiseModelFactory]:
        """Get a noise model factory by name."""
        factory = self._factories.get(name)
        if factory is None:
            raise KeyError(f"No noise model factory registered with name: {name}")
        return factory

    def list_factories(self) -> List[str]:
        """List all registered factory names."""
        return list(self._factories.keys())

    def __contains__(self, name: str) -> bool:
        """Check if a factory is registered with the given name."""
        return name in self._factories


# Global registry instance
noise_model_registry = NoiseModelRegistry()


class CustomNoiseModelBackend:
    """
    Marker class for custom noise models (not from fake backends).

    For fake backends, this would be Tokyo, Sherbrooke, etc.

    Used to distinguish custom noise models from fake backend noise models
    when determining basis gates. The version=-1 convention indicates
    this is a custom noise model.
    """
    version = -1

    def __init__(self, basis_gates: List[str] = None):
        if basis_gates is None:
            basis_gates = DefaultBasisGates2qb + DefaultBasisGates1qb + DefaultBasisGatesNoiseless
        self.basis_gates = basis_gates

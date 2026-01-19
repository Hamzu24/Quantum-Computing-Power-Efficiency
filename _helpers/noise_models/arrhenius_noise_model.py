from typing import ClassVar, List, Any, Tuple
import numpy as np
from qiskit_aer.noise import NoiseModel, thermal_relaxation_error
from _helpers.noise_models.base import (
    noise_model_registry, CustomNoiseModelBackend
)
from _helpers.helpers import get_config_value
from _helpers.constants import (
    DefaultBasisGatesNoiseless, DefaultBasisGates1qb, DefaultBasisGates2qb
)


@noise_model_registry.register
class ArrheniusNoiseModel:
    """
    Noise model where T1 is derived from the Arrhenius error probability.

    This accurately models the 'decay' to |0> rather than just scrambling.
    The Arrhenius model calculates thermal excitation probability based on
    temperature and qubit frequency.
    """

    registry_name: ClassVar[str] = "arrhenius"
    required_parameters: ClassVar[List[str]] = []
    optional_parameters: ClassVar[dict[str, Any]] = {
        "qubit_frequency_hz": 5e9,
        "gate_length": 50e-9,
    }

    # Physical constants
    K_B = 1.380649e-23  # Boltzmann constant
    H = 6.626070e-34    # Planck constant

    def __init__(self, config: dict, resolved_control_parameters: dict):
        """
        Initialize the Arrhenius noise model.

        Args:
            config: Configuration dictionary containing optional parameters:
                - qubit_frequency_hz: Qubit frequency in Hz (default: 5e9)
                - gate_length: Gate length in seconds (default: 50e-9)
            resolved_control_parameters: Resolved control parameters containing:
                - temperature: Temperature value and unit (required)
        """
        self.config = config
        self.resolved_control_parameters = resolved_control_parameters

        # Extract parameters with defaults
        self.qubit_frequency_hz = config.get(
            "qubit_frequency_hz",
            self.optional_parameters["qubit_frequency_hz"]
        )
        self.gate_length = config.get(
            "gate_length",
            self.optional_parameters["gate_length"]
        )

    def validate(self) -> None:
        """Validate that temperature is available in control parameters."""
        temperature = get_config_value(self.resolved_control_parameters, "temperature")
        if temperature is None:
            raise ValueError(
                "Arrhenius noise model requires 'temperature' in control_parameters"
            )

    def build(self) -> Tuple[NoiseModel, CustomNoiseModelBackend]:
        """
        Build the Arrhenius-based noise model.

        Returns:
            Tuple of (NoiseModel, CustomNoiseModelBackend)
        """
        # Get temperature from control parameters
        T = get_config_value(self.resolved_control_parameters, "temperature")

        # Calculate qubit energy
        E = self.H * self.qubit_frequency_hz

        # Calculate Arrhenius probability
        p_phys = np.exp(-E / (self.K_B * T))

        # Calculate T1 and T2 from physical probability
        if p_phys <= 0:
            T1 = np.inf
            T2 = np.inf
        else:
            T1 = self.gate_length / p_phys
            T2 = 2 * T1

        # Build noise model
        noise_model = NoiseModel()

        # Single-qubit gate errors
        error_1q = thermal_relaxation_error(T1, T2, self.gate_length)

        # Two-qubit gate errors (4x gate length)
        gate_length_2q = 4 * self.gate_length
        error_2q_single = thermal_relaxation_error(T1, T2, gate_length_2q)
        error_2q = error_2q_single.tensor(error_2q_single)

        # Add errors to all qubits
        noise_model.add_all_qubit_quantum_error(error_1q, DefaultBasisGates1qb)
        noise_model.add_all_qubit_quantum_error(error_2q, DefaultBasisGates2qb)

        return noise_model, CustomNoiseModelBackend(self.get_basis_gates())

    @classmethod
    def get_basis_gates(cls) -> List[str]:
        return DefaultBasisGates2qb + DefaultBasisGates1qb + DefaultBasisGatesNoiseless

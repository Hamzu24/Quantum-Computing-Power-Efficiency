from typing import ClassVar, List, Any, Tuple
import numpy as np
from qiskit.circuit.library import RZXGate, RZGate, RXGate, RZZGate
from qiskit_aer.noise import (
    NoiseModel, thermal_relaxation_error, depolarizing_error, coherent_unitary_error
)
from _helpers.noise_models.base import (
    noise_model_registry, CustomNoiseModelBackend
)
from _helpers.constants import (
    DefaultBasisGatesNoiseless, DefaultBasisGates1qb, DefaultBasisGates2qb
)


@noise_model_registry.register
class RandomNoiseModel:
    """
    Random noise model with randomly sampled T1/T2 times and coherent errors.

    This noise model samples T1/T2 times from normal distributions and
    applies random coherent errors, useful for testing robustness to
    device-to-device variation.
    """

    registry_name: ClassVar[str] = "random_simple_nm"
    required_parameters: ClassVar[List[str]] = []
    optional_parameters: ClassVar[dict[str, Any]] = {
        "seed": 0,
        "T1_mean": 50e3,
        "T1_std": 1e3,
        "T2_mean": 70e3,
        "T2_std": 1e3,
        "detuning_mean": 0.0,
        "detuning_std": np.pi / 150,
        "overrotation_mean": 0.0,
        "overrotation_std": np.pi / 100,
    }

    def __init__(self, config: dict, resolved_control_parameters: dict):
        """
        Initialize the random noise model.

        Args:
            config: Configuration dictionary containing optional parameters:
                - num_qubits: Number of qubits (default: 4)
                - seed: Random seed for reproducibility (default: 0, None for random)
                - T1_mean, T1_std: Mean and std for T1 sampling (default: 50e3, 1e3)
                - T2_mean, T2_std: Mean and std for T2 sampling (default: 70e3, 1e3)
                - detuning_mean, detuning_std: Detuning error parameters
                - overrotation_mean, overrotation_std: Overrotation error parameters
            resolved_control_parameters: Resolved control parameters (not used)
        """
        overall_config_data = read_config()
        self.num_qubits = overall_config_data.get("num_qubits")
        if self.num_qubits is None:
            raise ValueError("You need to include num_qubits value in config!")

        # Now dealing with the noise model config:
        self.config = config
        self.resolved_control_parameters = resolved_control_parameters

        # Extract parameters with defaults
        self.seed = config.get("seed", self.optional_parameters["seed"])

        # T1/T2 distribution parameters
        self.T1_mean = config.get("T1_mean", self.optional_parameters["T1_mean"])
        self.T1_std = config.get("T1_std", self.optional_parameters["T1_std"])
        self.T2_mean = config.get("T2_mean", self.optional_parameters["T2_mean"])
        self.T2_std = config.get("T2_std", self.optional_parameters["T2_std"])

        # Coherent error parameters
        self.detuning_mean = config.get("detuning_mean", self.optional_parameters["detuning_mean"])
        self.detuning_std = config.get("detuning_std", self.optional_parameters["detuning_std"])
        self.overrotation_mean = config.get(
            "overrotation_mean", self.optional_parameters["overrotation_mean"]
        )
        self.overrotation_std = config.get(
            "overrotation_std", self.optional_parameters["overrotation_std"]
        )

    def validate(self) -> None:
        pass

    def build(self) -> Tuple[NoiseModel, CustomNoiseModelBackend]:
        num_qubits = self.num_qubits

        # Set random seed
        if self.seed is not None:
            np.random.seed(self.seed)

        # Sample T1/T2 times from normal distributions
        T1s = np.random.normal(self.T1_mean, self.T1_std, num_qubits)
        T2s = np.random.normal(self.T2_mean, self.T2_std, num_qubits)

        # Truncate T2s <= 2*T1s (physical constraint)
        T2s = np.array([min(T2s[j], 2 * T1s[j]) for j in range(num_qubits)])

        # Fixed instruction times (in nanoseconds)
        time_rz = 0       # virtual gate
        time_sx = 50      # single X90 pulse
        time_x = 100      # two X90 pulses
        time_cx = 300
        time_reset = 1000  # 1 microsecond
        time_measure = 1000  # 1 microsecond

        # Create thermal relaxation errors for each qubit
        errors_reset = [
            thermal_relaxation_error(t1, t2, time_reset)
            for t1, t2 in zip(T1s, T2s)
        ]
        errors_measure = [
            thermal_relaxation_error(t1, t2, time_measure)
            for t1, t2 in zip(T1s, T2s)
        ]
        errors_u1 = [
            thermal_relaxation_error(t1, t2, time_rz)
            for t1, t2 in zip(T1s, T2s)
        ]
        errors_u2 = [
            thermal_relaxation_error(t1, t2, time_sx)
            for t1, t2 in zip(T1s, T2s)
        ]
        errors_u3 = [
            thermal_relaxation_error(t1, t2, time_x)
            for t1, t2 in zip(T1s, T2s)
        ]
        errors_cx = [
            [
                thermal_relaxation_error(t1a, t2a, time_cx).expand(
                    thermal_relaxation_error(t1b, t2b, time_cx)
                )
                for t1a, t2a in zip(T1s, T2s)
            ]
            for t1b, t2b in zip(T1s, T2s)
        ]

        # Sample random coherent error amounts
        overrotation_amount = np.random.normal(self.overrotation_mean, self.overrotation_std)
        detuning_amount = np.random.normal(self.detuning_mean, self.detuning_std)

        overrotation_unitary_1q = RXGate(overrotation_amount).to_matrix()
        detuning_unitary_1q = RZGate(detuning_amount).to_matrix()
        sx_gate_overrotation_error = coherent_unitary_error(overrotation_unitary_1q)
        sx_gate_detuning_error = coherent_unitary_error(detuning_unitary_1q)

        coherent_unitary_2q = RZXGate(overrotation_amount).to_matrix()
        zz_unitary_2q = RZZGate(overrotation_amount).to_matrix()
        coherent_unitary_2q_error = coherent_unitary_error(coherent_unitary_2q)
        zz_2q_error = coherent_unitary_error(zz_unitary_2q)

        # Build noise model
        noise_model = NoiseModel()

        for j in range(num_qubits):
            # Thermal relaxation errors
            noise_model.add_quantum_error(errors_reset[j], "reset", [j])
            noise_model.add_quantum_error(errors_measure[j], "measure", [j])
            noise_model.add_quantum_error(errors_u1[j], "rz", [j])
            noise_model.add_quantum_error(errors_u2[j], "sx", [j])
            noise_model.add_quantum_error(errors_u2[j], "id", [j])

            # Coherent errors on sx gate
            noise_model.add_quantum_error(
                sx_gate_overrotation_error, ['sx'], [j], warnings=False
            )
            noise_model.add_quantum_error(
                sx_gate_detuning_error, ['sx'], [j], warnings=False
            )

            # Depolarizing error on sx gate
            noise_model.add_quantum_error(
                depolarizing_error(0.0005, 1), ['sx'], [j], warnings=False
            )

            # Two-qubit errors
            for k in range(num_qubits):
                noise_model.add_quantum_error(errors_cx[j][k], "cx", [j, k])
                noise_model.add_quantum_error(
                    depolarizing_error(0.005, 2), ["cx"], [j, k], warnings=False
                )
                noise_model.add_quantum_error(
                    coherent_unitary_2q_error, ["cx"], [j, k], warnings=False
                )
                noise_model.add_quantum_error(
                    zz_2q_error, ["cx"], [j, k], warnings=False
                )

        return noise_model, CustomNoiseModelBackend(self.get_basis_gates())

    @classmethod
    def get_basis_gates(cls) -> List[str]:
        return DefaultBasisGates2qb + DefaultBasisGates1qb + DefaultBasisGatesNoiseless

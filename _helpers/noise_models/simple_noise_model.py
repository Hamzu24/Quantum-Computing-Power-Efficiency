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
    DEFAULT_INSTRUCTION_TIMES,
    DefaultBasisGatesNoiseless, DefaultBasisGates1qb, DefaultBasisGates2qb
)


@noise_model_registry.register
class SimpleNoiseModel:
    """
    Simple custom noise model with thermal relaxation and coherent errors.

    This noise model includes:
    - Thermal relaxation errors based on T1/T2 times
    - Coherent overrotation and detuning errors
    - Depolarizing errors
    """

    registry_name: ClassVar[str] = "simple_nm"
    required_parameters: ClassVar[List[str]] = []
    optional_parameters: ClassVar[dict[str, Any]] = {
        "T1s": 50e3,
        "T2s": 70e3,
        "instruction_times": DEFAULT_INSTRUCTION_TIMES,
        "overrotation_amount": np.pi / 100,
        "detuning_amount": np.pi / 120,
    }

    def __init__(self, config: dict, resolved_control_parameters: dict):
        """
        Initialize the simple noise model.

        Args:
            config: Configuration dictionary containing optional parameters:
                - num_qubits: Number of qubits (default: 4)
                - T1s: T1 relaxation times in ns, scalar or array (default: 50e3)
                - T2s: T2 dephasing times in ns, scalar or array (default: 70e3)
                - instruction_times: Dict of gate times (default: DEFAULT_INSTRUCTION_TIMES)
                - overrotation_amount: Overrotation angle (default: pi/100)
                - detuning_amount: Detuning angle (default: pi/120)
            resolved_control_parameters: Resolved control parameters (not used)
        """
        overall_config_data = read_config()
        self.num_qubits = overall_config_data.get("num_qubits")
        if self.num_qubits is None:
            raise ValueError("You need to include num_qubits value in config!")


        # Now dealing with the noise model config
        self.config = config
        self.resolved_control_parameters = resolved_control_parameters

        # Extract parameters with defaults
        self.T1s = config.get("T1s", self.optional_parameters["T1s"])
        self.T2s = config.get("T2s", self.optional_parameters["T2s"])
        self.instruction_times = config.get(
            "instruction_times",
            self.optional_parameters["instruction_times"]
        )
        self.overrotation_amount = config.get(
            "overrotation_amount",
            self.optional_parameters["overrotation_amount"]
        )
        self.detuning_amount = config.get(
            "detuning_amount",
            self.optional_parameters["detuning_amount"]
        )

    def validate(self) -> None:
        """Validate that all required instruction times are present."""
        required_times = ["time_rz", "time_sx", "time_x", "time_cx", "time_reset", "time_measure"]
        for time_key in required_times:
            if self.instruction_times.get(time_key) is None:
                raise ValueError(
                    f"Missing required instruction time: {time_key}. "
                    f"Required times: {required_times}"
                )

    def build(self) -> Tuple[NoiseModel, CustomNoiseModelBackend]:
        """
        Build the simple custom noise model.

        Returns:
            Tuple of (NoiseModel, CustomNoiseModelBackend)
        """
        num_qubits = self.num_qubits

        # Convert scalars to arrays if needed
        T1s = self.T1s
        T2s = self.T2s
        if isinstance(T1s, (int, float)):
            T1s = np.full(num_qubits, T1s)
        if isinstance(T2s, (int, float)):
            T2s = np.full(num_qubits, T2s)

        # Truncate T2s <= 2*T1s (physical constraint)
        T2s = np.array([min(T2s[j], 2 * T1s[j]) for j in range(num_qubits)])

        # Extract instruction times
        time_rz = self.instruction_times.get("time_rz")
        time_sx = self.instruction_times.get("time_sx")
        time_x = self.instruction_times.get("time_x")
        time_cx = self.instruction_times.get("time_cx")
        time_reset = self.instruction_times.get("time_reset")
        time_measure = self.instruction_times.get("time_measure")

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

        # Create coherent errors
        overrotation_amount = self.overrotation_amount
        detuning_amount = self.detuning_amount

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

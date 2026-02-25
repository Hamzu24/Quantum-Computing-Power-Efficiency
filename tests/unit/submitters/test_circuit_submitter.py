"""
Unit tests for circuit diagram capture in CircuitSubmitter

Tests sample_circuit storage during submit_circuits() and the
get_sample_circuit_figure() method that produces matplotlib Figures.
"""

import pytest
import sys
import pathlib
from unittest.mock import Mock, patch, MagicMock

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent.parent.resolve()))

from _helpers.circuit_submitter import CircuitSubmitter


def _make_bare_submitter(device_name="noiseless_sim"):
    """Create a CircuitSubmitter without calling __init__ (avoids filesystem/backend)."""
    sub = object.__new__(CircuitSubmitter)
    sub.sample_circuit = None
    sub.device_name = device_name
    sub.backend = Mock()
    sub.backend.get_costs.return_value = (0.0, 0.0)
    sub.tasks = []
    sub.circuits_padded = 0
    sub.benchmark_name = "test_benchmark"
    sub.benchmark_path = "/tmp/test"
    sub.circuits_path = "/tmp/test/circuits"
    sub.device_date_path = "/tmp/test"
    return sub


class TestSampleCircuitInit:
    """Tests that sample_circuit is initialised to None."""

    @patch.object(CircuitSubmitter, "initialise")
    def test_sample_circuit_initialized_to_none(self, mock_init):
        """
        Given: A fresh CircuitSubmitter
        When: __init__ completes
        Expected: sample_circuit is None
        """
        sub = CircuitSubmitter("bench", "noiseless_sim")
        assert sub.sample_circuit is None


class TestSampleCircuitStorage:
    """Tests that submit_circuits stores the first circuit as sample_circuit."""

    def test_stores_first_circuit_from_qasm_strs(self, sample_qasm_list):
        """
        Given: device_name='noiseless_sim', sample_circuit is None
        When: submit_circuits called with qasm_strs
        Expected: sample_circuit is set from circuits[0].copy()
        """
        sub = _make_bare_submitter("noiseless_sim")
        sub.backend.device.run_batch.return_value = Mock(tasks=[])

        sub.submit_circuits(
            shots=100, qasm_strs=sample_qasm_list,
            skip_asking=True, print_summary=False
        )

        assert sub.sample_circuit is not None
        assert sub.sample_circuit.num_qubits == 2

    def test_stores_first_circuit_from_qasm_paths(self, tmp_path, sample_qasm_string):
        """
        Given: device_name='noiseless_sim', sample_circuit is None
        When: submit_circuits called with qasm_paths
        Expected: sample_circuit is set
        """
        qasm_file = tmp_path / "test.qasm"
        qasm_file.write_text(sample_qasm_string)

        sub = _make_bare_submitter("noiseless_sim")
        sub.backend.device.run_batch.return_value = Mock(tasks=[])

        sub.submit_circuits(
            shots=100, qasm_paths=[str(qasm_file)],
            skip_asking=True, print_summary=False
        )

        assert sub.sample_circuit is not None

    def test_stores_circuit_skip_transpilation_qiskit_sim(self, sample_qasm_list):
        """
        Given: device_name='noisy_sim', skip_transpilation=True
        When: submit_circuits called with qasm_strs
        Expected: sample_circuit is set
        """
        sub = _make_bare_submitter("noisy_sim")
        sub.backend.device.run_batch.return_value = Mock(tasks=[])
        # The skip_transpilation path checks __class__.__bases__[0].__name__
        # For qiskit sims it enters the branch that parses qasm_strs
        # Mock __class__ so __bases__[0].__name__ != 'AwsBackendHelper'
        sub.backend.__class__ = type("MockBackend", (object,), {})

        sub.submit_circuits(
            shots=100, qasm_strs=sample_qasm_list,
            skip_asking=True, skip_transpilation=True, print_summary=False
        )

        assert sub.sample_circuit is not None

    def test_stores_circuit_for_noiseless_sim(self, sample_qasm_list):
        """
        Given: device_name='noiseless_sim'
        When: submit_circuits called with qasm_strs
        Expected: sample_circuit is set
        """
        sub = _make_bare_submitter("noiseless_sim")
        sub.backend.device.run_batch.return_value = Mock(tasks=[])

        sub.submit_circuits(
            shots=100, qasm_strs=sample_qasm_list,
            skip_asking=True, print_summary=False
        )

        assert sub.sample_circuit is not None

    def test_circuit_stored_is_a_copy(self, sample_qasm_list):
        """
        Given: device_name='noiseless_sim', sample_circuit is None
        When: submit_circuits called
        Expected: stored circuit is a copy, not the original object
        """
        sub = _make_bare_submitter("noiseless_sim")
        sub.backend.device.run_batch.return_value = Mock(tasks=[])

        from qiskit import QuantumCircuit

        original_circuits = [QuantumCircuit.from_qasm_str(s) for s in sample_qasm_list]

        with patch("qiskit.QuantumCircuit.from_qasm_str", side_effect=original_circuits):
            sub.submit_circuits(
                shots=100, qasm_strs=sample_qasm_list,
                skip_asking=True, print_summary=False
            )

        # The stored circuit should not be the exact same object as original_circuits[0]
        # because .copy() was called
        assert sub.sample_circuit is not None
        assert sub.sample_circuit is not original_circuits[0]


class TestSampleCircuitNotOverwritten:
    """Tests that once sample_circuit is set, subsequent calls don't overwrite it."""

    def test_second_submit_does_not_overwrite(self, sample_qasm_list):
        """
        Given: sample_circuit already set to a sentinel
        When: submit_circuits called
        Expected: sentinel is preserved
        """
        sub = _make_bare_submitter("noiseless_sim")
        sub.backend.device.run_batch.return_value = Mock(tasks=[])

        sentinel = Mock(name="sentinel_circuit")
        sub.sample_circuit = sentinel

        sub.submit_circuits(
            shots=100, qasm_strs=sample_qasm_list,
            skip_asking=True, print_summary=False
        )

        assert sub.sample_circuit is sentinel

    def test_multiple_submits_preserve_first(self, sample_qasm_list):
        """
        Given: sample_circuit is None
        When: submit_circuits called twice
        Expected: sample_circuit is the first circuit, not overwritten by second call
        """
        sub = _make_bare_submitter("noiseless_sim")
        sub.backend.device.run_batch.return_value = Mock(tasks=[])

        sub.submit_circuits(
            shots=100, qasm_strs=sample_qasm_list,
            skip_asking=True, print_summary=False
        )

        first_sample = sub.sample_circuit
        assert first_sample is not None

        sub.submit_circuits(
            shots=100, qasm_strs=sample_qasm_list,
            skip_asking=True, print_summary=False
        )

        assert sub.sample_circuit is first_sample


class TestSampleCircuitEmptyList:
    """Tests edge cases with empty circuit lists."""

    def test_empty_qasm_strs_does_not_set_sample(self):
        """
        Given: sample_circuit is None
        When: submit_circuits called with qasm_strs=[]
        Expected: sample_circuit stays None, no crash
        """
        sub = _make_bare_submitter("noiseless_sim")
        sub.backend.device.run_batch.return_value = Mock(tasks=[])

        sub.submit_circuits(
            shots=100, qasm_strs=[],
            skip_asking=True, print_summary=False
        )

        assert sub.sample_circuit is None

    def test_empty_list_does_not_crash_when_already_set(self):
        """
        Given: sample_circuit already set
        When: submit_circuits called with qasm_strs=[]
        Expected: no error, sample_circuit unchanged
        """
        sub = _make_bare_submitter("noiseless_sim")
        sub.backend.device.run_batch.return_value = Mock(tasks=[])

        sentinel = Mock(name="existing_circuit")
        sub.sample_circuit = sentinel

        sub.submit_circuits(
            shots=100, qasm_strs=[],
            skip_asking=True, print_summary=False
        )

        assert sub.sample_circuit is sentinel


class TestGetSampleCircuitFigure:
    """Tests for get_sample_circuit_figure() method."""

    def test_returns_none_when_no_sample_circuit(self):
        """
        Given: sample_circuit is None
        When: get_sample_circuit_figure() called
        Expected: returns None
        """
        sub = _make_bare_submitter()
        result = sub.get_sample_circuit_figure()
        assert result is None

    def test_returns_figure_when_circuit_exists(self, sample_qiskit_circuit):
        """
        Given: sample_circuit is a real QuantumCircuit
        When: get_sample_circuit_figure() called
        Expected: returns a matplotlib.figure.Figure
        """
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib.figure import Figure

        sub = _make_bare_submitter()
        sub.sample_circuit = sample_qiskit_circuit

        result = sub.get_sample_circuit_figure()
        assert isinstance(result, Figure)

    def test_calls_draw_with_mpl_output(self):
        """
        Given: sample_circuit is a mock circuit
        When: get_sample_circuit_figure() called
        Expected: circuit.draw(output='mpl') is called
        """
        sub = _make_bare_submitter()
        mock_circuit = Mock()
        mock_circuit.draw.return_value = Mock(name="figure")
        sub.sample_circuit = mock_circuit

        result = sub.get_sample_circuit_figure()

        mock_circuit.draw.assert_called_once_with(output="mpl")
        assert result is mock_circuit.draw.return_value


class TestSampleCircuitCliffordSkip:
    """Tests that pauli_mode (Clifford metrics) skips sample_circuit storage."""

    @patch("_helpers.enhanced_circuit_submitter.craft_noise_model")
    @patch("_helpers.enhanced_circuit_submitter.read_config")
    @patch.object(CircuitSubmitter, "initialise")
    def test_pauli_mode_does_not_set_sample_circuit(
        self, mock_init, mock_read_config, mock_craft
    ):
        """
        Given: Enhanced submitter with pauli_mode=True
        When: submit_circuits called with stim_circuits
        Expected: sample_circuit stays None
        """
        from _helpers.enhanced_circuit_submitter import CircuitSubmitter as EnhancedSubmitter
        from _helpers.constants import CLIFFORD_METRICS

        if not CLIFFORD_METRICS:
            pytest.skip("No Clifford metrics defined in constants")

        benchmark_name = list(CLIFFORD_METRICS)[0]

        mock_read_config.return_value = {
            "power_configs": {"default_power_config": {"cx": 2}},
            "noise_models": {"default": {"type": "fake_backend", "name": "tokyo"}},
            "device_tracking": {"noisy_sim": 0},
        }
        mock_craft.return_value = (Mock(), Mock(), {})

        sub = object.__new__(EnhancedSubmitter)
        sub.sample_circuit = None
        sub.device_name = "stim_sim"
        sub.benchmark_name = benchmark_name
        sub.pauli_mode = True
        sub.backend = Mock()
        sub.tasks = []
        sub.circuits_padded = 0
        sub.total_gates = __import__("collections").Counter()
        sub.gate_history = []

        mock_stim_circuit = Mock()
        sub.backend.get_basis_gates.return_value = None
        sub.backend.device.run_batch.return_value = Mock(tasks=[])

        sub.submit_circuits(
            shots=100, stim_circuits=[mock_stim_circuit], print_summary=False
        )

        assert sub.sample_circuit is None

    @patch("_helpers.enhanced_circuit_submitter.craft_noise_model")
    @patch("_helpers.enhanced_circuit_submitter.read_config")
    @patch.object(CircuitSubmitter, "initialise")
    def test_non_pauli_mode_sets_sample_circuit(
        self, mock_init, mock_read_config, mock_craft, sample_qasm_list
    ):
        """
        Given: Enhanced submitter with pauli_mode=False
        When: submit_circuits called with qasm_strs
        Expected: sample_circuit is set via super()
        """
        from _helpers.enhanced_circuit_submitter import CircuitSubmitter as EnhancedSubmitter

        mock_read_config.return_value = {
            "power_configs": {"default_power_config": {"cx": 2}},
            "noise_models": {"default": {"type": "fake_backend", "name": "tokyo"}},
            "device_tracking": {"noiseless_sim": 0},
        }
        mock_craft.return_value = (Mock(), Mock(), {})

        sub = object.__new__(EnhancedSubmitter)
        sub.sample_circuit = None
        sub.device_name = "noiseless_sim"
        sub.benchmark_name = "non_clifford_benchmark"
        sub.pauli_mode = False
        sub.backend = Mock()
        sub.backend.get_costs.return_value = (0.0, 0.0)
        sub.backend.device.run_batch.return_value = Mock(tasks=[])
        sub.tasks = []
        sub.circuits_padded = 0
        sub.total_gates = __import__("collections").Counter()
        sub.gate_history = []
        sub.benchmark_path = "/tmp/test"
        sub.circuits_path = "/tmp/test/circuits"
        sub.device_date_path = "/tmp/test"

        sub.submit_circuits(
            shots=100, qasm_strs=sample_qasm_list,
            skip_asking=True, print_summary=False
        )

        assert sub.sample_circuit is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

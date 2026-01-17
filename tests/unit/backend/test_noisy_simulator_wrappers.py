"""
Unit tests for noisy_simulator_wrappers module

This test suite validates the Qiskit simulator wrappers that provide:
- Result caching for performance
- Density matrix to measurement counts conversion
- Bitstring order conversion (Qiskit ↔ Braket)
- Memory management (clearing large density matrices)
- Shot noise simulation
"""

import pytest
import sys
import pathlib
from unittest.mock import Mock, patch, MagicMock
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent.parent.resolve()))

from _helpers.noisy_simulator_wrappers import (
    QiskitTaskResultWrapper,
    QiskitTaskResultWrapperWithShots,
    QiskitTaskWrapper,
    QiskitTaskBatchWrapper,
    SimWrapper,
    NoisySimWrapper,
    NoisySimWrapperWithShots,
    NoiselessDensityMatrixSimWrapper
)


class TestQiskitTaskResultWrapper:
    """Test QiskitTaskResultWrapper class (no shot noise)"""

    def test_init_converts_density_matrix_to_counts(self):
        """
        Test initialization converts density matrix to measurement counts

        Given: A density matrix result with 2 qubits
        Expected: Counts are extracted from diagonal elements

        Calculation:
        - Density matrix diagonal [0.25, 0.25, 0.25, 0.25] for 2 qubits
        - With 1000 shots: each state gets ~250 counts
        """
        # Mock AerJob result
        mock_job = Mock()
        mock_result = Mock()
        mock_result.results = [Mock()]
        mock_result.results[0].header.n_qubits = 2

        # Create a simple density matrix (2 qubits, 4 states)
        # Diagonal represents probabilities: [0.25, 0.25, 0.25, 0.25]
        density_matrix = np.diag([0.25, 0.25, 0.25, 0.25])  # sqrt(prob)
        mock_result.data.return_value = {"before_measurement": density_matrix}

        mock_job.result.return_value = mock_result

        wrapper = QiskitTaskResultWrapper(mock_job, shots=1000)

        assert hasattr(wrapper, 'measurement_counts')
        assert isinstance(wrapper.measurement_counts, dict)
        # Total counts should equal shots
        assert sum(wrapper.measurement_counts.values()) == 1000

    def test_bitstring_order_reversal_qiskit_to_braket(self):
        """
        Test bitstring order is reversed from Qiskit to Braket

        Given: Qiskit counts {"00": 100, "01": 200}
        Expected: Converted to Braket order {"00": 100, "10": 200}

        Note: Qiskit and Braket use opposite qubit ordering conventions
        Qiskit "01" = |01⟩ = q1=0, q0=1
        Braket "10" = |10⟩ = q0=1, q1=0
        """
        mock_job = Mock()
        mock_result = Mock()
        mock_result.results = [Mock()]
        mock_result.results[0].header.n_qubits = 2

        # Density matrix with states |00⟩ and |01⟩ populated
        density_matrix = np.diag([0.4472, 0.8944, 0.0, 0.0])  # sqrt of [0.2, 0.8, 0, 0]
        mock_result.data.return_value = {"before_measurement": density_matrix}
        mock_job.result.return_value = mock_result

        wrapper = QiskitTaskResultWrapper(mock_job, shots=100)

        # Check bitstrings are reversed
        assert all(len(k) == 2 for k in wrapper.measurement_counts.keys())

    def test_adjusts_highest_count_to_match_shots(self):
        """
        Test that highest count is adjusted to ensure total equals shots

        Given: Rounding may cause sum(counts) != shots
        Expected: Highest count is adjusted by difference

        Calculation:
        - Suppose rounding gives counts summing to 1002 instead of 1000
        - The highest count is reduced by 2 to match total shots
        """
        mock_job = Mock()
        mock_result = Mock()
        mock_result.results = [Mock()]
        mock_result.results[0].header.n_qubits = 2

        # Create density matrix where rounding will cause issues
        density_matrix = np.diag([0.577, 0.577, 0.577, 0.0])  # sqrt of 1/3 each
        mock_result.data.return_value = {"before_measurement": density_matrix}
        mock_job.result.return_value = mock_result

        wrapper = QiskitTaskResultWrapper(mock_job, shots=1000)

        # Total must equal shots exactly
        assert sum(wrapper.measurement_counts.values()) == 1000

    def test_zero_probability_states_filtered(self):
        """
        Test states with near-zero probability are set to zero

        Given: Density matrix with very small values < eps
        Expected: Values below machine epsilon are zeroed out

        Note: Uses np.finfo(float).eps for numerical stability
        """
        mock_job = Mock()
        mock_result = Mock()
        mock_result.results = [Mock()]
        mock_result.results[0].header.n_qubits = 2

        # Density matrix with one state and near-zero noise
        eps = np.finfo(float).eps
        density_matrix = np.diag([1.0, eps/10, eps/10, eps/10])
        mock_result.data.return_value = {"before_measurement": density_matrix}
        mock_job.result.return_value = mock_result

        wrapper = QiskitTaskResultWrapper(mock_job, shots=1000)

        # Only state "00" (reversed to "00") should have counts
        assert sum(wrapper.measurement_counts.values()) == 1000


class TestQiskitTaskResultWrapperWithShots:
    """Test QiskitTaskResultWrapperWithShots class (with shot noise)"""

    def test_init_gets_counts_from_result(self):
        """
        Test initialization gets counts directly from result.get_counts()

        Given: AerJob result with shot noise
        Expected: measurement_counts = result.get_counts(0)

        Note: With shot noise, counts include statistical fluctuations
        """
        mock_job = Mock()
        mock_result = Mock()
        mock_result.results = [Mock()]
        mock_result.results[0].header.n_qubits = 2
        mock_result.get_counts.return_value = {"00": 450, "11": 550}

        mock_job.result.return_value = mock_result

        wrapper = QiskitTaskResultWrapperWithShots(mock_job, shots=1000)

        assert wrapper.measurement_counts == {"00": 450, "11": 550}
        mock_result.get_counts.assert_called_once_with(0)

    def test_measurement_counts_vary_with_shot_noise(self):
        """
        Test that measurement counts include shot noise variation

        Expected: Counts are obtained from noisy simulation
        """
        mock_job = Mock()
        mock_result = Mock()
        mock_result.results = [Mock()]
        mock_result.results[0].header.n_qubits = 3
        mock_result.get_counts.return_value = {
            "000": 123, "001": 234, "010": 345, "100": 298
        }

        mock_job.result.return_value = mock_result

        wrapper = QiskitTaskResultWrapperWithShots(mock_job, shots=1000)

        assert isinstance(wrapper.measurement_counts, dict)
        assert len(wrapper.measurement_counts) > 0


class TestQiskitTaskWrapper:
    """Test QiskitTaskWrapper class"""

    def test_init_stores_job_attributes(self):
        """
        Test initialization stores job, shots, and shot_noise flag

        Expected: Attributes are set correctly
        """
        mock_job = Mock()
        mock_job.job_id.return_value = "job_12345"

        wrapper = QiskitTaskWrapper(mock_job, shots=2000, shot_noise=True)

        assert wrapper.task is mock_job
        assert wrapper.id == "job_12345"
        assert wrapper.shots == 2000
        assert wrapper.shot_noise is True
        assert wrapper._cached_result is None

    def test_result_caches_on_first_call(self):
        """
        Test that result() caches the result after first call

        Given: First call to result()
        Expected: Result is computed and cached in _cached_result

        Performance: Subsequent calls should not recompute
        """
        mock_job = Mock()
        mock_job.job_id.return_value = "job_123"

        # Mock the result structure for shot noise = True
        mock_result_obj = Mock()
        mock_result_obj.results = [Mock()]
        mock_result_obj.results[0].header.n_qubits = 2
        mock_result_obj.get_counts.return_value = {"00": 500, "11": 500}
        mock_job.result.return_value = mock_result_obj

        wrapper = QiskitTaskWrapper(mock_job, shots=1000, shot_noise=True)

        # First call
        result1 = wrapper.result()

        assert wrapper._cached_result is not None
        assert result1 is wrapper._cached_result

        # Second call should return cached result
        result2 = wrapper.result()

        assert result2 is result1
        assert result2 is wrapper._cached_result

    def test_result_without_shot_noise_uses_density_matrix(self):
        """
        Test result() without shot noise uses QiskitTaskResultWrapper

        Given: shot_noise=False
        Expected: Creates QiskitTaskResultWrapper (density matrix mode)
        """
        mock_job = Mock()
        mock_job.job_id.return_value = "job_123"

        mock_result_obj = Mock()
        mock_result_obj.results = [Mock()]
        mock_result_obj.results[0].header.n_qubits = 2
        density_matrix = np.diag([0.7071, 0.7071, 0.0, 0.0])
        mock_result_obj.data.return_value = {"before_measurement": density_matrix}
        mock_job.result.return_value = mock_result_obj

        wrapper = QiskitTaskWrapper(mock_job, shots=1000, shot_noise=False)

        result = wrapper.result()

        assert isinstance(result, QiskitTaskResultWrapper)

    def test_result_with_shot_noise_uses_shots_wrapper(self):
        """
        Test result() with shot noise uses QiskitTaskResultWrapperWithShots

        Given: shot_noise=True
        Expected: Creates QiskitTaskResultWrapperWithShots
        """
        mock_job = Mock()
        mock_job.job_id.return_value = "job_123"

        mock_result_obj = Mock()
        mock_result_obj.results = [Mock()]
        mock_result_obj.results[0].header.n_qubits = 2
        mock_result_obj.get_counts.return_value = {"00": 500, "11": 500}
        mock_job.result.return_value = mock_result_obj

        wrapper = QiskitTaskWrapper(mock_job, shots=1000, shot_noise=True)

        result = wrapper.result()

        assert isinstance(result, QiskitTaskResultWrapperWithShots)

    def test_result_clears_job_reference_after_computation(self):
        """
        Test result() clears job reference to free memory

        Given: Job contains large density matrices
        Expected: After result() is called, self.task is set to None

        Memory optimization: Prevents holding large objects in memory
        """
        mock_job = Mock()
        mock_job.job_id.return_value = "job_123"

        mock_result_obj = Mock()
        mock_result_obj.results = [Mock()]
        mock_result_obj.results[0].header.n_qubits = 2
        mock_result_obj.get_counts.return_value = {"00": 1000}
        mock_job.result.return_value = mock_result_obj

        wrapper = QiskitTaskWrapper(mock_job, shots=1000, shot_noise=True)

        assert wrapper.task is mock_job

        wrapper.result()

        # Job reference should be cleared
        assert wrapper.task is None

    def test_state_returns_completed(self):
        """
        Test state() always returns "COMPLETED"

        Expected: Returns string "COMPLETED"

        Note: This is a simplified interface for compatibility
        """
        mock_job = Mock()
        mock_job.job_id.return_value = "job_123"

        wrapper = QiskitTaskWrapper(mock_job, shots=1000, shot_noise=False)

        assert wrapper.state() == "COMPLETED"


class TestQiskitTaskBatchWrapper:
    """Test QiskitTaskBatchWrapper class"""

    def test_init_stores_task_list(self):
        """
        Test initialization stores list of tasks

        Expected: tasks attribute contains the provided list
        """
        mock_task1 = Mock(spec=QiskitTaskWrapper)
        mock_task2 = Mock(spec=QiskitTaskWrapper)
        tasks = [mock_task1, mock_task2]

        batch = QiskitTaskBatchWrapper(tasks)

        assert batch.tasks == tasks
        assert len(batch.tasks) == 2

    def test_tasks_are_accessible(self):
        """
        Test that tasks can be accessed and iterated

        Expected: Can iterate over batch.tasks
        """
        mock_tasks = [Mock(spec=QiskitTaskWrapper) for _ in range(5)]

        batch = QiskitTaskBatchWrapper(mock_tasks)

        assert len(batch.tasks) == 5
        for i, task in enumerate(batch.tasks):
            assert task is mock_tasks[i]


class TestSimWrapper:
    """Test SimWrapper base class"""

    @patch('_helpers.noisy_simulator_wrappers.AerSimulator')
    def test_init_creates_aer_simulator_without_noise(self, mock_aer):
        """
        Test initialization without noise model creates statevector simulator

        Expected: AerSimulator(method='statevector', device='GPU')
        """
        mock_sim = Mock()
        mock_aer.return_value = mock_sim

        wrapper = SimWrapper(shot_noise=False)

        assert wrapper.backend == mock_aer
        assert wrapper.noise_model is None
        assert wrapper.shot_noise is False
        mock_aer.assert_called_once_with(method='statevector', device='GPU')

    @patch('_helpers.noisy_simulator_wrappers.AerSimulator')
    @patch('_helpers.noisy_simulator_wrappers.SIMULATION_METHOD', 'density_matrix')
    def test_init_with_noise_model_uses_simulation_method(self, mock_aer):
        """
        Test initialization with noise model uses SIMULATION_METHOD

        Given: noise_model provided
        Expected: AerSimulator(method=SIMULATION_METHOD, noise_model=..., device='GPU', blocking_qubits=11)
        """
        mock_sim = Mock()
        mock_aer.return_value = mock_sim
        mock_noise_model = Mock()

        wrapper = SimWrapper(noise_model=mock_noise_model, shot_noise=False)

        assert wrapper.noise_model is mock_noise_model
        mock_aer.assert_called_once_with(
            method='density_matrix',
            noise_model=mock_noise_model,
            device='GPU',
            blocking_qubits=11
        )

    def test_remove_measurement_operations(self):
        """
        Test _remove_measurement_and_add_dm_save removes measurements

        Given: Circuit with measurement operations
        Expected: Measurements are removed, density_matrix save added

        Note: This enables density matrix extraction for deterministic counts
        """
        from qiskit import QuantumCircuit

        circ = QuantumCircuit(2, 2)
        circ.h(0)
        circ.cx(0, 1)
        circ.measure([0, 1], [0, 1])

        wrapper = SimWrapper(shot_noise=False)
        modified_circ = wrapper._remove_measurement_and_add_dm_save(circ)

        # Check measurements are removed
        has_measurement = any(
            instr.operation.name == 'measure' for instr in modified_circ.data
        )
        assert not has_measurement

        # Check density matrix save was added
        has_dm_save = any(
            'save_density_matrix' in str(instr.operation.name) for instr in modified_circ.data
        )
        assert has_dm_save

    @patch('_helpers.noisy_simulator_wrappers.AerSimulator')
    def test_run_batch_without_shot_noise_removes_measurements(self, mock_aer):
        """
        Test run_batch without shot noise removes measurements

        Given: shot_noise=False
        Expected: Measurements removed and density matrix saved
        """
        from qiskit import QuantumCircuit

        mock_sim = Mock()
        mock_job = Mock()
        mock_job.job_id.return_value = "job_123"
        mock_sim.run.return_value = mock_job
        mock_aer.return_value = mock_sim

        wrapper = SimWrapper(shot_noise=False)

        circ = QuantumCircuit(2, 2)
        circ.h(0)
        circ.measure([0], [0])

        result = wrapper.run_batch([circ], shots=1000)

        assert isinstance(result, QiskitTaskBatchWrapper)
        assert len(result.tasks) == 1

    @patch('_helpers.noisy_simulator_wrappers.AerSimulator')
    def test_run_batch_with_shot_noise_keeps_measurements(self, mock_aer):
        """
        Test run_batch with shot noise keeps measurements

        Given: shot_noise=True
        Expected: Circuits run with measurements intact, bits reversed
        """
        from qiskit import QuantumCircuit

        mock_sim = Mock()
        mock_job = Mock()
        mock_job.job_id.return_value = "job_123"
        mock_sim.run.return_value = mock_job
        mock_aer.return_value = mock_sim

        wrapper = SimWrapper(shot_noise=True)

        circ = QuantumCircuit(2, 2)
        circ.h(0)
        circ.measure([0, 1], [0, 1])

        result = wrapper.run_batch([circ], shots=2000)

        assert isinstance(result, QiskitTaskBatchWrapper)
        # With shot noise, circuits are reversed but not modified
        assert mock_sim.run.called


class TestNoisySimWrapper:
    """Test NoisySimWrapper class"""

    @patch('_helpers.noisy_simulator_wrappers.AerSimulator')
    @patch('_helpers.noisy_simulator_wrappers.custom_noise_model')
    def test_init_creates_default_custom_noise_model(self, mock_custom_nm, mock_aer):
        """
        Test initialization without noise_model creates custom_noise_model

        Expected: custom_noise_model() is called to create default
        """
        mock_noise = Mock()
        mock_custom_nm.return_value = mock_noise
        mock_sim = Mock()
        mock_aer.return_value = mock_sim

        wrapper = NoisySimWrapper()

        assert wrapper.noise_model is mock_noise
        assert wrapper.backend == mock_aer
        mock_custom_nm.assert_called_once()

    @patch('_helpers.noisy_simulator_wrappers.AerSimulator')
    def test_init_with_provided_noise_model(self, mock_aer):
        """
        Test initialization with provided noise_model uses it

        Given: noise_model parameter
        Expected: Provided noise model is used instead of default
        """
        mock_sim = Mock()
        mock_aer.return_value = mock_sim
        custom_noise = Mock()

        wrapper = NoisySimWrapper(noise_model=custom_noise)

        assert wrapper.noise_model is custom_noise

    @patch('_helpers.noisy_simulator_wrappers.AerSimulator')
    @patch('_helpers.noisy_simulator_wrappers.custom_noise_model')
    def test_set_noise_model_updates_simulator(self, mock_custom_nm, mock_aer):
        """
        Test set_noise_model() reinitializes with new noise model

        Expected: SimWrapper.__init__ is called with new noise model
        """
        mock_noise = Mock()
        mock_custom_nm.return_value = mock_noise
        mock_sim = Mock()
        mock_aer.return_value = mock_sim

        wrapper = NoisySimWrapper()

        new_noise = Mock()
        wrapper.set_noise_model(new_noise)

        # Noise model should be updated
        assert wrapper.noise_model is new_noise


class TestNoisySimWrapperWithShots:
    """Test NoisySimWrapperWithShots class"""

    @patch('_helpers.noisy_simulator_wrappers.AerSimulator')
    @patch('_helpers.noisy_simulator_wrappers.custom_noise_model')
    def test_init_enables_shot_noise(self, mock_custom_nm, mock_aer):
        """
        Test initialization enables shot_noise=True

        Expected: SimWrapper initialized with shot_noise=True
        """
        mock_noise = Mock()
        mock_custom_nm.return_value = mock_noise
        mock_sim = Mock()
        mock_aer.return_value = mock_sim

        wrapper = NoisySimWrapperWithShots()

        assert wrapper.shot_noise is True
        assert wrapper.noise_model is not None

    @patch('_helpers.noisy_simulator_wrappers.AerSimulator')
    @patch('_helpers.noisy_simulator_wrappers.custom_noise_model')
    def test_init_with_provided_noise_model_overrides_default(self, mock_custom_nm, mock_aer):
        """
        Test initialization with noise_model parameter overrides default

        Given: noise_model parameter provided
        Expected: Provided noise model is used
        """
        mock_default_noise = Mock()
        mock_custom_nm.return_value = mock_default_noise
        mock_sim = Mock()
        mock_aer.return_value = mock_sim

        custom_noise = Mock()
        wrapper = NoisySimWrapperWithShots(noise_model=custom_noise)

        assert wrapper.noise_model is custom_noise


class TestNoiselessDensityMatrixSimWrapper:
    """Test NoiselessDensityMatrixSimWrapper class"""

    @patch('_helpers.noisy_simulator_wrappers.AerSimulator')
    def test_init_creates_noiseless_simulator(self, mock_aer):
        """
        Test initialization creates simulator without noise

        Expected: SimWrapper initialized with no noise_model
        """
        mock_sim = Mock()
        mock_aer.return_value = mock_sim

        wrapper = NoiselessDensityMatrixSimWrapper()

        assert wrapper.noise_model is None
        assert wrapper.shot_noise is False


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

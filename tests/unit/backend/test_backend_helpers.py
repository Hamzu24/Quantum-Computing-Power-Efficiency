"""
Unit tests for backend_helpers module

This test suite validates the various backend helper classes that provide
abstraction over different quantum computing backends including simulators
and hardware devices.
"""

import pytest
import sys
import pathlib
from unittest.mock import Mock, patch, MagicMock
from typing import Tuple

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent.parent.resolve()))

from _helpers.backend_helpers import (
    AwsBackendHelper,
    IdealSimulatorHelper,
    NoisySimulatorHelper,
    NoisySimulatorHelperWithShots,
    DensityMatrixIdealSimulatorHelper,
    OQCLucyHelper,
    OQCDirectHelper,
    IonQHarmonyHelper,
    IonQAriaHelper,
    get_backend_helper
)


class TestAwsBackendHelperAbstractMethods:
    """Test abstract base class AwsBackendHelper"""

    def test_cannot_instantiate_abstract_class(self):
        """
        Test that AwsBackendHelper cannot be instantiated directly

        Expected: Raises TypeError due to abstract methods
        """
        # AwsBackendHelper has abstract methods, cannot instantiate
        with pytest.raises(TypeError, match="abstract"):
            AwsBackendHelper()

    def test_get_device_calibration_returns_json(self):
        """
        Test that get_device_calibration returns device properties JSON

        Given: A concrete implementation with a mock device
        Expected: Returns device.properties.json()
        """
        class ConcreteHelper(AwsBackendHelper):
            def get_qiskit_backend(self):
                return Mock()
            def get_basis_gates(self):
                return []
            def get_costs(self):
                return (0, 0)

        helper = ConcreteHelper()
        mock_device = Mock()
        mock_device.properties.json.return_value = '{"device": "test"}'
        helper.device = mock_device

        result = helper.get_device_calibration()

        assert result == '{"device": "test"}'
        mock_device.properties.json.assert_called_once()


class TestIdealSimulatorHelper:
    """Test IdealSimulatorHelper class"""

    def test_init_creates_local_simulator(self):
        """
        Test initialization creates Braket LocalSimulator

        Expected: LocalSimulator() is instantiated

        Note: LocalSimulator is imported inside __init__, so we verify
        the device attribute exists and is from Braket
        """
        with patch('braket.devices.LocalSimulator') as mock_local_sim:
            mock_instance = Mock()
            mock_local_sim.return_value = mock_instance

            helper = IdealSimulatorHelper()

            assert helper.device is not None
            # Device should be set (either mock or real LocalSimulator)

    @patch('_helpers.backend_helpers.LocalSimulator')
    def test_get_device_calibration_returns_string(self, mock_local_sim):
        """
        Test get_device_calibration returns descriptive string

        Expected: Returns "Local simulator with no noise"
        """
        helper = IdealSimulatorHelper()
        result = helper.get_device_calibration()

        assert result == "Local simulator with no noise"
        assert isinstance(result, str)

    @patch('_helpers.backend_helpers.LocalSimulator')
    @patch('_helpers.backend_helpers.BraketLocalBackend')
    def test_get_qiskit_backend_returns_braket_local(self, mock_braket_local, mock_local_sim):
        """
        Test get_qiskit_backend returns BraketLocalBackend

        Expected: Returns BraketLocalBackend instance
        """
        mock_backend = Mock()
        mock_braket_local.return_value = mock_backend

        helper = IdealSimulatorHelper()
        result = helper.get_qiskit_backend()

        assert result is mock_backend
        mock_braket_local.assert_called_once()

    @patch('_helpers.backend_helpers.LocalSimulator')
    def test_get_basis_gates_returns_standard_gates(self, mock_local_sim):
        """
        Test get_basis_gates returns standard gate set

        Expected: Returns ["rx", "ry", "rz", "h", "cx", "id"]
        """
        helper = IdealSimulatorHelper()
        result = helper.get_basis_gates()

        expected = ["rx", "ry", "rz", "h", "cx", "id"]
        assert result == expected
        assert isinstance(result, list)
        assert all(isinstance(gate, str) for gate in result)

    @patch('_helpers.backend_helpers.LocalSimulator')
    def test_get_costs_returns_zero(self, mock_local_sim):
        """
        Test get_costs returns (0, 0) for local simulator

        Expected: Returns (cost_per_circuit=0, cost_per_shot=0)
        """
        helper = IdealSimulatorHelper()
        result = helper.get_costs()

        assert result == (0, 0)
        assert isinstance(result, tuple)
        assert len(result) == 2


class TestNoisySimulatorHelper:
    """Test NoisySimulatorHelper class"""

    @patch('_helpers.backend_helpers.NoisySimWrapper')
    def test_init_without_noise_model(self, mock_wrapper):
        """
        Test initialization without providing noise_model

        Expected: NoisySimWrapper() is created without arguments
        """
        mock_instance = Mock()
        mock_wrapper.return_value = mock_instance

        helper = NoisySimulatorHelper()

        assert helper.device is mock_instance
        assert helper.name == "noisy_sim"
        mock_wrapper.assert_called_once_with()

    @patch('_helpers.backend_helpers.NoisySimWrapper')
    def test_init_with_noise_model(self, mock_wrapper):
        """
        Test initialization with custom noise_model

        Given: noise_model kwarg provided
        Expected: NoisySimWrapper(noise_model=...) is created
        """
        mock_instance = Mock()
        mock_wrapper.return_value = mock_instance
        mock_noise_model = Mock()

        helper = NoisySimulatorHelper(noise_model=mock_noise_model)

        assert helper.device is mock_instance
        assert helper.noise_model is mock_noise_model
        mock_wrapper.assert_called_once_with(noise_model=mock_noise_model)

    @patch('_helpers.backend_helpers.NoisySimWrapper')
    def test_get_device_calibration_includes_noise_model_str(self, mock_wrapper):
        """
        Test get_device_calibration includes noise model string

        Expected: Returns string containing str(self.device.noise_model)
        """
        mock_device = Mock()
        mock_device.noise_model = "TestNoiseModel"
        mock_wrapper.return_value = mock_device

        helper = NoisySimulatorHelper()
        result = helper.get_device_calibration()

        assert "Local simulator with two-qubit noise model" in result
        assert "TestNoiseModel" in result

    @patch('_helpers.backend_helpers.NoisySimWrapper')
    @patch('_helpers.backend_helpers.BraketLocalBackend')
    def test_get_qiskit_backend_returns_braket_local(self, mock_braket, mock_wrapper):
        """
        Test get_qiskit_backend returns BraketLocalBackend

        Expected: Returns BraketLocalBackend instance
        """
        mock_backend = Mock()
        mock_braket.return_value = mock_backend

        helper = NoisySimulatorHelper()
        result = helper.get_qiskit_backend()

        assert result is mock_backend

    @patch('_helpers.backend_helpers.NoisySimWrapper')
    def test_get_basis_gates_with_noise_model(self, mock_wrapper):
        """
        Test get_basis_gates returns noise model's basis gates

        Given: noise_model with basis_gates attribute
        Expected: Returns noise_model.basis_gates
        """
        mock_device = Mock()
        mock_noise_model = Mock()
        mock_noise_model.basis_gates = ["sx", "rz", "cx", "custom_gate"]
        mock_wrapper.return_value = mock_device

        helper = NoisySimulatorHelper(noise_model=mock_noise_model)
        helper.device.noise_model = mock_noise_model
        result = helper.get_basis_gates()

        assert result == ["sx", "rz", "cx", "custom_gate"]

    @patch('_helpers.backend_helpers.NoisySimWrapper')
    def test_get_basis_gates_without_noise_model(self, mock_wrapper):
        """
        Test get_basis_gates returns default gates when no noise model

        Expected: Returns ["sx", "rz", "cx", "id"]
        """
        helper = NoisySimulatorHelper()
        # Ensure noise_model is None or doesn't exist
        if hasattr(helper, 'noise_model'):
            helper.noise_model = None

        result = helper.get_basis_gates()

        assert result == ["sx", "rz", "cx", "id"]

    @patch('_helpers.backend_helpers.NoisySimWrapper')
    def test_get_costs_returns_zero(self, mock_wrapper):
        """
        Test get_costs returns (0, 0) for noisy simulator

        Expected: Returns (0, 0)
        """
        helper = NoisySimulatorHelper()
        result = helper.get_costs()

        assert result == (0, 0)


class TestNoisySimulatorHelperWithShots:
    """Test NoisySimulatorHelperWithShots class"""

    @patch('_helpers.backend_helpers.NoisySimWrapperWithShots')
    def test_init_without_noise_model(self, mock_wrapper):
        """
        Test initialization without providing noise_model

        Expected: NoisySimWrapperWithShots() is created without arguments
        """
        mock_instance = Mock()
        mock_wrapper.return_value = mock_instance

        helper = NoisySimulatorHelperWithShots()

        assert helper.device is mock_instance
        assert helper.name == "noisy_sim_with_shots"

    @patch('_helpers.backend_helpers.NoisySimWrapperWithShots')
    def test_init_with_noise_model(self, mock_wrapper):
        """
        Test initialization with custom noise_model

        Given: noise_model kwarg provided
        Expected: NoisySimWrapperWithShots(noise_model=...) is created
        """
        mock_instance = Mock()
        mock_wrapper.return_value = mock_instance
        mock_noise_model = Mock()

        helper = NoisySimulatorHelperWithShots(noise_model=mock_noise_model)

        assert helper.device is mock_instance
        assert helper.noise_model is mock_noise_model

    @patch('_helpers.backend_helpers.NoisySimWrapperWithShots')
    def test_get_device_calibration_mentions_shot_noise(self, mock_wrapper):
        """
        Test get_device_calibration mentions shot noise

        Expected: Returns string containing "with shots noise"
        """
        mock_device = Mock()
        mock_device.noise_model = "TestNoiseModel"
        mock_wrapper.return_value = mock_device

        helper = NoisySimulatorHelperWithShots()
        result = helper.get_device_calibration()

        assert "with shots noise" in result
        assert isinstance(result, str)

    @patch('_helpers.backend_helpers.NoisySimWrapperWithShots')
    def test_get_costs_returns_zero(self, mock_wrapper):
        """
        Test get_costs returns (0, 0)

        Expected: Returns (0, 0)
        """
        helper = NoisySimulatorHelperWithShots()
        result = helper.get_costs()

        assert result == (0, 0)


class TestDensityMatrixIdealSimulatorHelper:
    """Test DensityMatrixIdealSimulatorHelper class"""

    @patch('_helpers.backend_helpers.NoiselessDensityMatrixSimWrapper')
    def test_init_creates_noiseless_wrapper(self, mock_wrapper):
        """
        Test initialization creates NoiselessDensityMatrixSimWrapper

        Expected: NoiselessDensityMatrixSimWrapper() is instantiated
        """
        mock_instance = Mock()
        mock_wrapper.return_value = mock_instance

        helper = DensityMatrixIdealSimulatorHelper()

        assert helper.device is mock_instance
        assert helper.name == "noiseless_sim"

    @patch('_helpers.backend_helpers.NoiselessDensityMatrixSimWrapper')
    def test_get_device_calibration_returns_string(self, mock_wrapper):
        """
        Test get_device_calibration returns descriptive string

        Expected: Returns "Local noiseless simulator with perfect shots:\n"
        """
        helper = DensityMatrixIdealSimulatorHelper()
        result = helper.get_device_calibration()

        assert result == "Local noiseless simulator with perfect shots:\n"

    @patch('_helpers.backend_helpers.NoiselessDensityMatrixSimWrapper')
    def test_get_basis_gates_returns_standard_gates(self, mock_wrapper):
        """
        Test get_basis_gates returns standard gate set

        Expected: Returns ["rx", "ry", "rz", "cx", "id"]
        """
        helper = DensityMatrixIdealSimulatorHelper()
        result = helper.get_basis_gates()

        expected = ["rx", "ry", "rz", "cx", "id"]
        assert result == expected


class TestOQCLucyHelper:
    """Test OQCLucyHelper class (deprecated device)"""

    @patch('_helpers.backend_helpers.AwsDevice')
    def test_init_creates_aws_device(self, mock_aws_device):
        """
        Test initialization creates AwsDevice with Lucy ARN

        Expected: AwsDevice is created with correct ARN
        """
        mock_device = Mock()
        mock_aws_device.return_value = mock_device

        helper = OQCLucyHelper()

        assert helper.device is mock_device
        mock_aws_device.assert_called_once_with(
            "arn:aws:braket:eu-west-2::device/qpu/oqc/Lucy"
        )

    @patch('_helpers.backend_helpers.AwsDevice')
    @patch('_helpers.backend_helpers.AWSBraketProvider')
    def test_get_qiskit_backend_returns_lucy_backend(self, mock_provider_class, mock_aws_device):
        """
        Test get_qiskit_backend returns Lucy backend from provider

        Expected: Returns AWSBraketProvider().get_backend("Lucy")
        """
        mock_backend = Mock()
        mock_provider = Mock()
        mock_provider.get_backend.return_value = mock_backend
        mock_provider_class.return_value = mock_provider

        helper = OQCLucyHelper()
        result = helper.get_qiskit_backend()

        assert result is mock_backend
        mock_provider.get_backend.assert_called_once_with("Lucy")

    @patch('_helpers.backend_helpers.AwsDevice')
    def test_get_basis_gates_returns_oqc_gates(self, mock_aws_device):
        """
        Test get_basis_gates returns OQC-specific gate set

        Expected: Returns ["ecr", "i", "rz", "sx", "x"]
        """
        helper = OQCLucyHelper()
        result = helper.get_basis_gates()

        expected = ["ecr", "i", "rz", "sx", "x"]
        assert result == expected

    @patch('_helpers.backend_helpers.AwsDevice')
    def test_get_costs_returns_lucy_costs(self, mock_aws_device):
        """
        Test get_costs returns Lucy pricing

        Expected: Returns (0.3, 0.00035)
        """
        helper = OQCLucyHelper()
        result = helper.get_costs()

        assert result == (0.3, 0.00035)
        assert isinstance(result[0], (int, float))
        assert isinstance(result[1], (int, float))


class TestIonQAriaHelper:
    """Test IonQAriaHelper class"""

    @patch('_helpers.backend_helpers.AwsDevice')
    def test_init_creates_aws_device_with_aria_arn(self, mock_aws_device):
        """
        Test initialization creates AwsDevice with Aria ARN

        Expected: AwsDevice is created with correct Aria-1 ARN
        """
        mock_device = Mock()
        mock_aws_device.return_value = mock_device

        helper = IonQAriaHelper()

        assert helper.device is mock_device
        mock_aws_device.assert_called_once_with(
            "arn:aws:braket:us-east-1::device/qpu/ionq/Aria-1"
        )

    @patch('_helpers.backend_helpers.AwsDevice')
    @patch('_helpers.backend_helpers.qiskit_gate_names_to_braket_gates', {})
    def test_get_basis_gates_configures_gate_mappings(self, mock_aws_device):
        """
        Test get_basis_gates configures Qiskit to Braket gate mappings

        Expected: Returns ["x", "y", "sx", "h", "rxx"] and sets up gate equivalences
        """
        from _helpers.backend_helpers import qiskit_gate_names_to_braket_gates

        helper = IonQAriaHelper()
        result = helper.get_basis_gates()

        expected = ["x", "y", "sx", "h", "rxx"]
        assert result == expected

        # Verify gate mappings were added
        assert "x" in qiskit_gate_names_to_braket_gates
        assert "y" in qiskit_gate_names_to_braket_gates
        assert "sx" in qiskit_gate_names_to_braket_gates

    @patch('_helpers.backend_helpers.AwsDevice')
    def test_get_costs_returns_aria_costs(self, mock_aws_device):
        """
        Test get_costs returns Aria pricing

        Expected: Returns (0.1, 0.03)

        Note: Aria costs are higher per shot than Harmony (0.03 vs 0.01)
        """
        helper = IonQAriaHelper()
        result = helper.get_costs()

        assert result == (0.1, 0.03)


class TestGetBackendHelper:
    """Test get_backend_helper factory function"""

    @patch('_helpers.backend_helpers.LocalSimulator')
    def test_get_backend_helper_simulator(self, mock_local_sim):
        """
        Test get_backend_helper("simulator") returns IdealSimulatorHelper

        Expected: Returns IdealSimulatorHelper instance
        """
        result = get_backend_helper("simulator")

        assert isinstance(result, IdealSimulatorHelper)

    @patch('_helpers.backend_helpers.NoisySimWrapper')
    def test_get_backend_helper_noisy_sim(self, mock_wrapper):
        """
        Test get_backend_helper("noisy_sim") returns NoisySimulatorHelper

        Expected: Returns NoisySimulatorHelper instance
        """
        result = get_backend_helper("noisy_sim")

        assert isinstance(result, NoisySimulatorHelper)

    @patch('_helpers.backend_helpers.NoisySimWrapperWithShots')
    def test_get_backend_helper_noisy_sim_with_shots(self, mock_wrapper):
        """
        Test get_backend_helper("noisy_sim_with_shots") returns correct helper

        Expected: Returns NoisySimulatorHelperWithShots instance
        """
        result = get_backend_helper("noisy_sim_with_shots")

        assert isinstance(result, NoisySimulatorHelperWithShots)

    @patch('_helpers.backend_helpers.NoiselessDensityMatrixSimWrapper')
    def test_get_backend_helper_noiseless_sim(self, mock_wrapper):
        """
        Test get_backend_helper("noiseless_sim") returns correct helper

        Expected: Returns DensityMatrixIdealSimulatorHelper instance
        """
        result = get_backend_helper("noiseless_sim")

        assert isinstance(result, DensityMatrixIdealSimulatorHelper)

    @patch('_helpers.backend_helpers.AwsDevice')
    def test_get_backend_helper_aria(self, mock_aws_device):
        """
        Test get_backend_helper("Aria") returns IonQAriaHelper

        Expected: Returns IonQAriaHelper instance
        """
        result = get_backend_helper("Aria")

        assert isinstance(result, IonQAriaHelper)

    def test_get_backend_helper_lucy_raises(self):
        """
        Test get_backend_helper("Lucy") raises ValueError

        Expected: Raises ValueError indicating device no longer supported
        """
        with pytest.raises(ValueError, match="Lucy.*no longer supported"):
            get_backend_helper("Lucy")

    def test_get_backend_helper_oqcdirect_raises(self):
        """
        Test get_backend_helper("OQCDirect") raises ValueError

        Expected: Raises ValueError indicating device no longer supported
        """
        with pytest.raises(ValueError, match="OQCDirect.*no longer supported"):
            get_backend_helper("OQCDirect")

    def test_get_backend_helper_harmony_raises(self):
        """
        Test get_backend_helper("Harmony") raises ValueError

        Expected: Raises ValueError indicating device no longer supported
        """
        with pytest.raises(ValueError, match="Harmony.*no longer supported"):
            get_backend_helper("Harmony")

    def test_get_backend_helper_unsupported_device(self):
        """
        Test get_backend_helper with unsupported device name

        Given: Unknown device name "FakeDevice"
        Expected: Raises ValueError with "Unsupported device name"
        """
        with pytest.raises(ValueError, match="Unsupported device name FakeDevice"):
            get_backend_helper("FakeDevice")

    def test_get_backend_helper_case_sensitive(self):
        """
        Test that get_backend_helper is case-sensitive

        Given: "SIMULATOR" instead of "simulator"
        Expected: Raises ValueError (case-sensitive matching)
        """
        with pytest.raises(ValueError, match="Unsupported device name"):
            get_backend_helper("SIMULATOR")


class TestBackendHelperIntegration:
    """Test integration scenarios across backend helpers"""

    @patch('_helpers.backend_helpers.LocalSimulator')
    @patch('_helpers.backend_helpers.NoisySimWrapper')
    @patch('_helpers.backend_helpers.NoiselessDensityMatrixSimWrapper')
    def test_all_simulators_return_zero_cost(self, mock_noiseless, mock_noisy, mock_ideal):
        """
        Test that all local simulators return (0, 0) costs

        Expected: simulator, noisy_sim, noisy_sim_with_shots, noiseless_sim all return (0, 0)
        """
        simulators = ["simulator", "noisy_sim", "noiseless_sim"]

        for sim_name in simulators:
            with patch('_helpers.backend_helpers.NoisySimWrapperWithShots'):
                helper = get_backend_helper(sim_name)
                costs = helper.get_costs()
                assert costs == (0, 0), f"{sim_name} should have zero costs"

    @patch('_helpers.backend_helpers.LocalSimulator')
    @patch('_helpers.backend_helpers.BraketLocalBackend')
    def test_simulators_use_braket_local_backend(self, mock_braket, mock_local_sim):
        """
        Test that simulators use BraketLocalBackend

        Expected: All simulator helpers return BraketLocalBackend from get_qiskit_backend
        """
        mock_backend = Mock()
        mock_braket.return_value = mock_backend

        helper = IdealSimulatorHelper()
        result = helper.get_qiskit_backend()

        assert result is mock_backend


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

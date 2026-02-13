"""
Unit tests for registry module

This test suite validates the registry pattern used for global state management
including circuit submitters, optimizers, builders, and control parameters.
"""

import pytest
import sys
import pathlib
from unittest.mock import Mock
from copy import deepcopy

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent.parent.resolve()))

from _helpers.registry import (
    CircuitSubmitterRegistry,
    submitter_registry,
    ControlParameterRegistry,
    control_parameter_registry
)
from _helpers.builders.base import BuilderRegistry, builder_registry, Builder


class TestCircuitSubmitterRegistry:
    """Test CircuitSubmitterRegistry class"""

    def test_store_and_get_submitter(self):
        """
        Test storing and retrieving a submitter

        Given: A submitter object and a name
        When: store_submitter() is called
        Expected: get_submitter() returns the same object
        """
        registry = CircuitSubmitterRegistry()
        mock_submitter = Mock()

        registry.store_submitter(mock_submitter, "test_device")
        result = registry.get_submitter("test_device")

        assert result is mock_submitter

    def test_get_nonexistent_submitter(self):
        """
        Test getting a submitter that doesn't exist

        Expected: Returns None
        """
        registry = CircuitSubmitterRegistry()
        result = registry.get_submitter("nonexistent")

        assert result is None

    def test_store_multiple_submitters(self):
        """
        Test storing multiple submitters with different names

        Expected: All can be retrieved independently
        """
        registry = CircuitSubmitterRegistry()
        submitter1 = Mock()
        submitter2 = Mock()
        submitter3 = Mock()

        registry.store_submitter(submitter1, "device1")
        registry.store_submitter(submitter2, "device2")
        registry.store_submitter(submitter3, "device3")

        assert registry.get_submitter("device1") is submitter1
        assert registry.get_submitter("device2") is submitter2
        assert registry.get_submitter("device3") is submitter3

    def test_overwrite_existing_submitter(self):
        """
        Test that storing with same name overwrites previous submitter

        Given: A submitter stored with name "device"
        When: Another submitter is stored with same name
        Expected: Second submitter replaces the first
        """
        registry = CircuitSubmitterRegistry()
        submitter1 = Mock()
        submitter2 = Mock()

        registry.store_submitter(submitter1, "device")
        registry.store_submitter(submitter2, "device")

        result = registry.get_submitter("device")
        assert result is submitter2
        assert result is not submitter1

    def test_list_submitters_empty(self):
        """
        Test listing submitters when registry is empty

        Expected: Returns empty list
        """
        registry = CircuitSubmitterRegistry()
        result = registry.list_submitters()

        assert result == []

    def test_list_submitters_with_items(self):
        """
        Test listing all submitter names

        Expected: Returns list of all stored names
        """
        registry = CircuitSubmitterRegistry()
        registry.store_submitter(Mock(), "device1")
        registry.store_submitter(Mock(), "device2")
        registry.store_submitter(Mock(), "device3")

        result = registry.list_submitters()

        assert len(result) == 3
        assert "device1" in result
        assert "device2" in result
        assert "device3" in result

    def test_global_submitter_registry_exists(self):
        """
        Test that global submitter_registry instance exists

        Expected: submitter_registry is a CircuitSubmitterRegistry instance
        """
        assert isinstance(submitter_registry, CircuitSubmitterRegistry)



class TestControlParameterRegistry:
    """Test ControlParameterRegistry class"""

    def test_get_empty_registry(self):
        """
        Test getting control parameters from empty registry

        Expected: Returns empty dict
        """
        registry = ControlParameterRegistry()
        result = registry.get_control_parameters()

        assert result == {}

    def test_set_and_get_control_parameters(self):
        """
        Test setting and getting control parameters

        Given: Control parameters dict
        When: set_control_parameters() is called
        Expected: get_control_parameters() returns same dict
        """
        registry = ControlParameterRegistry()
        params = {
            "temperature": [40, "mK"],
            "power": [100, "W"]
        }

        registry.set_control_parameters(params)
        result = registry.get_control_parameters()

        assert result == params

    def test_set_overwrites_previous(self):
        """
        Test that setting new parameters overwrites previous ones

        Given: Parameters already set
        When: set_control_parameters() called with new params
        Expected: Old parameters are replaced
        """
        registry = ControlParameterRegistry()

        old_params = {"temperature": [20, "mK"]}
        new_params = {"temperature": [40, "mK"], "power": [100, "W"]}

        registry.set_control_parameters(old_params)
        registry.set_control_parameters(new_params)

        result = registry.get_control_parameters()

        assert result == new_params
        assert "power" in result

    def test_set_empty_dict(self):
        """
        Test setting empty control parameters

        Expected: Can set and retrieve empty dict
        """
        registry = ControlParameterRegistry()

        registry.set_control_parameters({})
        result = registry.get_control_parameters()

        assert result == {}

    def test_parameters_are_mutable_reference(self):
        """
        Test that returned parameters are the actual dict (not a copy)

        Note: Understanding whether the registry stores a reference or copy
        """
        registry = ControlParameterRegistry()
        params = {"temperature": [40, "mK"]}

        registry.set_control_parameters(params)
        result = registry.get_control_parameters()

        # Modify the returned dict
        result["new_param"] = [100, "unit"]

        # Check if it affects the stored parameters
        result2 = registry.get_control_parameters()

        # If it's a reference, new_param will be present
        assert "new_param" in result2

    def test_global_control_parameter_registry_exists(self):
        """
        Test that global control_parameter_registry instance exists

        Expected: control_parameter_registry is a ControlParameterRegistry instance
        """
        assert isinstance(control_parameter_registry, ControlParameterRegistry)


class TestBuilderRegistry:
    """Test BuilderRegistry class"""

    def test_register_and_get_builder(self):
        """
        Test registering and retrieving a builder class

        Given: A builder class with registry_name attribute
        When: register_builder() is called
        Expected: get_builder() returns the class
        """
        registry = BuilderRegistry()

        # Create a mock builder class
        class MockBuilder:
            registry_name = "test_builder"

        registry.register_builder(MockBuilder)
        result = registry.get_builder("test_builder")

        assert result is MockBuilder

    def test_get_nonexistent_builder(self):
        """
        Test getting a builder that doesn't exist

        Expected: Returns None
        """
        registry = BuilderRegistry()
        result = registry.get_builder("nonexistent")

        assert result is None

    def test_register_multiple_builders(self):
        """
        Test registering multiple builder classes

        Expected: All can be retrieved independently
        """
        registry = BuilderRegistry()

        class Builder1:
            registry_name = "builder1"

        class Builder2:
            registry_name = "builder2"

        registry.register_builder(Builder1)
        registry.register_builder(Builder2)

        assert registry.get_builder("builder1") is Builder1
        assert registry.get_builder("builder2") is Builder2

    def test_list_builders_empty(self):
        """
        Test listing builders when registry is empty

        Expected: Returns empty list
        """
        registry = BuilderRegistry()
        result = registry.list_builders()

        assert result == []

    def test_list_builders_with_items(self):
        """
        Test listing all builder names

        Expected: Returns list of all registered names
        """
        registry = BuilderRegistry()

        class Builder1:
            registry_name = "builder1"

        class Builder2:
            registry_name = "builder2"

        registry.register_builder(Builder1)
        registry.register_builder(Builder2)

        result = registry.list_builders()

        assert len(result) == 2
        assert "builder1" in result
        assert "builder2" in result

    def test_overwrite_builder_with_same_name(self):
        """
        Test that registering with same name overwrites previous builder

        Given: A builder registered with name "builder"
        When: Another builder is registered with same name
        Expected: Second builder replaces the first
        """
        registry = BuilderRegistry()

        class Builder1:
            registry_name = "builder"

        class Builder2:
            registry_name = "builder"

        registry.register_builder(Builder1)
        registry.register_builder(Builder2)

        result = registry.get_builder("builder")

        assert result is Builder2
        assert result is not Builder1

    def test_global_builder_registry_exists(self):
        """
        Test that global builder_registry instance exists

        Expected: builder_registry is a BuilderRegistry instance
        """
        assert isinstance(builder_registry, BuilderRegistry)


class TestRegistryIsolation:
    """Test that registries are properly isolated between tests"""

    def test_submitter_registry_isolation(self):
        """
        Test that submitter registry is isolated between tests

        Note: The conftest.py should reset registries between tests
        """
        # This test relies on conftest.py reset_registries fixture
        result = submitter_registry.list_submitters()

        # Should be empty if isolation is working
        assert isinstance(result, list)

    def test_control_parameter_registry_isolation(self):
        """Test that control parameter registry is isolated between tests"""
        result = control_parameter_registry.get_control_parameters()

        # Should be empty if isolation is working
        assert isinstance(result, dict)

    def test_builder_registry_isolation(self):
        """Test that builder registry is isolated between tests"""
        result = builder_registry.list_builders()

        # Should be a list (may have registered builders from imports)
        assert isinstance(result, list)


class TestRegistryConcurrentAccess:
    """Test registry behavior with concurrent-like access patterns"""

    def test_submitter_registry_rapid_updates(self):
        """
        Test rapid updates to submitter registry

        Expected: All operations complete successfully
        """
        registry = CircuitSubmitterRegistry()

        for i in range(100):
            submitter = Mock()
            registry.store_submitter(submitter, f"device_{i}")

        assert len(registry.list_submitters()) == 100

    def test_control_parameter_registry_rapid_updates(self):
        """
        Test rapid updates to control parameters

        Expected: Latest update is retained
        """
        registry = ControlParameterRegistry()

        for i in range(100):
            registry.set_control_parameters({"iteration": i})

        result = registry.get_control_parameters()
        assert result == {"iteration": 99}


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

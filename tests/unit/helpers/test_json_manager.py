"""
Unit tests for JsonManager class

This test suite validates the JsonManager's ability to navigate and manipulate
JSON data structures using dot notation paths and array indexing.
"""

import pytest
import json
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent.parent.resolve()))

from _helpers.json_manager import JsonManager


class TestJsonManagerInitialization:
    """Test JsonManager initialization"""

    def test_init_with_valid_file(self, tmp_path):
        """
        Test initialization with a valid JSON file

        Expected: File is loaded and data is accessible
        """
        data = {"key": "value", "number": 42}
        json_file = tmp_path / "test.json"
        with open(json_file, 'w') as f:
            json.dump(data, f)

        jm = JsonManager(str(json_file))

        assert jm.filename == str(json_file)
        assert jm.file_data == data
        assert jm.file_data["key"] == "value"
        assert jm.file_data["number"] == 42

    def test_init_with_nested_structure(self, tmp_path):
        """Test initialization with nested JSON structure"""
        data = {
            "level1": {
                "level2": {
                    "value": "deep"
                }
            }
        }
        json_file = tmp_path / "nested.json"
        with open(json_file, 'w') as f:
            json.dump(data, f)

        jm = JsonManager(str(json_file))

        assert jm.file_data["level1"]["level2"]["value"] == "deep"

    def test_init_with_array(self, tmp_path):
        """Test initialization with arrays"""
        data = {
            "items": [1, 2, 3],
            "objects": [{"name": "a"}, {"name": "b"}]
        }
        json_file = tmp_path / "array.json"
        with open(json_file, 'w') as f:
            json.dump(data, f)

        jm = JsonManager(str(json_file))

        assert len(jm.file_data["items"]) == 3
        assert jm.file_data["objects"][0]["name"] == "a"

    def test_init_file_not_found(self):
        """Test that FileNotFoundError is raised for missing file"""
        with pytest.raises(FileNotFoundError):
            JsonManager("/nonexistent/file.json")

    def test_init_invalid_json(self, tmp_path):
        """Test that JSONDecodeError is raised for invalid JSON"""
        json_file = tmp_path / "invalid.json"
        with open(json_file, 'w') as f:
            f.write("{invalid json")

        with pytest.raises(json.JSONDecodeError):
            JsonManager(str(json_file))


class TestJsonManagerResolve:
    """Test the resolve() method for path navigation"""

    def test_resolve_simple_path(self, tmp_path):
        """
        Test resolving a simple one-level path

        Given: {"config": {"value": 100}}
        When: resolve("config")
        Expected: Returns {"value": 100}
        """
        data = {"config": {"value": 100}}
        json_file = tmp_path / "test.json"
        with open(json_file, 'w') as f:
            json.dump(data, f)

        jm = JsonManager(str(json_file))
        result = jm.resolve("config")

        assert result == {"value": 100}

    def test_resolve_nested_path(self, tmp_path):
        """
        Test resolving multi-level nested path

        Given: {"a": {"b": {"c": "value"}}}
        When: resolve("a.b.c")
        Expected: Returns "value"
        """
        data = {"a": {"b": {"c": "value"}}}
        json_file = tmp_path / "test.json"
        with open(json_file, 'w') as f:
            json.dump(data, f)

        jm = JsonManager(str(json_file))
        result = jm.resolve("a.b.c")

        assert result == "value"

    def test_resolve_array_index(self, tmp_path):
        """
        Test resolving path with array indexing

        Given: {"qubits": [{"T1": 50}, {"T1": 60}]}
        When: resolve("qubits.[0].T1")
        Expected: Returns 50
        """
        data = {"qubits": [{"T1": 50}, {"T1": 60}]}
        json_file = tmp_path / "test.json"
        with open(json_file, 'w') as f:
            json.dump(data, f)

        jm = JsonManager(str(json_file))
        result = jm.resolve("qubits.[0].T1")

        assert result == 50

    def test_resolve_trailing_dot(self, tmp_path):
        """
        Test that trailing dot in path is handled correctly

        Given: {"qubits": [...]}
        When: resolve("qubits.")
        Expected: Returns the qubits list (trailing dot removed)
        """
        data = {"qubits": [1, 2, 3]}
        json_file = tmp_path / "test.json"
        with open(json_file, 'w') as f:
            json.dump(data, f)

        jm = JsonManager(str(json_file))
        result = jm.resolve("qubits.")

        assert result == [1, 2, 3]

    def test_resolve_nonexistent_path_no_create(self, tmp_path):
        """
        Test that ValueError is raised for nonexistent path when create_path=False

        Expected: Raises ValueError with descriptive message
        """
        data = {"existing": "value"}
        json_file = tmp_path / "test.json"
        with open(json_file, 'w') as f:
            json.dump(data, f)

        jm = JsonManager(str(json_file))

        with pytest.raises(ValueError, match="specified path does not exist"):
            jm.resolve("nonexistent.path", create_path=False)

    def test_resolve_nonexistent_path_with_create(self, tmp_path):
        """
        Test that path is created when create_path=True

        Given: {"existing": "value"}
        When: resolve("new.path", create_path=True)
        Expected: Creates nested structure and returns {}
        """
        data = {"existing": "value"}
        json_file = tmp_path / "test.json"
        with open(json_file, 'w') as f:
            json.dump(data, f)

        jm = JsonManager(str(json_file))
        result = jm.resolve("new.path", create_path=True)

        assert result == {}
        assert "new" in jm.file_data
        assert "path" in jm.file_data["new"]


class TestJsonManagerUpdate:
    """Test the update() method for modifying values"""

    def test_update_simple_value(self, tmp_path):
        """
        Test updating a simple value

        Given: {"key": "old"}
        When: update("key", "new")
        Expected: {"key": "new"}
        """
        data = {"key": "old"}
        json_file = tmp_path / "test.json"
        with open(json_file, 'w') as f:
            json.dump(data, f)

        jm = JsonManager(str(json_file))
        jm.update("key", "new")

        assert jm.file_data["key"] == "new"

    def test_update_nested_value(self, tmp_path):
        """
        Test updating a nested value

        Given: {"a": {"b": {"c": 1}}}
        When: update("a.b.c", 42)
        Expected: {"a": {"b": {"c": 42}}}
        """
        data = {"a": {"b": {"c": 1}}}
        json_file = tmp_path / "test.json"
        with open(json_file, 'w') as f:
            json.dump(data, f)

        jm = JsonManager(str(json_file))
        jm.update("a.b.c", 42)

        assert jm.file_data["a"]["b"]["c"] == 42

    def test_update_array_element(self, tmp_path):
        """
        Test updating an array element

        Given: {"items": [10, 20, 30]}
        When: update("items.[1]", 99)
        Expected: {"items": [10, 99, 30]}
        """
        data = {"items": [10, 20, 30]}
        json_file = tmp_path / "test.json"
        with open(json_file, 'w') as f:
            json.dump(data, f)

        jm = JsonManager(str(json_file))
        jm.update("items.[1]", 99)

        assert jm.file_data["items"][1] == 99
        assert jm.file_data["items"] == [10, 99, 30]

    def test_update_create_path_true(self, tmp_path):
        """
        Test creating new path when create_path=True

        Given: {"existing": "value"}
        When: update("new.nested.path", "value", create_path=True)
        Expected: Path is created with value
        """
        data = {"existing": "value"}
        json_file = tmp_path / "test.json"
        with open(json_file, 'w') as f:
            json.dump(data, f)

        jm = JsonManager(str(json_file))
        jm.update("new.nested.path", "created", create_path=True)

        assert jm.file_data["new"]["nested"]["path"] == "created"

    def test_update_create_path_false_raises(self, tmp_path):
        """
        Test that ValueError is raised when path doesn't exist and create_path=False

        Expected: Raises ValueError
        """
        data = {"existing": "value"}
        json_file = tmp_path / "test.json"
        with open(json_file, 'w') as f:
            json.dump(data, f)

        jm = JsonManager(str(json_file))

        with pytest.raises(ValueError, match="specified path does not exist"):
            jm.update("nonexistent.path", "value", create_path=False)


class TestJsonManagerWrite:
    """Test the write() method for persisting changes"""

    def test_write_saves_to_file(self, tmp_path):
        """
        Test that write() persists changes to disk

        Given: Modified JsonManager data
        When: write() is called
        Expected: Changes are saved to file
        """
        data = {"key": "original"}
        json_file = tmp_path / "test.json"
        with open(json_file, 'w') as f:
            json.dump(data, f)

        jm = JsonManager(str(json_file))
        jm.update("key", "modified")
        jm.write()

        # Read file directly to verify
        with open(json_file, 'r') as f:
            saved_data = json.load(f)

        assert saved_data["key"] == "modified"

    def test_write_preserves_structure(self, tmp_path):
        """Test that write() preserves complex structure"""
        data = {
            "nested": {"value": 1},
            "array": [1, 2, 3],
            "mixed": [{"id": 1}, {"id": 2}]
        }
        json_file = tmp_path / "test.json"
        with open(json_file, 'w') as f:
            json.dump(data, f)

        jm = JsonManager(str(json_file))
        jm.update("nested.value", 42)
        jm.write()

        # Reload and verify
        with open(json_file, 'r') as f:
            saved_data = json.load(f)

        assert saved_data["nested"]["value"] == 42
        assert saved_data["array"] == [1, 2, 3]
        assert len(saved_data["mixed"]) == 2


class TestJsonManagerFindValue:
    """Test find_value() method for searching in lists"""

    def test_find_value_in_list(self, tmp_path):
        """
        Test finding a value in a list of dictionaries

        Given: List with [{"name": "T1", "value": 50}, {"name": "T2", "value": 70}]
        When: find_value("T1", "value", path)
        Expected: Returns 50
        """
        data = {
            "qubits": [
                [
                    {"name": "T1", "value": 50},
                    {"name": "T2", "value": 70}
                ]
            ]
        }
        json_file = tmp_path / "test.json"
        with open(json_file, 'w') as f:
            json.dump(data, f)

        jm = JsonManager(str(json_file))
        result = jm.find_value("T1", "value", "qubits.[0].")

        assert result == 50

    def test_find_value_not_found(self, tmp_path):
        """
        Test that None is returned when value not found

        Expected: Returns None
        """
        data = {
            "qubits": [
                [
                    {"name": "T1", "value": 50}
                ]
            ]
        }
        json_file = tmp_path / "test.json"
        with open(json_file, 'w') as f:
            json.dump(data, f)

        jm = JsonManager(str(json_file))
        result = jm.find_value("nonexistent", "value", "qubits.[0].")

        assert result is None

    def test_find_value_returns_first_match(self, tmp_path):
        """
        Test that find_value returns the first matching item

        Given: Multiple items with same name
        Expected: Returns value from first match
        """
        data = {
            "props": [
                {"name": "param", "value": 100},
                {"name": "param", "value": 200}
            ]
        }
        json_file = tmp_path / "test.json"
        with open(json_file, 'w') as f:
            json.dump(data, f)

        jm = JsonManager(str(json_file))
        result = jm.find_value("param", "value", "props.")

        assert result == 100


class TestJsonManagerFindPath:
    """Test find_path() method for finding property paths"""

    def test_find_path_in_list(self, tmp_path):
        """
        Test finding a path to a property in a list

        Given: List with items having "name" property
        When: find_path("T1", "value", path)
        Expected: Returns full path to T1's value
        """
        data = {
            "qubits": [
                [
                    {"name": "T1", "value": 50},
                    {"name": "T2", "value": 70}
                ]
            ]
        }
        json_file = tmp_path / "test.json"
        with open(json_file, 'w') as f:
            json.dump(data, f)

        jm = JsonManager(str(json_file))
        result = jm.find_path("T1", "value", "qubits.[0].")

        assert "T1" in result or "[0]" in result
        assert result is not None

    def test_find_path_not_found(self, tmp_path):
        """
        Test that None is returned when path not found

        Expected: Returns None
        """
        data = {
            "props": [
                {"name": "existing", "value": 1}
            ]
        }
        json_file = tmp_path / "test.json"
        with open(json_file, 'w') as f:
            json.dump(data, f)

        jm = JsonManager(str(json_file))
        result = jm.find_path("nonexistent", "value", "props.")

        assert result is None


class TestJsonManagerGetPaths:
    """Test get_qubit_paths() and get_gate_paths() methods"""

    def test_get_qubit_paths(self, tmp_path):
        """
        Test getting all qubit paths

        Given: {"qubits": [{...}, {...}, {...}]}
        Expected: Returns ["qubits.[0].", "qubits.[1].", "qubits.[2]."]
        """
        data = {
            "qubits": [
                {"T1": 50},
                {"T1": 60},
                {"T1": 70}
            ]
        }
        json_file = tmp_path / "test.json"
        with open(json_file, 'w') as f:
            json.dump(data, f)

        jm = JsonManager(str(json_file))
        paths = jm.get_qubit_paths()

        assert len(paths) == 3
        assert paths[0] == "qubits.[0]."
        assert paths[1] == "qubits.[1]."
        assert paths[2] == "qubits.[2]."

    def test_get_gate_paths(self, tmp_path):
        """
        Test getting all gate paths

        Given: {"gates": [{...}, {...}]}
        Expected: Returns ["gates.[0].", "gates.[1]."]
        """
        data = {
            "gates": [
                {"name": "cx"},
                {"name": "sx"}
            ]
        }
        json_file = tmp_path / "test.json"
        with open(json_file, 'w') as f:
            json.dump(data, f)

        jm = JsonManager(str(json_file))
        paths = jm.get_gate_paths()

        assert len(paths) == 2
        assert paths[0] == "gates.[0]."
        assert paths[1] == "gates.[1]."

    def test_get_qubit_paths_empty_list(self, tmp_path):
        """Test get_qubit_paths with empty qubit list"""
        data = {"qubits": []}
        json_file = tmp_path / "test.json"
        with open(json_file, 'w') as f:
            json.dump(data, f)

        jm = JsonManager(str(json_file))
        paths = jm.get_qubit_paths()

        assert len(paths) == 0
        assert paths == []


class TestJsonManagerUnitConversion:
    """Test find_value_with_units() and update_with_units() methods"""

    def test_find_value_with_units_microseconds(self, tmp_path):
        """
        Test finding value with unit conversion

        Given: {"name": "T1", "value": 50, "unit": "us"}
        When: find_value_with_units("T1", path)
        Expected: Returns 50 * 1e-6 = 50e-6
        """
        data = {
            "props": [
                {"name": "T1", "value": 50, "unit": "us"}
            ]
        }
        json_file = tmp_path / "test.json"
        with open(json_file, 'w') as f:
            json.dump(data, f)

        jm = JsonManager(str(json_file))
        result = jm.find_value_with_units("T1", "props.")

        expected = 50 * 1e-6
        assert abs(result - expected) < 1e-10

    def test_find_value_with_units_gigahertz(self, tmp_path):
        """
        Test unit conversion for GHz

        Given: {"name": "frequency", "value": 5.0, "unit": "GHz"}
        Expected: Returns 5.0 * 1e9
        """
        data = {
            "props": [
                {"name": "frequency", "value": 5.0, "unit": "GHz"}
            ]
        }
        json_file = tmp_path / "test.json"
        with open(json_file, 'w') as f:
            json.dump(data, f)

        jm = JsonManager(str(json_file))
        result = jm.find_value_with_units("frequency", "props.")

        assert result == 5.0e9

    def test_find_value_with_units_not_found(self, tmp_path):
        """Test that None is returned when property not found"""
        data = {"props": []}
        json_file = tmp_path / "test.json"
        with open(json_file, 'w') as f:
            json.dump(data, f)

        jm = JsonManager(str(json_file))
        result = jm.find_value_with_units("missing", "props.")

        assert result is None

    def test_update_with_units_converts_value(self, tmp_path):
        """
        Test that update_with_units converts value correctly

        Given: T1 property (unit: "us" from PROPERTY_UNITS)
        When: update_with_units("T1", 50e-6, path)
        Expected: Stores 50 (divided by 1e-6) and unit "us"
        """
        data = {
            "qubits": [
                [
                    {"name": "T1", "value": 0, "unit": ""}
                ]
            ]
        }
        json_file = tmp_path / "test.json"
        with open(json_file, 'w') as f:
            json.dump(data, f)

        jm = JsonManager(str(json_file))
        jm.update_with_units("T1", 50e-6, "qubits.[0].")

        # Should store 50 (value / unit_multiplier)
        stored_value = jm.find_value("T1", "value", "qubits.[0].")
        assert abs(stored_value - 50) < 1e-6

        # Unit should be set to "us"
        stored_unit = jm.find_value("T1", "unit", "qubits.[0].")
        assert stored_unit == "us"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

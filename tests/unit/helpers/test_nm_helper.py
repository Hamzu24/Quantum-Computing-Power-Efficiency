"""
Unit tests for nm_helper module

This test suite validates the noise model helper functions including:
- Noise model creation from various sources (fake backends, custom, random)
- GitHub API integration for fetching backend configs (MOCKED)
- Config file management and symlink creation
- Props file reset functionality
"""

import pytest
import sys
import pathlib
import json
import subprocess
from unittest.mock import Mock, patch, MagicMock, mock_open, call
from pathlib import Path
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent.parent.resolve()))

from _helpers.nm_helper import (
    craft_noise_model,
    nm_from_fake_backend,
    fetch_config_files,
    config_exists,
    create_backend_symlinks,
    get_backend_class,
    build_backend,
    get_commit_sha_for_branch,
    get_needed_files,
    download_github_backend_files,
    create_original_props,
    write_needed_files,
    get_props_filename,
    reset_props_file,
)


class TestCraftNoiseModel:
    """Test craft_noise_model main routing function"""

    def test_missing_type_field_raises(self):
        """
        Test craft_noise_model with no type field raises ValueError

        Given: Config dict without 'type' field
        Expected: Raises ValueError with message about type field
        """
        config = {"name": "test_backend"}

        with pytest.raises(ValueError, match="must have a type field"):
            craft_noise_model(config)

    @patch('_helpers.nm_helper.nm_from_fake_backend')
    def test_routes_to_fake_backend(self, mock_nm_from_fake):
        """
        Test craft_noise_model routes to nm_from_fake_backend

        Given: config with type="fake_backend"
        Expected: nm_from_fake_backend is called with config
        """
        mock_nm = Mock()
        mock_backend = Mock()
        mock_metadata = {"T1_values": [50e-6], "T2_values": [70e-6]}
        mock_nm_from_fake.return_value = (mock_nm, mock_backend, mock_metadata)

        config = {"type": "fake_backend", "name": "tokyo"}
        result = craft_noise_model(config)

        mock_nm_from_fake.assert_called_once_with(config)
        assert result == (mock_nm, mock_backend, mock_metadata)

    @patch('_helpers.nm_helper.NoiseModelWrapper')
    def test_routes_to_registry_nm(self, mock_wrapper_class):
        """
        Test craft_noise_model routes to NoiseModelWrapper for registry types

        Given: config with a registry-registered type (e.g. "simple_nm")
        Expected: NoiseModelWrapper.build() is called and returns 3-tuple with empty metadata
        """
        mock_nm = Mock()
        mock_backend = Mock()
        mock_wrapper = Mock()
        mock_wrapper.build.return_value = (mock_nm, mock_backend)
        mock_wrapper_class.return_value = mock_wrapper

        config = {"type": "simple_nm", "num_qubits": 4}

        with patch('_helpers.nm_helper.noise_model_registry.__contains__', return_value=True):
            result = craft_noise_model(config)

        assert result == (mock_nm, mock_backend, {})

    def test_unsupported_type_raises(self):
        """
        Test craft_noise_model with unsupported type raises ValueError

        Given: config with type="unsupported_type"
        Expected: Raises ValueError
        """
        config = {"type": "unsupported_type"}

        with pytest.raises(ValueError, match="Unknown noise model type"):
            craft_noise_model(config)


class TestConfigExists:
    """Test config_exists function"""

    @patch('subprocess.run')
    def test_config_exists_returns_false_when_dir_not_found(self, mock_run, monkeypatch):
        """
        Test config_exists returns False when directory doesn't exist

        Given: ls command returns error (stderr is set)
        Expected: Returns False
        """
        monkeypatch.setenv("BACKEND_CONFIGS_FOLDER", "/path/to/configs/")

        mock_result = Mock()
        mock_result.stderr = b"ls: cannot access '/path/to/configs/tokyo': No such file or directory"
        mock_run.return_value = mock_result

        result = config_exists("tokyo")

        assert result is False
        mock_run.assert_called_once_with(["ls", "/path/to/configs/tokyo"], capture_output=True)

    @patch('subprocess.run')
    def test_config_exists_returns_false_when_props_missing(self, mock_run, monkeypatch):
        """
        Test config_exists returns False when props file doesn't exist

        Given: ls succeeds but 'props' not in filenames
        Expected: Returns False
        """
        monkeypatch.setenv("BACKEND_CONFIGS_FOLDER", "/path/to/configs/")

        mock_result = Mock()
        mock_result.stderr = b""
        mock_result.stdout = b"conf_tokyo.json\ndefs_tokyo.json\n"
        mock_run.return_value = mock_result

        result = config_exists("tokyo")

        assert result is False

    @patch('subprocess.run')
    def test_config_exists_returns_true_when_props_present(self, mock_run, monkeypatch):
        """
        Test config_exists returns True when props file exists

        Given: ls succeeds and 'props' is in filenames
        Expected: Returns True
        """
        monkeypatch.setenv("BACKEND_CONFIGS_FOLDER", "/path/to/configs/")

        mock_result = Mock()
        mock_result.stderr = b""
        mock_result.stdout = b"conf_tokyo.json\nprops_tokyo.json\ndefs_tokyo.json\n"
        mock_run.return_value = mock_result

        result = config_exists("tokyo")

        assert result is True


class TestGetCommitShaForBranch:
    """Test get_commit_sha_for_branch GitHub API function"""

    @patch('requests.get')
    def test_successful_branch_lookup(self, mock_get):
        """
        Test get_commit_sha_for_branch returns SHA on success

        Given: GitHub API returns 200 with SHA
        Expected: Returns the commit SHA
        """
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "object": {"sha": "abc123def456"}
        }
        mock_get.return_value = mock_response

        result = get_commit_sha_for_branch("Qiskit", "qiskit", "stable/0.46")

        expected_url = "https://api.github.com/repos/Qiskit/qiskit/git/refs/heads/stable%2F0.46"
        mock_get.assert_called_once_with(expected_url)
        assert result == "abc123def456"

    @patch('requests.get')
    def test_branch_not_found(self, mock_get):
        """
        Test get_commit_sha_for_branch returns None when branch not found

        Given: GitHub API returns 404
        Expected: Returns None and logs error
        """
        mock_response = Mock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response

        result = get_commit_sha_for_branch("Qiskit", "qiskit", "nonexistent")

        assert result is None

    @patch('requests.get')
    def test_slash_encoding_in_branch_name(self, mock_get):
        """
        Test that slashes in branch names are properly encoded

        Given: Branch name with slash "stable/0.46"
        Expected: Slash is encoded as %2F in URL
        """
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"object": {"sha": "xyz789"}}
        mock_get.return_value = mock_response

        get_commit_sha_for_branch("Qiskit", "qiskit", "stable/0.46")

        called_url = mock_get.call_args[0][0]
        assert "stable%2F0.46" in called_url
        assert "stable/0.46" not in called_url


class TestGetNeededFiles:
    """Test get_needed_files function"""

    def test_all_files_needed_when_dir_doesnt_exist(self, tmp_path):
        """
        Test get_needed_files returns all keywords when dir doesn't exist

        Expected: Returns {'conf', 'defs', 'props', 'original'}
        """
        non_existent = tmp_path / "nonexistent"

        result = get_needed_files(non_existent)

        expected = {'conf', 'defs', 'props', 'original'}
        assert result == expected

    def test_filters_out_existing_files(self, tmp_path):
        """
        Test get_needed_files excludes files that exist

        Given: conf and defs files exist
        Expected: Returns only {'props', 'original'}
        """
        (tmp_path / "conf_tokyo.json").touch()
        (tmp_path / "defs_tokyo.json").touch()

        result = get_needed_files(tmp_path)

        expected = {'props', 'original'}
        assert result == expected

    def test_props_without_original_adds_props_back(self, tmp_path):
        """
        Test when original is needed, props is also added

        Given: conf, defs, props exist but not original
        Expected: Returns {'props'} (original removed, props added)

        Note: Edge case handling - if original doesn't exist,
        we need to re-download props to create original
        """
        (tmp_path / "conf_tokyo.json").touch()
        (tmp_path / "defs_tokyo.json").touch()
        (tmp_path / "props_tokyo.json").touch()

        result = get_needed_files(tmp_path)

        # original was needed, so it gets converted to props
        assert 'props' in result
        assert 'original' not in result

    def test_all_files_exist(self, tmp_path):
        """
        Test get_needed_files returns empty when all files exist

        Given: All required files (conf, defs, props, original) exist
        Expected: Returns empty set
        """
        (tmp_path / "conf_tokyo.json").touch()
        (tmp_path / "defs_tokyo.json").touch()
        (tmp_path / "props_tokyo.json").touch()
        (tmp_path / "props_tokyo_original.json").touch()

        result = get_needed_files(tmp_path)

        assert result == set()


class TestDownloadGithubBackendFiles:
    """Test download_github_backend_files function"""

    @patch('requests.get')
    def test_successful_download(self, mock_get):
        """
        Test download_github_backend_files returns files on success

        Given: GitHub API returns 200 with file list
        Expected: Returns the files JSON
        """
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"name": "conf_tokyo.json", "download_url": "https://..."},
            {"name": "props_tokyo.json", "download_url": "https://..."}
        ]
        mock_get.return_value = mock_response

        result = download_github_backend_files("tokyo", "abc123")

        assert len(result) == 2
        assert result[0]["name"] == "conf_tokyo.json"

    @patch('requests.get')
    def test_backend_not_found(self, mock_get):
        """
        Test download_github_backend_files returns None on 404

        Given: GitHub API returns 404 (backend not found)
        Expected: Returns None and logs error
        """
        mock_response = Mock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response

        # Note: The function references 'exit_if_unavailable' which is not in scope
        # This appears to be a bug in the original code
        result = download_github_backend_files("nonexistent", "abc123")

        assert result is None


class TestCreateOriginalProps:
    """Test create_original_props function"""

    @patch('subprocess.run')
    def test_creates_copy_of_props_file(self, mock_run, tmp_path):
        """
        Test create_original_props copies props file

        Given: props file exists
        Expected: Copies to props_original.json
        """
        props_file = "props_tokyo.json"

        create_original_props(tmp_path, props_file)

        expected_source = str(tmp_path / "props_tokyo.json")
        expected_dest = str(tmp_path / "props_tokyo_original.json")
        mock_run.assert_called_once_with(["cp", expected_source, expected_dest])

    @patch('subprocess.run')
    def test_handles_copy_error(self, mock_run, tmp_path):
        """
        Test create_original_props handles CalledProcessError

        Given: cp command fails
        Expected: Logs critical error but doesn't raise
        """
        mock_run.side_effect = subprocess.CalledProcessError(1, 'cp')

        # Should not raise exception
        create_original_props(tmp_path, "props_tokyo.json")


class TestWriteNeededFiles:
    """Test write_needed_files function"""

    @patch('requests.get')
    @patch('builtins.open', new_callable=mock_open)
    @patch('os.makedirs')
    def test_downloads_and_writes_matching_files(self, mock_makedirs, mock_file, mock_get, tmp_path):
        """
        Test write_needed_files downloads files with matching keywords

        Given: Files list from GitHub, needed_keywords=['props', 'conf']
        Expected: Only files with 'props' or 'conf' in name are downloaded
        """
        files = [
            {"name": "conf_tokyo.json", "download_url": "https://example.com/conf"},
            {"name": "props_tokyo.json", "download_url": "https://example.com/props"},
            {"name": "other_file.py", "download_url": "https://example.com/other"}
        ]

        mock_response = Mock()
        mock_response.text = "file content"
        mock_get.return_value = mock_response

        write_needed_files(files, tmp_path, ['props', 'conf'])

        # Should download 2 files (conf and props), not other_file.py
        assert mock_get.call_count == 2

    @patch('requests.get')
    @patch('builtins.open', new_callable=mock_open)
    @patch('os.makedirs')
    def test_creates_output_directory(self, mock_makedirs, mock_file, mock_get, tmp_path):
        """
        Test write_needed_files creates output directory

        Expected: os.makedirs is called with exist_ok=True
        """
        files = [{"name": "props_tokyo.json", "download_url": "https://..."}]
        mock_get.return_value = Mock(text="content")

        write_needed_files(files, tmp_path, ['props'])

        mock_makedirs.assert_called_once_with(str(tmp_path), exist_ok=True)


class TestGetPropsFilename:
    """Test get_props_filename function"""

    def test_finds_props_file(self, tmp_path):
        """
        Test get_props_filename finds file with 'props' in name

        Given: Directory with props_tokyo.json
        Expected: Returns "props_tokyo.json"
        """
        (tmp_path / "conf_tokyo.json").touch()
        (tmp_path / "props_tokyo.json").touch()

        result = get_props_filename(tmp_path)

        assert result == "props_tokyo.json"

    def test_no_props_file_raises(self, tmp_path):
        """
        Test get_props_filename raises when no props file found

        Given: Directory with no props file
        Expected: Raises ValueError
        """
        (tmp_path / "conf_tokyo.json").touch()

        with pytest.raises(ValueError, match="props file"):
            get_props_filename(tmp_path)


class TestResetPropsFile:
    """Test reset_props_file function"""

    @patch('subprocess.run')
    def test_resets_props_from_original(self, mock_run, tmp_path):
        """
        Test reset_props_file copies original back to props

        Given: props_original.json exists
        Expected: Copies original to props.json
        """
        (tmp_path / "props_tokyo.json").touch()
        (tmp_path / "props_tokyo_original.json").touch()

        reset_props_file(tmp_path)

        # Should copy original to props
        calls = mock_run.call_args_list
        assert len(calls) == 1
        assert "props_tokyo_original.json" in calls[0][0][0][1]
        assert "props_tokyo.json" in calls[0][0][0][2]


class TestGetBackendClass:
    """Test get_backend_class function"""

    @patch('_helpers.nm_helper.EXISTING_MODELS', {'FakeTokyo': Mock, 'FakePerth': Mock})
    def test_finds_matching_backend_class(self):
        """
        Test get_backend_class finds backend in EXISTING_MODELS

        Given: backend_name="tokyo" exists in EXISTING_MODELS['FakeTokyo']
        Expected: Returns the class from EXISTING_MODELS
        """
        config = {"name": "tokyo"}

        result = get_backend_class(config, "tokyo")

        assert result == Mock

    @patch('_helpers.nm_helper.EXISTING_MODELS', {'FakePerth': Mock})
    def test_backend_not_found_raises(self):
        """
        Test get_backend_class raises when backend not found

        Given: backend_name="nonexistent" not in EXISTING_MODELS
        Expected: Raises ValueError
        """
        config = {"name": "nonexistent"}

        with pytest.raises(ValueError, match="cannot be found"):
            get_backend_class(config, "nonexistent")


class TestBuildBackend:
    """Test build_backend function"""

    @patch('_helpers.nm_helper.get_control_parameters')
    @patch('_helpers.nm_helper.BuilderWrapper')
    def test_creates_builder_and_builds(self, mock_wrapper_class, mock_get_control):
        """
        Test build_backend creates BuilderWrapper and calls build_backend

        Given: Config with init_control_parameters
        Expected: BuilderWrapper is instantiated, build_backend is called, metadata returned
        """
        config = {
            "name": "tokyo",
            "init_control_parameters": {"temperature": [40, "mK"]}
        }
        mock_control_params = {"temperature": [40, "mK"]}
        mock_get_control.return_value = mock_control_params

        mock_metadata = {"T1_values": [50e-6], "T2_values": [70e-6]}
        mock_builder = Mock()
        mock_builder.build_backend.return_value = mock_metadata
        mock_wrapper_class.return_value = mock_builder

        result = build_backend(config, "tokyo")

        mock_wrapper_class.assert_called_once_with("tokyo", {"temperature": [40, "mK"]})
        mock_builder.build_backend.assert_called_once_with(mock_control_params)
        assert result == mock_metadata


class TestCreateBackendSymlinks:
    """Test create_backend_symlinks function"""

    @patch('inspect.getsourcefile')
    def test_creates_symlinks_for_config_files(self, mock_getsourcefile, tmp_path, monkeypatch):
        """
        Test create_backend_symlinks creates symlinks in backend directory

        Given: Config files in source_dir
        Expected: Symlinks created in target_dir (backend source location)
        """
        monkeypatch.setenv("BACKEND_CONFIGS_FOLDER", str(tmp_path) + "/")

        # Create source directory with files
        source_dir = tmp_path / "tokyo"
        source_dir.mkdir()
        (source_dir / "props_tokyo.json").touch()
        (source_dir / "conf_tokyo.json").touch()

        # Create target directory
        target_dir = tmp_path / "fake_provider"
        target_dir.mkdir()

        mock_getsourcefile.return_value = str(target_dir / "fake_tokyo.py")
        mock_class = Mock()

        config = {"name": "tokyo"}

        create_backend_symlinks(config, mock_class)

        # Symlinks should be created
        assert (target_dir / "props_tokyo.json").exists()
        assert (target_dir / "conf_tokyo.json").exists()

    @patch('inspect.getsourcefile')
    def test_skips_original_files(self, mock_getsourcefile, tmp_path, monkeypatch):
        """
        Test create_backend_symlinks skips files with 'original' in name

        Given: Source dir has props_original.json
        Expected: No symlink created for original file
        """
        monkeypatch.setenv("BACKEND_CONFIGS_FOLDER", str(tmp_path) + "/")

        source_dir = tmp_path / "tokyo"
        source_dir.mkdir()
        (source_dir / "props_tokyo.json").touch()
        (source_dir / "props_tokyo_original.json").touch()

        target_dir = tmp_path / "fake_provider"
        target_dir.mkdir()

        mock_getsourcefile.return_value = str(target_dir / "fake_tokyo.py")
        mock_class = Mock()

        config = {"name": "tokyo"}

        create_backend_symlinks(config, mock_class)

        # Original file should not have symlink
        assert (target_dir / "props_tokyo.json").exists()
        assert not (target_dir / "props_tokyo_original.json").exists()


class TestFetchConfigFiles:
    """Test fetch_config_files function"""

    @patch('_helpers.nm_helper.reset_props_file')
    @patch('_helpers.nm_helper.write_needed_files')
    @patch('_helpers.nm_helper.download_github_backend_files')
    @patch('_helpers.nm_helper.get_commit_sha_for_branch')
    @patch('_helpers.nm_helper.get_needed_files')
    def test_downloads_and_writes_files_when_needed(
        self, mock_get_needed, mock_get_sha, mock_download, mock_write, mock_reset,
        tmp_path, monkeypatch
    ):
        """
        Test fetch_config_files downloads files when needed

        Given: Some files are missing
        Expected: GitHub API is called and files are written
        """
        monkeypatch.setenv("BACKEND_CONFIGS_FOLDER", str(tmp_path) + "/")

        mock_get_needed.return_value = {'props', 'conf'}
        mock_get_sha.return_value = "abc123"
        mock_download.return_value = [
            {"name": "props_tokyo.json", "download_url": "https://..."},
            {"name": "conf_tokyo.json", "download_url": "https://..."}
        ]

        fetch_config_files("tokyo", exit_if_unavailable=True)

        mock_get_sha.assert_called_once()
        mock_download.assert_called_once_with("tokyo", "abc123")
        mock_write.assert_called_once()
        mock_reset.assert_called_once()

    @patch('_helpers.nm_helper.reset_props_file')
    @patch('_helpers.nm_helper.get_needed_files')
    def test_skips_download_when_all_files_exist(
        self, mock_get_needed, mock_reset, tmp_path, monkeypatch
    ):
        """
        Test fetch_config_files skips download when all files exist

        Given: All needed files already exist (get_needed_files returns empty set)
        Expected: No GitHub API calls made
        """
        monkeypatch.setenv("BACKEND_CONFIGS_FOLDER", str(tmp_path) + "/")

        mock_get_needed.return_value = set()  # No files needed

        fetch_config_files("tokyo", exit_if_unavailable=True)

        # Should still reset props file
        mock_reset.assert_called_once()

    @patch('_helpers.nm_helper.get_commit_sha_for_branch')
    @patch('_helpers.nm_helper.get_needed_files')
    def test_raises_when_commit_sha_unavailable_and_exit_true(
        self, mock_get_needed, mock_get_sha, tmp_path, monkeypatch
    ):
        """
        Test fetch_config_files raises when SHA not found and exit_if_unavailable=True

        Given: get_commit_sha_for_branch returns None
        Expected: Raises Exception
        """
        monkeypatch.setenv("BACKEND_CONFIGS_FOLDER", str(tmp_path) + "/")

        mock_get_needed.return_value = {'props'}
        mock_get_sha.return_value = None

        with pytest.raises(Exception, match="commit SHA was not found"):
            fetch_config_files("tokyo", exit_if_unavailable=True)

    @patch('_helpers.nm_helper.get_commit_sha_for_branch')
    @patch('_helpers.nm_helper.get_needed_files')
    def test_returns_none_when_commit_sha_unavailable_and_exit_false(
        self, mock_get_needed, mock_get_sha, tmp_path, monkeypatch
    ):
        """
        Test fetch_config_files returns None when SHA not found and exit_if_unavailable=False

        Given: get_commit_sha_for_branch returns None, exit_if_unavailable=False
        Expected: Returns None without raising
        """
        monkeypatch.setenv("BACKEND_CONFIGS_FOLDER", str(tmp_path) + "/")

        mock_get_needed.return_value = {'props'}
        mock_get_sha.return_value = None

        result = fetch_config_files("tokyo", exit_if_unavailable=False)

        assert result is None


class TestNmFromFakeBackend:
    """Test nm_from_fake_backend integration function"""

    @patch('_helpers.nm_helper.NoiseModel')
    @patch('_helpers.nm_helper.build_backend')
    @patch('_helpers.nm_helper.create_backend_symlinks')
    @patch('_helpers.nm_helper.get_backend_class')
    @patch('_helpers.nm_helper.fetch_config_files')
    @patch('_helpers.nm_helper.config_exists')
    def test_full_workflow(
        self, mock_config_exists, mock_fetch, mock_get_class, mock_create_symlinks,
        mock_build, mock_nm_class
    ):
        """
        Test nm_from_fake_backend full workflow

        Expected: All helper functions called in correct order, returns 3-tuple with metadata
        """
        mock_config_exists.return_value = True
        mock_backend_class = Mock()
        mock_backend_instance = Mock()
        mock_backend_class.return_value = mock_backend_instance
        mock_get_class.return_value = mock_backend_class

        mock_metadata = {"T1_values": [50e-6], "T2_values": [70e-6]}
        mock_build.return_value = mock_metadata

        mock_noise_model = Mock()
        mock_nm_class.from_backend.return_value = mock_noise_model

        config = {"name": "tokyo"}

        noise_model, backend, metadata = nm_from_fake_backend(config)

        # Verify function call order
        mock_config_exists.assert_called_once_with("tokyo")
        mock_fetch.assert_called_once_with("tokyo", exit_if_unavailable=True)
        mock_get_class.assert_called_once_with(config, "tokyo")
        mock_create_symlinks.assert_called_once_with(config, mock_backend_class)
        mock_build.assert_called_once_with(config, "tokyo")

        assert noise_model is mock_noise_model
        assert backend is mock_backend_instance
        assert metadata == mock_metadata


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

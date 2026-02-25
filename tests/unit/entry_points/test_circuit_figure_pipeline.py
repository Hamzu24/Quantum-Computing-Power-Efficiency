"""
Unit tests for circuit figure pipeline

Tests that circuit_figure is correctly threaded through:
  run_metric → optimise / depth_sweep → experiment_runner
"""

import pytest
import sys
import pathlib
from unittest.mock import Mock, patch, MagicMock, PropertyMock
from pathlib import Path

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent.parent.resolve()))


# ---------------------------------------------------------------------------
# run_metric
# ---------------------------------------------------------------------------

class TestRunMetricCircuitFigure:
    """Tests that run_metric returns circuit_figure from the submitter."""

    @patch("metric_executor.runpy")
    def test_returns_circuit_figure_without_consumption(self, mock_runpy, monkeypatch):
        """
        Given: Submitter in registry with get_sample_circuit_figure() returning a fig
        When: run_metric called with calculate_consumption=False
        Expected: output contains 'circuit_figure' equal to mock fig
        """
        from _helpers.registry import submitter_registry
        from metric_executor import run_metric

        mock_fig = Mock(name="circuit_figure")
        mock_submitter = Mock()
        mock_submitter.get_sample_circuit_figure.return_value = mock_fig
        mock_submitter.get_noise_model_metadata.return_value = {}
        submitter_registry.store_submitter(mock_submitter, "noisy_sim")

        monkeypatch.setenv("PERF_VALUE", "0.75")

        output = run_metric("dummy_path", calculate_consumption=False)

        assert output["circuit_figure"] is mock_fig

    @patch("metric_executor.runpy")
    def test_returns_circuit_figure_with_consumption(self, mock_runpy, monkeypatch):
        """
        Given: Submitter in registry, calculate_consumption=True
        When: run_metric called
        Expected: output still contains circuit_figure
        """
        from _helpers.registry import submitter_registry
        from metric_executor import run_metric

        mock_fig = Mock(name="circuit_figure")
        mock_submitter = Mock()
        mock_submitter.get_sample_circuit_figure.return_value = mock_fig
        mock_submitter.get_noise_model_metadata.return_value = {}
        mock_submitter.get_power_consumption.return_value = (
            {"cx": 10, "rz": 5},
            [{"cx": 10, "rz": 5}],
        )
        submitter_registry.store_submitter(mock_submitter, "noisy_sim")

        monkeypatch.setenv("PERF_VALUE", "0.75")
        monkeypatch.setenv("NUM_QUBITS", "4")

        with patch("metric_executor.get_num_qubits", return_value=[4]):
            output = run_metric("dummy_path", calculate_consumption=True)

        assert output["circuit_figure"] is mock_fig

    @patch("metric_executor.runpy")
    def test_returns_none_when_submitter_returns_none(self, mock_runpy, monkeypatch):
        """
        Given: Submitter's get_sample_circuit_figure() returns None
        When: run_metric called
        Expected: output has circuit_figure=None
        """
        from _helpers.registry import submitter_registry
        from metric_executor import run_metric

        mock_submitter = Mock()
        mock_submitter.get_sample_circuit_figure.return_value = None
        mock_submitter.get_noise_model_metadata.return_value = {}
        submitter_registry.store_submitter(mock_submitter, "noisy_sim")

        monkeypatch.setenv("PERF_VALUE", "0.75")

        output = run_metric("dummy_path", calculate_consumption=False)

        assert output["circuit_figure"] is None


class TestRunMetricNoSubmitter:
    """Tests run_metric when no submitter is in the registry."""

    @patch("metric_executor.runpy")
    def test_returns_none_figure_when_no_submitter(self, mock_runpy, monkeypatch):
        """
        Given: No submitter in registry
        When: run_metric called
        Expected: output has circuit_figure=None
        """
        from metric_executor import run_metric

        monkeypatch.setenv("PERF_VALUE", "0.75")

        output = run_metric("dummy_path", calculate_consumption=False)

        assert output["circuit_figure"] is None


# ---------------------------------------------------------------------------
# optimise
# ---------------------------------------------------------------------------

class TestOptimiseCircuitFigure:
    """Tests that optimise captures circuit_figure from the first iteration."""

    @staticmethod
    def _configure_plt_mock(mock_plt):
        """Configure plt mock so subplots() returns (fig, ax) tuple."""
        mock_plt.subplots.return_value = (Mock(name="plot_fig"), Mock(name="plot_ax"))

    @patch("optimiser.plt")
    @patch("optimiser.extract_metric_name", return_value="test_metric")
    @patch("optimiser.get_config_value", return_value=20.0)
    @patch("optimiser.control_parameter_registry")
    @patch("optimiser.set_circuit_optimisation")
    @patch("optimiser.read_config")
    @patch("optimiser.run_metric")
    def test_captures_from_first_iteration(
        self, mock_run, mock_read, mock_set_opt, mock_cp_reg,
        mock_get_val, mock_extract, mock_plt
    ):
        """
        Given: 3 iterations, all return a fig
        When: optimise called
        Expected: output circuit_figure is the fig from first iteration
        """
        from optimiser import optimise

        self._configure_plt_mock(mock_plt)
        mock_fig = Mock(name="circuit_fig")
        mock_read.return_value = {"optimisation_iterations": 3}
        mock_run.return_value = {
            "performance": 0.8,
            "nm_metadata": {},
            "circuit_figure": mock_fig,
        }
        mock_cp_reg.get_control_parameters.return_value = {"temperature": [20, "mK"]}

        output = optimise("dummy_path", save_image=False, show_plot=False, display_metadata=False)

        assert output["circuit_figure"] is mock_fig

    @patch("optimiser.plt")
    @patch("optimiser.extract_metric_name", return_value="test_metric")
    @patch("optimiser.get_config_value", return_value=20.0)
    @patch("optimiser.control_parameter_registry")
    @patch("optimiser.set_circuit_optimisation")
    @patch("optimiser.read_config")
    @patch("optimiser.run_metric")
    def test_ignores_later_iterations(
        self, mock_run, mock_read, mock_set_opt, mock_cp_reg,
        mock_get_val, mock_extract, mock_plt
    ):
        """
        Given: 2 iterations returning fig_a then fig_b
        When: optimise called
        Expected: output circuit_figure is fig_a
        """
        from optimiser import optimise

        self._configure_plt_mock(mock_plt)
        fig_a = Mock(name="fig_a")
        fig_b = Mock(name="fig_b")
        mock_read.return_value = {"optimisation_iterations": 2}
        mock_run.side_effect = [
            {"performance": 0.8, "nm_metadata": {}, "circuit_figure": fig_a},
            {"performance": 0.7, "nm_metadata": {}, "circuit_figure": fig_b},
        ]
        mock_cp_reg.get_control_parameters.return_value = {"temperature": [20, "mK"]}

        output = optimise("dummy_path", save_image=False, show_plot=False, display_metadata=False)

        assert output["circuit_figure"] is fig_a

    @patch("optimiser.plt")
    @patch("optimiser.extract_metric_name", return_value="test_metric")
    @patch("optimiser.get_config_value", return_value=20.0)
    @patch("optimiser.control_parameter_registry")
    @patch("optimiser.set_circuit_optimisation")
    @patch("optimiser.read_config")
    @patch("optimiser.run_metric")
    def test_returns_none_when_run_metric_returns_none(
        self, mock_run, mock_read, mock_set_opt, mock_cp_reg,
        mock_get_val, mock_extract, mock_plt
    ):
        """
        Given: 1 iteration returns None figure
        When: optimise called
        Expected: output circuit_figure is None
        """
        from optimiser import optimise

        self._configure_plt_mock(mock_plt)
        mock_read.return_value = {"optimisation_iterations": 1}
        mock_run.return_value = {
            "performance": 0.8,
            "nm_metadata": {},
            "circuit_figure": None,
        }
        mock_cp_reg.get_control_parameters.return_value = {"temperature": [20, "mK"]}

        output = optimise("dummy_path", save_image=False, show_plot=False, display_metadata=False)

        assert output["circuit_figure"] is None


# ---------------------------------------------------------------------------
# depth_sweep
# ---------------------------------------------------------------------------

class TestDepthSweepCircuitFigure:
    """Tests that depth_sweep captures circuit_figure from the first depth."""

    @staticmethod
    def _configure_plt_mock(mock_plt):
        """Configure plt mock so subplots() returns (fig, ax) tuple."""
        mock_plt.subplots.return_value = (Mock(name="plot_fig"), Mock(name="plot_ax"))

    @patch("depth_optimiser.curve_fit")
    @patch("depth_optimiser.plt")
    @patch("depth_optimiser.set_mirror_depth")
    @patch("depth_optimiser.get_num_qubits", return_value=4)
    @patch("depth_optimiser.set_circuit_optimisation")
    @patch("depth_optimiser.read_config")
    @patch("depth_optimiser.run_metric")
    def test_captures_from_first_depth(
        self, mock_run, mock_read, mock_set_opt, mock_get_nq,
        mock_set_depth, mock_plt, mock_curve_fit
    ):
        """
        Given: 2 depths, both return a fig
        When: depth_sweep called
        Expected: output circuit_figure is the fig from first depth
        """
        import numpy as np
        from depth_optimiser import depth_sweep

        self._configure_plt_mock(mock_plt)
        mock_fig = Mock(name="circuit_fig")
        mock_read.return_value = {}
        mock_run.return_value = {
            "performance": 0.9,
            "circuit_figure": mock_fig,
        }
        mock_curve_fit.return_value = (np.array([0.99, 0.8, 0.0]), np.zeros((3, 3)))

        output = depth_sweep("dummy_path", depths=[1, 2], save_image=False, show_plot=False)

        assert output["circuit_figure"] is mock_fig

    @patch("depth_optimiser.curve_fit")
    @patch("depth_optimiser.plt")
    @patch("depth_optimiser.set_mirror_depth")
    @patch("depth_optimiser.get_num_qubits", return_value=4)
    @patch("depth_optimiser.set_circuit_optimisation")
    @patch("depth_optimiser.read_config")
    @patch("depth_optimiser.run_metric")
    def test_ignores_later_depths(
        self, mock_run, mock_read, mock_set_opt, mock_get_nq,
        mock_set_depth, mock_plt, mock_curve_fit
    ):
        """
        Given: 2 depths returning fig_a then fig_b
        When: depth_sweep called
        Expected: output circuit_figure is fig_a
        """
        import numpy as np
        from depth_optimiser import depth_sweep

        self._configure_plt_mock(mock_plt)
        fig_a = Mock(name="fig_a")
        fig_b = Mock(name="fig_b")
        mock_read.return_value = {}
        mock_run.side_effect = [
            {"performance": 0.9, "circuit_figure": fig_a},
            {"performance": 0.8, "circuit_figure": fig_b},
        ]
        mock_curve_fit.return_value = (np.array([0.99, 0.8, 0.0]), np.zeros((3, 3)))

        output = depth_sweep("dummy_path", depths=[1, 2], save_image=False, show_plot=False)

        assert output["circuit_figure"] is fig_a

    @patch("depth_optimiser.curve_fit")
    @patch("depth_optimiser.plt")
    @patch("depth_optimiser.set_mirror_depth")
    @patch("depth_optimiser.get_num_qubits", return_value=4)
    @patch("depth_optimiser.set_circuit_optimisation")
    @patch("depth_optimiser.read_config")
    @patch("depth_optimiser.run_metric")
    def test_returns_none_when_no_figure(
        self, mock_run, mock_read, mock_set_opt, mock_get_nq,
        mock_set_depth, mock_plt, mock_curve_fit
    ):
        """
        Given: All iterations return None figure
        When: depth_sweep called
        Expected: output circuit_figure is None
        """
        import numpy as np
        from depth_optimiser import depth_sweep

        self._configure_plt_mock(mock_plt)
        mock_read.return_value = {}
        mock_run.return_value = {
            "performance": 0.9,
            "circuit_figure": None,
        }
        mock_curve_fit.return_value = (np.array([0.99, 0.8, 0.0]), np.zeros((3, 3)))

        output = depth_sweep("dummy_path", depths=[1, 2], save_image=False, show_plot=False)

        assert output["circuit_figure"] is None


# ---------------------------------------------------------------------------
# ExperimentRunner dispatch
# ---------------------------------------------------------------------------

class TestExperimentRunnerDispatch:
    """Tests that _run_single, _run_optimiser, _run_depth_sweep pass circuit_figure in extras."""

    def _make_runner(self, tmp_path):
        """Create an ExperimentRunner pointing at tmp_path, bypassing real config."""
        from experiment_runner import ExperimentRunner

        runner = object.__new__(ExperimentRunner)
        runner.output_dir = tmp_path
        runner.timestamp = "test"
        runner.results_file = tmp_path / "results.json"
        runner.results = []
        runner.original_config = {}
        return runner

    @patch("experiment_runner.set_circuit_optimisation")
    @patch("experiment_runner.run_metric")
    def test_run_single_passes_circuit_figure_in_extras(
        self, mock_run, mock_set_opt, tmp_path
    ):
        """
        Given: run_metric returns output with circuit_figure
        When: _run_single called
        Expected: extras dict contains circuit_figure
        """
        mock_fig = Mock(name="circuit_fig")
        mock_run.return_value = {"performance": 0.8, "circuit_figure": mock_fig}

        runner = self._make_runner(tmp_path)
        perfs, x_vals, x_label, fig, meta_fig, extras = runner._run_single("dummy")

        assert extras["circuit_figure"] is mock_fig

    @patch("experiment_runner.set_num_qubits_list")
    @patch("experiment_runner.optimise")
    def test_run_optimiser_passes_circuit_figure_in_extras(
        self, mock_optimise, mock_set_nq, tmp_path
    ):
        """
        Given: optimise returns output with circuit_figure
        When: _run_optimiser called
        Expected: extras dict contains circuit_figure
        """
        mock_fig = Mock(name="circuit_fig")
        mock_optimise.return_value = {
            "performances": [0.8],
            "temperatures": [20],
            "figure": Mock(),
            "nm_metadata": [],
            "metadata_figure": None,
            "circuit_figure": mock_fig,
        }

        runner = self._make_runner(tmp_path)
        perfs, x_vals, x_label, fig, meta_fig, extras = runner._run_optimiser("dummy")

        assert extras["circuit_figure"] is mock_fig

    @patch("experiment_runner.set_num_qubits_list")
    @patch("experiment_runner.depth_sweep")
    def test_run_depth_sweep_passes_circuit_figure_in_extras(
        self, mock_depth, mock_set_nq, tmp_path
    ):
        """
        Given: depth_sweep returns output with circuit_figure
        When: _run_depth_sweep called
        Expected: extras dict contains circuit_figure
        """
        import numpy as np

        mock_fig = Mock(name="circuit_fig")
        mock_depth.return_value = {
            "performances": [0.9, 0.8],
            "depths": [1, 2],
            "fit_params": np.array([0.99, 0.8, 0.0]),
            "pcov": np.zeros((3, 3)),
            "alpha": 0.99,
            "num_qubits": 4,
            "figure": Mock(),
            "circuit_figure": mock_fig,
        }

        runner = self._make_runner(tmp_path)
        perfs, x_vals, x_label, fig, meta_fig, extras = runner._run_depth_sweep("dummy", [1, 2])

        assert extras["circuit_figure"] is mock_fig


# ---------------------------------------------------------------------------
# ExperimentRunner saves circuit figure
# ---------------------------------------------------------------------------

class TestExperimentRunnerSavesCircuit:
    """Tests that run_trial saves circuit figure to circuits/ subfolder."""

    def _make_runner(self, tmp_path):
        from experiment_runner import ExperimentRunner

        runner = object.__new__(ExperimentRunner)
        runner.output_dir = tmp_path
        runner.timestamp = "test"
        runner.results_file = tmp_path / "results.json"
        runner.results = []
        runner.original_config = {}
        return runner

    @patch("experiment_runner.resolve_metric_path", return_value="dummy_path")
    @patch("experiment_runner.ExperimentRunner.display_trial_info")
    @patch("experiment_runner.ExperimentRunner._build_full_config")
    @patch("experiment_runner.ExperimentRunner._run_optimiser")
    def test_saves_figure_to_circuits_subfolder(
        self, mock_run_opt, mock_build, mock_display, mock_resolve, tmp_path
    ):
        """
        Given: extras has a mock figure
        When: run_trial called
        Expected: save_plot called with circuits dir path
        """
        mock_fig = Mock(name="circuit_fig")
        mock_fig.savefig = Mock()
        mock_perf_fig = Mock(name="perf_fig")
        mock_perf_fig.savefig = Mock()

        mock_build.return_value = (
            {"noise_models": {"experiment": {"name": "tokyo", "type": "fake_backend"}},
             "selected_noise_model": "experiment"},
            "optimiser",
            "quantum_volume",
        )
        mock_run_opt.return_value = (
            [0.8], [20], "Temperature (mK)", mock_perf_fig, None,
            {"nm_metadata": [], "circuit_figure": mock_fig},
        )

        runner = self._make_runner(tmp_path)
        trials_dir = tmp_path / "trials"
        trials_dir.mkdir()

        with patch("experiment_runner.ExperimentRunner._save_config"):
            result = runner.run_trial(
                0,
                {"run_type": "optimiser", "metric": "quantum_volume",
                 "num_qubits": 4, "noise_model_type": "fake_backend",
                 "noise_model_name": "tokyo"},
                output_dir=trials_dir,
            )

        # Verify the circuit figure was saved
        mock_fig.savefig.assert_called_once()
        call_args = mock_fig.savefig.call_args
        saved_path = str(call_args[0][0])
        assert "circuits" in saved_path

    @patch("experiment_runner.resolve_metric_path", return_value="dummy_path")
    @patch("experiment_runner.ExperimentRunner.display_trial_info")
    @patch("experiment_runner.ExperimentRunner._build_full_config")
    @patch("experiment_runner.ExperimentRunner._run_optimiser")
    def test_sets_circuit_plot_path_in_result(
        self, mock_run_opt, mock_build, mock_display, mock_resolve, tmp_path
    ):
        """
        Given: extras has a mock figure
        When: run_trial called
        Expected: result dict has circuit_plot_path string
        """
        mock_fig = Mock(name="circuit_fig")
        mock_fig.savefig = Mock()
        mock_perf_fig = Mock(name="perf_fig")
        mock_perf_fig.savefig = Mock()

        mock_build.return_value = (
            {"noise_models": {"experiment": {"name": "tokyo", "type": "fake_backend"}},
             "selected_noise_model": "experiment"},
            "optimiser",
            "quantum_volume",
        )
        mock_run_opt.return_value = (
            [0.8], [20], "Temperature (mK)", mock_perf_fig, None,
            {"nm_metadata": [], "circuit_figure": mock_fig},
        )

        runner = self._make_runner(tmp_path)
        trials_dir = tmp_path / "trials"
        trials_dir.mkdir()

        with patch("experiment_runner.ExperimentRunner._save_config"):
            result = runner.run_trial(
                0,
                {"run_type": "optimiser", "metric": "quantum_volume",
                 "num_qubits": 4, "noise_model_type": "fake_backend",
                 "noise_model_name": "tokyo"},
                output_dir=trials_dir,
            )

        assert result["circuit_plot_path"] is not None
        assert isinstance(result["circuit_plot_path"], str)

    @patch("experiment_runner.resolve_metric_path", return_value="dummy_path")
    @patch("experiment_runner.ExperimentRunner.display_trial_info")
    @patch("experiment_runner.ExperimentRunner._build_full_config")
    @patch("experiment_runner.ExperimentRunner._run_optimiser")
    def test_creates_circuits_directory(
        self, mock_run_opt, mock_build, mock_display, mock_resolve, tmp_path
    ):
        """
        Given: extras has a mock figure
        When: run_trial called
        Expected: circuits/ subdir created under output_dir
        """
        mock_fig = Mock(name="circuit_fig")
        mock_fig.savefig = Mock()
        mock_perf_fig = Mock(name="perf_fig")
        mock_perf_fig.savefig = Mock()

        mock_build.return_value = (
            {"noise_models": {"experiment": {"name": "tokyo", "type": "fake_backend"}},
             "selected_noise_model": "experiment"},
            "optimiser",
            "quantum_volume",
        )
        mock_run_opt.return_value = (
            [0.8], [20], "Temperature (mK)", mock_perf_fig, None,
            {"nm_metadata": [], "circuit_figure": mock_fig},
        )

        runner = self._make_runner(tmp_path)
        trials_dir = tmp_path / "trials"
        trials_dir.mkdir()

        with patch("experiment_runner.ExperimentRunner._save_config"):
            runner.run_trial(
                0,
                {"run_type": "optimiser", "metric": "quantum_volume",
                 "num_qubits": 4, "noise_model_type": "fake_backend",
                 "noise_model_name": "tokyo"},
                output_dir=trials_dir,
            )

        circuits_dir = trials_dir / "circuits"
        assert circuits_dir.exists()
        assert circuits_dir.is_dir()


class TestExperimentRunnerNoCircuit:
    """Tests run_trial behaviour when circuit_figure is None."""

    def _make_runner(self, tmp_path):
        from experiment_runner import ExperimentRunner

        runner = object.__new__(ExperimentRunner)
        runner.output_dir = tmp_path
        runner.timestamp = "test"
        runner.results_file = tmp_path / "results.json"
        runner.results = []
        runner.original_config = {}
        return runner

    @patch("experiment_runner.resolve_metric_path", return_value="dummy_path")
    @patch("experiment_runner.ExperimentRunner.display_trial_info")
    @patch("experiment_runner.ExperimentRunner._build_full_config")
    @patch("experiment_runner.ExperimentRunner._run_optimiser")
    def test_no_save_when_figure_is_none(
        self, mock_run_opt, mock_build, mock_display, mock_resolve, tmp_path
    ):
        """
        Given: extras has circuit_figure=None
        When: run_trial called
        Expected: save_plot not called for circuit
        """
        mock_perf_fig = Mock(name="perf_fig")
        mock_perf_fig.savefig = Mock()

        mock_build.return_value = (
            {"noise_models": {"experiment": {"name": "tokyo", "type": "fake_backend"}},
             "selected_noise_model": "experiment"},
            "optimiser",
            "quantum_volume",
        )
        mock_run_opt.return_value = (
            [0.8], [20], "Temperature (mK)", mock_perf_fig, None,
            {"nm_metadata": [], "circuit_figure": None},
        )

        runner = self._make_runner(tmp_path)
        trials_dir = tmp_path / "trials"
        trials_dir.mkdir()

        with patch("experiment_runner.ExperimentRunner._save_config"), \
             patch.object(runner, "save_plot", wraps=runner.save_plot) as spy_save:
            result = runner.run_trial(
                0,
                {"run_type": "optimiser", "metric": "quantum_volume",
                 "num_qubits": 4, "noise_model_type": "fake_backend",
                 "noise_model_name": "tokyo"},
                output_dir=trials_dir,
            )

        # save_plot should have been called for perf fig and metadata fig,
        # but NOT for circuit fig (the circuit branch is guarded by `if circuit_fig is not None`)
        for call in spy_save.call_args_list:
            args = call[0]
            if len(args) >= 2:
                assert "circuit" not in str(args[1]).lower() or args[0] is None

    @patch("experiment_runner.resolve_metric_path", return_value="dummy_path")
    @patch("experiment_runner.ExperimentRunner.display_trial_info")
    @patch("experiment_runner.ExperimentRunner._build_full_config")
    @patch("experiment_runner.ExperimentRunner._run_optimiser")
    def test_circuit_plot_path_is_none(
        self, mock_run_opt, mock_build, mock_display, mock_resolve, tmp_path
    ):
        """
        Given: extras has circuit_figure=None
        When: run_trial called
        Expected: result dict circuit_plot_path is None
        """
        mock_perf_fig = Mock(name="perf_fig")
        mock_perf_fig.savefig = Mock()

        mock_build.return_value = (
            {"noise_models": {"experiment": {"name": "tokyo", "type": "fake_backend"}},
             "selected_noise_model": "experiment"},
            "optimiser",
            "quantum_volume",
        )
        mock_run_opt.return_value = (
            [0.8], [20], "Temperature (mK)", mock_perf_fig, None,
            {"nm_metadata": [], "circuit_figure": None},
        )

        runner = self._make_runner(tmp_path)
        trials_dir = tmp_path / "trials"
        trials_dir.mkdir()

        with patch("experiment_runner.ExperimentRunner._save_config"):
            result = runner.run_trial(
                0,
                {"run_type": "optimiser", "metric": "quantum_volume",
                 "num_qubits": 4, "noise_model_type": "fake_backend",
                 "noise_model_name": "tokyo"},
                output_dir=trials_dir,
            )

        assert result["circuit_plot_path"] is None

    @patch("experiment_runner.resolve_metric_path", return_value="dummy_path")
    @patch("experiment_runner.ExperimentRunner.display_trial_info")
    @patch("experiment_runner.ExperimentRunner._build_full_config")
    @patch("experiment_runner.ExperimentRunner._run_optimiser")
    def test_no_circuits_dir_created(
        self, mock_run_opt, mock_build, mock_display, mock_resolve, tmp_path
    ):
        """
        Given: extras has circuit_figure=None
        When: run_trial called
        Expected: no circuits/ subdir created
        """
        mock_perf_fig = Mock(name="perf_fig")
        mock_perf_fig.savefig = Mock()

        mock_build.return_value = (
            {"noise_models": {"experiment": {"name": "tokyo", "type": "fake_backend"}},
             "selected_noise_model": "experiment"},
            "optimiser",
            "quantum_volume",
        )
        mock_run_opt.return_value = (
            [0.8], [20], "Temperature (mK)", mock_perf_fig, None,
            {"nm_metadata": [], "circuit_figure": None},
        )

        runner = self._make_runner(tmp_path)
        trials_dir = tmp_path / "trials"
        trials_dir.mkdir()

        with patch("experiment_runner.ExperimentRunner._save_config"):
            runner.run_trial(
                0,
                {"run_type": "optimiser", "metric": "quantum_volume",
                 "num_qubits": 4, "noise_model_type": "fake_backend",
                 "noise_model_name": "tokyo"},
                output_dir=trials_dir,
            )

        circuits_dir = trials_dir / "circuits"
        assert not circuits_dir.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

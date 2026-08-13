"""Tests for `nobrainer export bundle` (nobrainer/export/bundle.py)."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import monai
from monai.bundle import ConfigParser, verify_net_in_out
import numpy as np
import pytest
import torch

import nobrainer
from nobrainer.export.bundle import (
    SUPPORTED_ARCHITECTURES,
    BundleExportError,
    build_metadata,
    export_bundle,
)
from nobrainer.models import get as get_model
from nobrainer.processing.segmentation import Segmentation

REQUIRED_METADATA_KEYS = {
    "schema",
    "version",
    "monai_version",
    "pytorch_version",
    "numpy_version",
    "required_packages_version",
    "task",
    "description",
    "authors",
    "copyright",
    "network_data_format",
}
REQUIRED_TENSOR_KEYS = {
    "type",
    "format",
    "num_channels",
    "spatial_shape",
    "dtype",
    "value_range",
}


def _save_estimator(
    tmp_path: Path,
    *,
    base_model: str = "unet",
    model_args: dict | None = None,
    n_classes: int = 3,
    block_shape: tuple[int, int, int] | list = (16, 16, 16),
    dirname: str = "my_model",
) -> Path:
    """Build a tiny trained-looking estimator and save it like ``fit()`` would."""
    model_args = (
        model_args
        if model_args is not None
        else {
            "channels": (4, 8),
            "strides": (2,),
        }
    )
    est = Segmentation(base_model, model_args=model_args)
    est.model_ = get_model(base_model)(n_classes=n_classes, **model_args)
    est.n_classes_ = n_classes
    est.block_shape_ = tuple(block_shape) if block_shape else block_shape
    est.volume_shape_ = tuple(block_shape) if block_shape else block_shape
    est._training_result = {"history": [{"loss": 0.5}, {"loss": 0.3}]}
    est._dataset = None
    save_dir = tmp_path / dirname
    est.save(save_dir)
    return save_dir


@pytest.fixture
def unet_model_dir(tmp_path: Path) -> Path:
    return _save_estimator(tmp_path)


@pytest.fixture
def exported_bundle(tmp_path: Path, unet_model_dir: Path) -> Path:
    return export_bundle(unet_model_dir, tmp_path / "MyBundle", verify=False)


class TestLayout:
    def test_required_files_present(self, exported_bundle: Path) -> None:
        assert (exported_bundle / "LICENSE").is_file()
        assert (exported_bundle / "configs" / "metadata.json").is_file()
        assert (exported_bundle / "configs" / "inference.json").is_file()
        assert (exported_bundle / "models" / "model.pt").is_file()
        assert (exported_bundle / "docs" / "README.md").is_file()

    def test_output_dir_must_not_exist(
        self, tmp_path: Path, unet_model_dir: Path
    ) -> None:
        out = tmp_path / "AlreadyThere"
        out.mkdir()
        with pytest.raises(BundleExportError):
            export_bundle(unet_model_dir, out, verify=False)

    def test_readme_has_not_for_clinical_use_disclaimer(
        self, exported_bundle: Path
    ) -> None:
        readme = (exported_bundle / "docs" / "README.md").read_text()
        assert "NOT FOR CLINICAL USE" in readme


class TestMetadataSchema:
    def test_required_top_level_keys_present(self, exported_bundle: Path) -> None:
        metadata = json.loads(
            (exported_bundle / "configs" / "metadata.json").read_text()
        )
        missing = REQUIRED_METADATA_KEYS - metadata.keys()
        assert not missing, f"missing required metadata keys: {missing}"

    def test_required_tensor_keys_present(self, exported_bundle: Path) -> None:
        metadata = json.loads(
            (exported_bundle / "configs" / "metadata.json").read_text()
        )
        ndf = metadata["network_data_format"]
        for block in (ndf["inputs"]["image"], ndf["outputs"]["pred"]):
            missing = REQUIRED_TENSOR_KEYS - block.keys()
            assert not missing, f"missing required tensor keys: {missing}"

    def test_versions_are_read_live_not_hardcoded(self, exported_bundle: Path) -> None:
        metadata = json.loads(
            (exported_bundle / "configs" / "metadata.json").read_text()
        )
        assert metadata["monai_version"] == monai.__version__
        assert metadata["numpy_version"] == np.__version__
        assert "+" not in metadata["pytorch_version"]
        assert metadata["pytorch_version"] == torch.__version__.split("+")[0]
        assert (
            metadata["required_packages_version"]["nobrainer"] == nobrainer.__version__
        )

    def test_spatial_shape_matches_block_shape(self, exported_bundle: Path) -> None:
        metadata = json.loads(
            (exported_bundle / "configs" / "metadata.json").read_text()
        )
        inference = json.loads(
            (exported_bundle / "configs" / "inference.json").read_text()
        )
        image_shape = metadata["network_data_format"]["inputs"]["image"][
            "spatial_shape"
        ]
        pred_shape = metadata["network_data_format"]["outputs"]["pred"]["spatial_shape"]
        assert image_shape == [16, 16, 16]
        assert pred_shape == [16, 16, 16]
        assert inference["inferer"]["roi_size"] == [16, 16, 16]

    def test_channel_def_background_and_count(self, exported_bundle: Path) -> None:
        metadata = json.loads(
            (exported_bundle / "configs" / "metadata.json").read_text()
        )
        channel_def = metadata["network_data_format"]["outputs"]["pred"]["channel_def"]
        assert len(channel_def) == 3
        assert channel_def["0"] == "background"
        assert set(channel_def) == {"0", "1", "2"}

    def test_channel_def_labels_override(self) -> None:
        metadata = build_metadata(
            base_model="unet",
            in_channels=1,
            n_classes=3,
            spatial_shape=(16, 16, 16),
            provenance={},
            version="0.0.1",
            name=None,
            task=None,
            description=None,
            authors="a",
            copyright_="c",
            labels=["bg", "gray", "white"],
            references=None,
            stochastic=False,
        )
        channel_def = metadata["network_data_format"]["outputs"]["pred"]["channel_def"]
        assert channel_def == {"0": "bg", "1": "gray", "2": "white"}

    def test_labels_length_mismatch_raises(self) -> None:
        with pytest.raises(BundleExportError):
            build_metadata(
                base_model="unet",
                in_channels=1,
                n_classes=3,
                spatial_shape=(16, 16, 16),
                provenance={},
                version="0.0.1",
                name=None,
                task=None,
                description=None,
                authors="a",
                copyright_="c",
                labels=["only_one"],
                references=None,
                stochastic=False,
            )


class TestRoundTrip:
    def test_network_def_resolves_and_loads_state_dict(
        self, exported_bundle: Path
    ) -> None:
        parser = ConfigParser()
        parser.read_config(str(exported_bundle / "configs" / "inference.json"))
        net = parser.get_parsed_content("network_def")
        assert isinstance(net, torch.nn.Module)
        state = torch.load(exported_bundle / "models" / "model.pt", weights_only=True)
        net.load_state_dict(state, strict=True)

    def test_verify_net_in_out(self, exported_bundle: Path) -> None:
        verify_net_in_out(
            net_id="network_def",
            meta_file=str(exported_bundle / "configs" / "metadata.json"),
            config_file=str(exported_bundle / "configs" / "inference.json"),
            device="cpu",
        )


class TestSpatialShapeResolution:
    def test_missing_block_shape_raises(self, tmp_path: Path) -> None:
        model_dir = _save_estimator(tmp_path, block_shape=())
        with pytest.raises(BundleExportError):
            export_bundle(model_dir, tmp_path / "Bundle", verify=False)

    def test_spatial_shape_override(self, tmp_path: Path) -> None:
        model_dir = _save_estimator(tmp_path, block_shape=())
        out = export_bundle(
            model_dir,
            tmp_path / "Bundle",
            spatial_shape=(16, 16, 16),
            verify=False,
        )
        metadata = json.loads((out / "configs" / "metadata.json").read_text())
        assert metadata["network_data_format"]["inputs"]["image"]["spatial_shape"] == [
            16,
            16,
            16,
        ]


class TestArchitectureScope:
    def test_rejected_architecture_raises(self, tmp_path: Path) -> None:
        model_dir = _save_estimator(
            tmp_path,
            base_model="autoencoder",
            model_args={"input_shape": (16, 16, 16)},
            n_classes=1,
        )
        with pytest.raises(BundleExportError, match="autoencoder"):
            export_bundle(model_dir, tmp_path / "Bundle", verify=False)

    def test_supported_architectures_excludes_non_segmentation_nets(self) -> None:
        assert "autoencoder" not in SUPPORTED_ARCHITECTURES
        assert "simsiam" not in SUPPORTED_ARCHITECTURES
        assert "dcgan" not in SUPPORTED_ARCHITECTURES
        assert "progressivegan" not in SUPPORTED_ARCHITECTURES


class TestTorchScript:
    def test_script_failure_warns_and_omits_model_ts(self, tmp_path: Path) -> None:
        model_dir = _save_estimator(
            tmp_path,
            base_model="swin_unetr",
            model_args={"feature_size": 12},
            n_classes=2,
            block_shape=(64, 64, 64),
        )
        with pytest.warns(UserWarning, match="torch.jit.script failed"):
            out = export_bundle(model_dir, tmp_path / "Bundle", verify=False)
        assert not (out / "models" / "model.ts").exists()
        assert (out / "models" / "model.pt").exists()
        assert (out / "configs" / "metadata.json").exists()

    def test_script_success_writes_model_ts(self, exported_bundle: Path) -> None:
        assert (exported_bundle / "models" / "model.ts").exists()


class TestStochasticGate:
    def test_stochastic_model_requires_allow_stochastic(self, tmp_path: Path) -> None:
        model_dir = _save_estimator(
            tmp_path,
            base_model="bayesian_meshnet",
            model_args={"filters": 8, "receptive_field": 37},
            n_classes=2,
            block_shape=(8, 8, 8),
        )
        with pytest.raises(BundleExportError, match="stochastic|allow_stochastic"):
            export_bundle(model_dir, tmp_path / "Bundle", verify=False)

    def test_stochastic_model_exports_with_flag(self, tmp_path: Path) -> None:
        model_dir = _save_estimator(
            tmp_path,
            base_model="bayesian_meshnet",
            model_args={"filters": 8, "receptive_field": 37},
            n_classes=2,
            block_shape=(8, 8, 8),
        )
        out = export_bundle(
            model_dir, tmp_path / "Bundle", allow_stochastic=True, verify=False
        )
        metadata = json.loads((out / "configs" / "metadata.json").read_text())
        assert "posterior draw" in metadata["intended_use"]


class TestCLIContract:
    def _help(self, cmd: list[str]) -> str:
        result = subprocess.run(
            [sys.executable, "-m", "nobrainer.cli.main"] + cmd + ["--help"],
            capture_output=True,
            text=True,
        )
        assert (
            result.returncode == 0
        ), f"'{' '.join(cmd)} --help' exited {result.returncode}:\n{result.stderr}"
        return result.stdout

    def test_export_bundle_help_exits_zero(self) -> None:
        self._help(["export", "bundle"])

    @pytest.mark.parametrize(
        "option",
        [
            "--no-torchscript",
            "--trace",
            "--allow-stochastic",
            "--spatial-shape",
            "--version",
            "--name",
            "--task",
            "--description",
            "--authors",
            "--copyright",
            "--labels",
            "--reference",
            "--no-verify",
        ],
    )
    def test_export_bundle_has_option(self, option: str) -> None:
        out = self._help(["export", "bundle"])
        assert option in out

    def test_export_bundle_cli_end_to_end(self, tmp_path: Path) -> None:
        model_dir = _save_estimator(tmp_path)
        out_dir = tmp_path / "CliBundle"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "nobrainer.cli.main",
                "export",
                "bundle",
                str(model_dir),
                str(out_dir),
                "--no-verify",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert (out_dir / "configs" / "metadata.json").is_file()


class TestVerifyMetadataSubprocess:
    """Exercises the `python -m monai.bundle verify_metadata` integration.

    Requires the optional `nobrainer[bundle]` extra (fire, jsonschema) and
    network access to fetch the schema once; skips gracefully otherwise so
    the rest of the suite is not network-dependent.
    """

    def test_verify_metadata_exits_zero_on_fresh_export(self, tmp_path: Path) -> None:
        pytest.importorskip("fire")
        pytest.importorskip("jsonschema")
        model_dir = _save_estimator(tmp_path)
        try:
            out = export_bundle(model_dir, tmp_path / "Bundle", verify=True)
        except BundleExportError as exc:
            pytest.fail(f"verify_metadata rejected a fresh export: {exc}")
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "monai.bundle",
                "verify_metadata",
                "--meta_file",
                str(out / "configs" / "metadata.json"),
                "--filepath",
                str(tmp_path / "schema_cache.json"),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr

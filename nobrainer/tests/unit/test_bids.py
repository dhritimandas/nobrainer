"""Tests for nobrainer.data.bids -- BIDS discovery and reshaping."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import nibabel as nib
import numpy as np
import pytest

from nobrainer.data.bids import (
    SkipReason,
    _extract_subject_id,
    parse_bids_filename,
    scan_bids,
    to_bids,
)
from nobrainer.data.spec import DataSpec, Severity, validate

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_nifti(path: Path, shape: tuple[int, ...] = (8, 8, 8)) -> Path:
    """Create a tiny NIfTI file at *path*, creating parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.random.rand(*shape).astype(np.float32)
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(path))
    return path


def _make_bids_tree(root: Path) -> Path:
    """A minimal, valid BIDS tree: sub-01 (flat), sub-02 (ses-01), sub-03 (image only)."""
    _make_nifti(root / "sub-01/anat/sub-01_T1w.nii.gz")
    _make_nifti(
        root
        / "derivatives/nobrainer/sub-01/anat/sub-01_space-orig_desc-aseg_dseg.nii.gz"
    )
    _make_nifti(root / "sub-02/ses-01/anat/sub-02_ses-01_T1w.nii.gz")
    _make_nifti(
        root
        / "derivatives/nobrainer/sub-02/ses-01/anat"
        / "sub-02_ses-01_space-orig_desc-aseg_dseg.nii.gz"
    )
    _make_nifti(root / "sub-03/anat/sub-03_T1w.nii.gz")
    return root


@pytest.fixture
def bids_tree(tmp_path: Path) -> Path:
    return _make_bids_tree(tmp_path / "bids")


# ---------------------------------------------------------------------------
# Entity grammar
# ---------------------------------------------------------------------------


class TestParseBidsFilename:
    def test_plus_accepted_in_label(self) -> None:
        parsed = parse_bids_filename("sub-01_acq-x+y_T1w.nii.gz")
        assert parsed is not None
        assert parsed.entities["acq"] == "x+y"

    @pytest.mark.parametrize("bad_value", ["x_y", "x-y", "x.y"])
    def test_hyphen_underscore_dot_rejected_in_value(self, bad_value: str) -> None:
        # A literal "-", "_", or "." inside a value breaks the token grammar
        # (it either splits into a new entity or is not alphanumeric/"+").
        name = f"sub-01_acq-{bad_value}_T1w.nii.gz"
        parsed = parse_bids_filename(name)
        if parsed is not None:
            assert parsed.entities.get("acq") != bad_value

    def test_run_is_index_and_keeps_zero_padding(self) -> None:
        parsed = parse_bids_filename("sub-01_run-01_T1w.nii.gz")
        assert parsed is not None
        assert parsed.entities["run"] == "01"

    def test_repeated_entity_is_unparseable(self) -> None:
        assert parse_bids_filename("sub-01_acq-a_acq-b_T1w.nii.gz") is None

    def test_dotfile_is_unparseable(self) -> None:
        assert parse_bids_filename(".sub-01_T1w.nii.gz") is None

    def test_extension_required(self) -> None:
        assert parse_bids_filename("sub-01_T1w") is None


class TestEntityForbiddenForSuffix:
    def test_desc_not_allowed_on_raw_t1w(self, tmp_path: Path) -> None:
        root = tmp_path / "bids"
        _make_nifti(root / "sub-01/anat/sub-01_desc-x_T1w.nii.gz")
        _make_nifti(root / "sub-02/anat/sub-02_T1w.nii.gz")  # keeps the scan non-empty
        result = scan_bids(root, backend="walker", require_labels=False)
        reasons = {s.reason for s in result.skipped}
        assert SkipReason.ENTITY_NOT_ALLOWED in reasons
        assert len(result.entries) == 1  # sub-02 still returned


class TestDotfilesAndNonDatatypeDirsIgnored:
    def test_dotfile_in_anat_ignored(self, tmp_path: Path) -> None:
        root = tmp_path / "bids"
        _make_nifti(root / "sub-01/anat/sub-01_T1w.nii.gz")
        (root / "sub-01/anat/.hidden_T1w.nii.gz").touch()
        result = scan_bids(root, backend="walker", require_labels=False)
        assert len(result.entries) == 1

    def test_non_datatype_dir_ignored(self, tmp_path: Path) -> None:
        root = tmp_path / "bids"
        _make_nifti(root / "sub-01/anat/sub-01_T1w.nii.gz")
        _make_nifti(root / "sub-01/randomstuff/sub-01_T1w.nii.gz")
        result = scan_bids(root, backend="walker", require_labels=False)
        assert len(result.entries) == 1


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


class TestDiscovery:
    def test_flat_tree(self, tmp_path: Path) -> None:
        root = tmp_path / "bids"
        _make_nifti(root / "sub-01/anat/sub-01_T1w.nii.gz")
        _make_nifti(
            root
            / "derivatives/nobrainer/sub-01/anat/sub-01_space-orig_desc-aseg_dseg.nii.gz"
        )
        result = scan_bids(root, backend="walker")
        assert len(result.entries) == 1
        assert result.entries[0]["image"].endswith("sub-01_T1w.nii.gz")
        assert result.entries[0]["label"].endswith("dseg.nii.gz")
        assert result.subject_ids == ["01"]

    def test_session_tree(self, bids_tree: Path) -> None:
        result = scan_bids(bids_tree, backend="walker")
        # sub-02 has a session; its entry must exist and carry the session
        img = next(e for e in result.entries if "sub-02" in e["image"])
        assert "ses-01" in img["image"]
        assert "ses-01" in img["label"]

    def test_ses_in_filename_without_ses_dir_is_skipped(self, tmp_path: Path) -> None:
        root = tmp_path / "bids"
        # filename claims ses-02 but sits directly under sub-01/anat (no ses-02/ dir)
        _make_nifti(root / "sub-01/anat/sub-01_ses-02_T1w.nii.gz")
        _make_nifti(root / "sub-02/anat/sub-02_T1w.nii.gz")  # keeps the scan non-empty
        result = scan_bids(root, backend="walker", require_labels=False)
        assert len(result.entries) == 1
        assert any(s.reason == SkipReason.NOT_A_DATATYPE_DIR for s in result.skipped)

    def test_duplicate_extension(self, tmp_path: Path) -> None:
        root = tmp_path / "bids"
        _make_nifti(root / "sub-01/anat/sub-01_T1w.nii.gz")
        _make_nifti(root / "sub-01/anat/sub-01_T1w.nii")
        _make_nifti(root / "sub-02/anat/sub-02_T1w.nii.gz")  # keeps the scan non-empty
        result = scan_bids(root, backend="walker", require_labels=False)
        assert len(result.entries) == 1
        assert "sub-02" in result.entries[0]["image"]
        assert any(s.reason == SkipReason.DUPLICATE_EXTENSION for s in result.skipped)

    def test_image_with_no_label_is_skipped_but_others_return(
        self, bids_tree: Path
    ) -> None:
        result = scan_bids(bids_tree, backend="walker")
        assert len(result.entries) == 2  # sub-01, sub-02 -- sub-03 has no label
        assert any(s.reason == SkipReason.NO_MATCHING_LABEL for s in result.skipped)
        assert any("sub-03" in s.path for s in result.skipped)

    def test_mixed_labelling_never_yields_partial_label_set(
        self, bids_tree: Path
    ) -> None:
        result = scan_bids(bids_tree, backend="walker")
        has_label = [("label" in e) for e in result.entries]
        assert all(has_label) or not any(has_label)

        result_no_labels = scan_bids(bids_tree, backend="walker", require_labels=False)
        has_label2 = [("label" in e) for e in result_no_labels.entries]
        assert not any(has_label2)
        assert len(result_no_labels.entries) == 3  # all three images, unlabelled

    def test_ambiguous_label(self, tmp_path: Path) -> None:
        root = tmp_path / "bids"
        _make_nifti(root / "sub-01/anat/sub-01_T1w.nii.gz")
        _make_nifti(
            root
            / "derivatives/nobrainer/sub-01/anat/sub-01_space-orig_seg-manual_desc-aseg_dseg.nii.gz"
        )
        _make_nifti(
            root
            / "derivatives/nobrainer/sub-01/anat/sub-01_space-orig_seg-auto_desc-aseg_dseg.nii.gz"
        )
        _make_nifti(root / "sub-02/anat/sub-02_T1w.nii.gz")
        _make_nifti(
            root
            / "derivatives/nobrainer/sub-02/anat/sub-02_space-orig_desc-aseg_dseg.nii.gz"
        )
        result = scan_bids(root, backend="walker")
        assert len(result.entries) == 1
        assert "sub-02" in result.entries[0]["image"]
        assert any(s.reason == SkipReason.AMBIGUOUS_LABEL for s in result.skipped)

    def test_sources_sidecar_overrides_entity_join(self, tmp_path: Path) -> None:
        root = tmp_path / "bids"
        _make_nifti(root / "sub-01/anat/sub-01_acq-a_T1w.nii.gz")
        _make_nifti(root / "sub-01/anat/sub-01_acq-b_T1w.nii.gz")
        label_path = _make_nifti(
            root
            / "derivatives/nobrainer/sub-01/anat/sub-01_space-orig_desc-aseg_dseg.nii.gz"
        )
        # No acq entity on the label -> plain join key would not match either
        # acq-a or acq-b image. Sources sidecar disambiguates to acq-b.
        sidecar = label_path.with_suffix("").with_suffix(".json")
        sidecar.write_text(
            json.dumps({"Sources": ["bids::sub-01/anat/sub-01_acq-b_T1w.nii.gz"]})
        )
        result = scan_bids(root, backend="walker")
        assert len(result.entries) == 1
        assert "acq-b" in result.entries[0]["image"]

    def test_empty_tree_raises_with_histogram(self, tmp_path: Path) -> None:
        root = tmp_path / "bids"
        root.mkdir()
        with pytest.raises(ValueError, match="No BIDS entries found"):
            scan_bids(root, backend="walker")

    def test_irrelevant_tree_raises_with_reasons(self, tmp_path: Path) -> None:
        root = tmp_path / "bids"
        _make_nifti(root / "sub-01/anat/sub-01_desc-x_T1w.nii.gz")  # ENTITY_NOT_ALLOWED
        with pytest.raises(ValueError, match="Skip reasons"):
            scan_bids(root, backend="walker", require_labels=False)

    def test_nonexistent_root_raises_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            scan_bids(tmp_path / "does-not-exist", backend="walker")


# ---------------------------------------------------------------------------
# validate() integration -- the annex path
# ---------------------------------------------------------------------------


class TestValidateIntegration:
    def test_annex_missing_symlink_is_included_and_reported(
        self, tmp_path: Path
    ) -> None:
        root = tmp_path / "bids"
        _make_nifti(root / "sub-01/anat/dummy.nii.gz")  # forces dir creation
        (root / "sub-01/anat/dummy.nii.gz").unlink()

        anat_dir = root / "sub-01/anat"
        anat_dir.mkdir(parents=True, exist_ok=True)
        link = anat_dir / "sub-01_T1w.nii.gz"
        os.symlink("/nonexistent/.git/annex/objects/xx/yy/data", str(link))
        _make_nifti(
            root
            / "derivatives/nobrainer/sub-01/anat/sub-01_space-orig_desc-aseg_dseg.nii.gz"
        )

        result = scan_bids(root, backend="walker")
        assert len(result.entries) == 1  # from_bids does NOT drop it

        spec = DataSpec(entries=result.entries)
        findings = validate(spec)
        assert any(
            f.severity == Severity.ERROR and "datalad get" in f.message
            for f in findings
        )

    def test_from_bids_output_round_trips_through_json(self, bids_tree: Path) -> None:
        spec = DataSpec.from_bids(bids_tree, backend="walker")
        out = bids_tree.parent / "manifest.json"
        spec.to_json(out)
        spec2 = DataSpec.from_json(out)
        assert spec.entries == spec2.entries

    def test_from_bids_entries_pass_validate_cleanly(self, bids_tree: Path) -> None:
        spec = DataSpec.from_bids(bids_tree, backend="walker")
        findings = validate(spec)
        errors = [f for f in findings if f.severity == Severity.ERROR]
        assert errors == []


# ---------------------------------------------------------------------------
# get_dataset compatibility -- the literal goal criterion
# ---------------------------------------------------------------------------


class TestGetDatasetCompatibility:
    def test_from_bids_entries_feed_get_dataset_unmodified(
        self, bids_tree: Path
    ) -> None:
        from nobrainer.dataset import get_dataset

        result = scan_bids(bids_tree, backend="walker")
        image_paths = [e["image"] for e in result.entries]
        label_paths = [e["label"] for e in result.entries]

        loader = get_dataset(image_paths, label_paths, batch_size=1, cache_rate=0.0)
        batch = next(iter(loader))
        assert batch["image"].shape[0] == 1
        assert "label" in batch


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


class TestBackends:
    def test_walker_backend_works_with_pybids_absent(self, bids_tree: Path) -> None:
        # This is the default CI condition (pybids is not installed) --
        # runs unconditionally, no importorskip.
        result = scan_bids(bids_tree, backend="walker")
        assert len(result.entries) == 2

    def test_pybids_backend_without_pybids_raises_clear_import_error(
        self, bids_tree: Path
    ) -> None:
        pytest.importorskip(
            "bids", reason="only meaningful to test when pybids is ABSENT"
        )

    def test_pybids_backend_missing_raises_named_extra(
        self, bids_tree: Path, monkeypatch
    ) -> None:
        import nobrainer.data.bids as bids_mod

        monkeypatch.setattr(bids_mod, "_pybids_available", lambda: False)
        with pytest.raises(ImportError, match=r"\[bids\]"):
            scan_bids(bids_tree, backend="pybids")

    def test_pybids_and_walker_parity(self, bids_tree: Path) -> None:
        pytest.importorskip("bids")
        walker_result = scan_bids(bids_tree, backend="walker")
        pybids_result = scan_bids(bids_tree, backend="pybids")
        assert sorted(walker_result.entries, key=str) == sorted(
            pybids_result.entries, key=str
        )

    def test_pybids_bare_filename_loses_subject_our_wrapper_does_not(
        self, tmp_path: Path
    ) -> None:
        pytest.importorskip("bids")
        from nobrainer.data.bids import _pybids_parse

        path = tmp_path / "bids" / "sub-01" / "anat" / "sub-01_T1w.nii.gz"
        path.parent.mkdir(parents=True)
        path.touch()
        parsed = _pybids_parse(path)
        assert parsed is not None
        assert parsed.entities.get("sub") == "01"


# ---------------------------------------------------------------------------
# to_bids
# ---------------------------------------------------------------------------


class TestToBids:
    def test_round_trip(self, bids_tree: Path, tmp_path: Path) -> None:
        result = scan_bids(bids_tree, backend="walker")
        out = tmp_path / "reshaped"
        report = to_bids(result.entries, out, dataset_name="round trip")
        assert report.n_written == 2

        result2 = scan_bids(out, backend="walker")
        assert len(result2.entries) == 2
        assert all("label" in e for e in result2.entries)

    def test_writes_symlinks_never_copies(
        self, bids_tree: Path, tmp_path: Path
    ) -> None:
        result = scan_bids(bids_tree, backend="walker")
        out = tmp_path / "reshaped"
        to_bids(result.entries, out)
        written = list(out.glob("sub-*/anat/*_T1w.nii.gz"))
        assert written
        for p in written:
            assert p.is_symlink()

    def test_dataset_description_json_both_levels(
        self, bids_tree: Path, tmp_path: Path
    ) -> None:
        result = scan_bids(bids_tree, backend="walker")
        out = tmp_path / "reshaped"
        to_bids(result.entries, out, dataset_name="my dataset")

        root_desc = json.loads((out / "dataset_description.json").read_text())
        assert root_desc["Name"] == "my dataset"
        assert root_desc["BIDSVersion"]
        assert "DatasetType" not in root_desc

        deriv_desc = json.loads(
            (out / "derivatives/nobrainer/dataset_description.json").read_text()
        )
        assert deriv_desc["DatasetType"] == "derivative"
        assert deriv_desc["GeneratedBy"][0]["Name"] == "nobrainer"

    def test_labels_land_under_derivatives_never_raw_dseg(
        self, bids_tree: Path, tmp_path: Path
    ) -> None:
        result = scan_bids(bids_tree, backend="walker")
        out = tmp_path / "reshaped"
        to_bids(result.entries, out)
        assert not list(out.glob("sub-*/anat/*_dseg.*"))
        assert list(out.glob("derivatives/nobrainer/sub-*/anat/*_dseg.*"))

    def test_dseg_tsv_has_index_and_name_columns(
        self, bids_tree: Path, tmp_path: Path
    ) -> None:
        result = scan_bids(bids_tree, backend="walker")
        out = tmp_path / "reshaped"
        to_bids(result.entries, out, label_names=["background", "brain"])
        tsv_text = (out / "derivatives/nobrainer/dseg.tsv").read_text()
        header = tsv_text.splitlines()[0].split("\t")
        assert header == ["index", "name"]
        assert "background" in tsv_text
        assert "brain" in tsv_text

    def test_unsanitizable_stem_falls_back_to_sequential(self, tmp_path: Path) -> None:
        image = _make_nifti(tmp_path / "!!!.nii.gz")
        out = tmp_path / "reshaped"
        report = to_bids([{"image": str(image)}], out)
        assert report.n_written == 1
        assert report.subject_mapping[0].strategy == "sequential"
        assert report.subject_mapping[0].subject_label == "001"

    def test_garbage_entry_among_good_ones_is_skipped_others_convert(
        self, bids_tree: Path, tmp_path: Path
    ) -> None:
        result = scan_bids(bids_tree, backend="walker")
        entries = list(result.entries) + [{"image": "/nonexistent/path.nii.gz"}]
        out = tmp_path / "reshaped"
        report = to_bids(entries, out)
        assert report.n_written == 2
        assert any(s.reason == SkipReason.UNREADABLE_FILE for s in report.skipped)

    def test_case_folded_subject_collision_falls_back_to_sequential(
        self, tmp_path: Path
    ) -> None:
        img1 = _make_nifti(tmp_path / "src1/S1_T1w.nii.gz")
        img2 = _make_nifti(tmp_path / "src2/s1_T1w.nii.gz")
        out = tmp_path / "reshaped"
        report = to_bids([{"image": str(img1)}, {"image": str(img2)}], out)
        assert report.n_written == 2
        labels = [m.subject_label for m in report.subject_mapping]
        assert len({label.casefold() for label in labels}) == 2
        assert report.subject_mapping[1].strategy == "sequential"

    def test_nonempty_out_root_without_overwrite_raises(
        self, bids_tree: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "reshaped"
        out.mkdir()
        (out / "marker.txt").write_text("x")
        result = scan_bids(bids_tree, backend="walker")
        with pytest.raises(FileExistsError):
            to_bids(result.entries, out)
        # overwrite=True proceeds
        report = to_bids(result.entries, out, overwrite=True)
        assert report.n_written == 2

    def test_empty_entries_raises(self, tmp_path: Path) -> None:
        out = tmp_path / "reshaped"
        with pytest.raises(ValueError):
            to_bids([], out)


# ---------------------------------------------------------------------------
# CLI contract
# ---------------------------------------------------------------------------


def _help(cmd: list[str]) -> str:
    result = subprocess.run(
        [sys.executable, "-m", "nobrainer.cli.main"] + cmd + ["--help"],
        capture_output=True,
        text=True,
    )
    assert (
        result.returncode == 0
    ), f"'{' '.join(cmd)} --help' exited {result.returncode}:\n{result.stderr}"
    return result.stdout


class TestCLIContract:
    def test_data_from_bids_help_exits_zero(self) -> None:
        _help(["data", "from-bids"])

    def test_data_to_bids_help_exits_zero(self) -> None:
        _help(["data", "to-bids"])

    def test_from_bids_json_output_has_entries_and_skip_histogram(
        self, bids_tree: Path, tmp_path: Path
    ) -> None:
        out_manifest = tmp_path / "manifest.json"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "nobrainer.cli.main",
                "data",
                "from-bids",
                str(bids_tree),
                str(out_manifest),
                "--json",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        payload = json.loads(result.stdout)
        assert payload["n_entries"] == 2
        assert payload["skipped"]["no_matching_label"] == 1

    def test_to_bids_cli_end_to_end(self, bids_tree: Path, tmp_path: Path) -> None:
        manifest = tmp_path / "manifest.json"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "nobrainer.cli.main",
                "data",
                "from-bids",
                str(bids_tree),
                str(manifest),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        out_dir = tmp_path / "reshaped"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "nobrainer.cli.main",
                "data",
                "to-bids",
                str(manifest),
                str(out_dir),
                "--json",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        payload = json.loads(result.stdout)
        assert payload["n_written"] == 2


# ---------------------------------------------------------------------------
# Reused-not-duplicated helper
# ---------------------------------------------------------------------------


def test_extract_subject_id_mirrors_openneuro_helper(tmp_path: Path) -> None:
    from nobrainer.datasets.openneuro import _extract_subject_id as _openneuro_extract

    path = tmp_path / "sub-07" / "anat" / "sub-07_T1w.nii.gz"
    assert _extract_subject_id(path) == _openneuro_extract(path)

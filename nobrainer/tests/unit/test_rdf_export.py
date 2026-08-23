"""Tests for nobrainer.provenance.rdf_export.

Placed under nobrainer/tests/unit/ (not a top-level tests/ dir, which does
not exist in this repo) to match pyproject.toml's
``testpaths = ["nobrainer/tests"]`` and the existing unit-test convention.
Run with: uv run pytest nobrainer/tests/unit/test_rdf_export.py -q
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest
from rdflib import RDF, Graph
from rdflib.namespace import PROV

from nobrainer.provenance import (
    ProvenanceError,
    build_graph,
    export_provenance,
    to_jsonld,
    to_turtle,
)
from nobrainer.provenance.rdf_export import (
    _assert_domain_only,
    canonical_json_bytes,
    digest,
    normalize_float,
)


def _write_bundle(
    tmp_path: Path,
    *,
    name: str = "run1",
    source_datasets: list[dict] | None = None,
    model_sha256: str = "c4f0deadbeef",
    training_date: str = "2026-02-11T18:22:04.113295+00:00",
    nobrainer_version: str = "2.0.0a17.dev6+gb85a1ca5b",
    pytorch_version: str = "2.9.0+cu128",
    model_architecture: str = "unet",
    model_args: dict | None = None,
    n_classes: int | None = 2,
    block_shape: list[int] | None = None,
    final_loss: float | None = 0.1873,
    best_loss: float | None = 0.1712,
    write_weights: bool = True,
) -> Path:
    """Write a minimal Segmentation.save()-shaped bundle directory."""
    bundle_dir = tmp_path / name
    bundle_dir.mkdir(parents=True, exist_ok=True)
    if source_datasets is None:
        source_datasets = [
            {"path": "/data/sub-01_T1w.nii.gz", "sha256": "3a7bd3e2360a3d29"}
        ]
    if model_args is None:
        model_args = {"channels": [4, 8], "strides": [2]}
    if block_shape is None:
        block_shape = [16, 16, 16]

    doc = {
        "@context": {"@vocab": "https://schema.org/"},
        "@type": "sc:Dataset",
        "conformsTo": "http://mlcommons.org/croissant/1.0",
        "name": f"nobrainer-{model_architecture}",
        "description": f"Trained {model_architecture} model via nobrainer",
        "distribution": [
            {
                "@type": "cr:FileObject",
                "name": "model.pth",
                "contentUrl": "model.pth",
                "encodingFormat": "application/x-pytorch",
                "sha256": model_sha256,
            }
        ],
        "nobrainer:provenance": {
            "source_datasets": source_datasets,
            "training_date": training_date,
            "nobrainer_version": nobrainer_version,
            "pytorch_version": pytorch_version,
            "optimizer": {"class": "Adam", "args": {"lr": "0.001"}},
            "loss_function": "CrossEntropyLoss",
            "epochs_trained": 12,
            "final_loss": final_loss,
            "best_loss": best_loss,
            "model_architecture": model_architecture,
            "model_args": model_args,
            "n_classes": n_classes,
            "block_shape": block_shape,
            "gpu_count": 1,
        },
    }
    (bundle_dir / "croissant.json").write_text(json.dumps(doc, indent=2))
    if write_weights:
        (bundle_dir / "model.pth").write_bytes(b"dummy-weights")
    return bundle_dir


def _write_dataspec(tmp_path: Path, image_path: Path, label_path: Path) -> Path:
    spec_path = tmp_path / "manifest.json"
    spec_path.write_text(
        json.dumps({"entries": [{"image": str(image_path), "label": str(label_path)}]})
    )
    return spec_path


# ---------------------------------------------------------------------------
# Hashing primitives
# ---------------------------------------------------------------------------


class TestHashingPrimitives:
    def test_canonical_json_is_stable_across_key_order(self) -> None:
        a = canonical_json_bytes({"b": 1, "a": 2})
        b = canonical_json_bytes({"a": 2, "b": 1})
        assert a == b

    def test_canonical_json_rejects_nan(self) -> None:
        with pytest.raises(ValueError):
            canonical_json_bytes({"x": float("nan")})

    def test_digest_domain_separation(self) -> None:
        payload = {"a": 1}
        assert digest("entity", payload) != digest("activity", payload)

    def test_digest_length_is_128_bits(self) -> None:
        assert len(digest("k", {"a": 1})) == 32

    def test_digest_is_deterministic(self) -> None:
        payload = {"a": 1, "b": [1, 2, 3]}
        assert digest("k", payload) == digest("k", payload)

    def test_normalize_float_passes_none(self) -> None:
        assert normalize_float(None) is None

    def test_normalize_float_rejects_nan(self) -> None:
        with pytest.raises(ProvenanceError):
            normalize_float(float("nan"))

    def test_normalize_float_rejects_inf(self) -> None:
        with pytest.raises(ProvenanceError):
            normalize_float(float("inf"))

    def test_normalize_float_returns_repr_string(self) -> None:
        assert normalize_float(0.5) == repr(0.5)


# ---------------------------------------------------------------------------
# build_graph: required shape (the /goal's four graph requirements)
# ---------------------------------------------------------------------------


class TestGraphShape:
    def test_run_is_prov_activity(self, tmp_path: Path) -> None:
        g = build_graph(_write_bundle(tmp_path))
        activities = list(g.subjects(RDF.type, PROV.Activity))
        assert len(activities) == 1

    def test_dataset_entity_is_used_by_run(self, tmp_path: Path) -> None:
        g = build_graph(_write_bundle(tmp_path))
        run = next(g.subjects(RDF.type, PROV.Activity))
        used = list(g.objects(run, PROV.used))
        assert len(used) == 1
        assert (used[0], RDF.type, PROV.Entity) in g

    def test_model_entity_was_generated_by_run(self, tmp_path: Path) -> None:
        g = build_graph(_write_bundle(tmp_path))
        run = next(g.subjects(RDF.type, PROV.Activity))
        generated = list(g.objects(run, PROV.generated))
        assert len(generated) == 1
        assert (generated[0], PROV.wasGeneratedBy, run) in g

    def test_agent_is_associated_with_run(self, tmp_path: Path) -> None:
        g = build_graph(_write_bundle(tmp_path))
        run = next(g.subjects(RDF.type, PROV.Activity))
        agents = list(g.objects(run, PROV.wasAssociatedWith))
        assert len(agents) >= 1
        assert (agents[0], RDF.type, PROV.Agent) in g

    def test_no_blank_nodes(self, tmp_path: Path) -> None:
        from rdflib import BNode

        g = build_graph(_write_bundle(tmp_path))
        for s, p, o in g:
            assert not isinstance(s, BNode)
            assert not isinstance(o, BNode)

    def test_model_sha256_is_recorded(self, tmp_path: Path) -> None:
        g = build_graph(_write_bundle(tmp_path, model_sha256="c4f0deadbeef"))
        run = next(g.subjects(RDF.type, PROV.Activity))
        model = next(g.objects(run, PROV.generated))
        # The model IRI itself is content-addressed under sha256/.
        assert "sha256/c4f0deadbeef" in str(model)


# ---------------------------------------------------------------------------
# --agent
# ---------------------------------------------------------------------------


class TestAgentOption:
    def test_no_agent_still_has_software_agents(self, tmp_path: Path) -> None:
        g = build_graph(_write_bundle(tmp_path), agent=None)
        run = next(g.subjects(RDF.type, PROV.Activity))
        agents = list(g.objects(run, PROV.wasAssociatedWith))
        assert len(agents) >= 1

    def test_human_agent_text_adds_person(self, tmp_path: Path) -> None:
        g = build_graph(_write_bundle(tmp_path), agent="Jane Doe <jane@lab.org>")
        run = next(g.subjects(RDF.type, PROV.Activity))
        agents = list(g.objects(run, PROV.wasAssociatedWith))
        persons = [a for a in agents if (a, RDF.type, PROV.Person) in g]
        assert len(persons) == 1

    def test_agent_email_is_extracted(self, tmp_path: Path) -> None:
        from rdflib.namespace import Namespace

        schema = Namespace("https://schema.org/")
        g = build_graph(_write_bundle(tmp_path), agent="Jane Doe <jane@lab.org>")
        run = next(g.subjects(RDF.type, PROV.Activity))
        persons = [
            a
            for a in g.objects(run, PROV.wasAssociatedWith)
            if (a, RDF.type, PROV.Person) in g
        ]
        emails = list(g.objects(persons[0], schema.email))
        assert str(emails[0]) == "jane@lab.org"

    def test_iri_agent_used_directly(self, tmp_path: Path) -> None:
        from rdflib import URIRef

        g = build_graph(
            _write_bundle(tmp_path), agent="https://orcid.org/0000-0000-0000-0000"
        )
        run = next(g.subjects(RDF.type, PROV.Activity))
        agents = list(g.objects(run, PROV.wasAssociatedWith))
        assert URIRef("https://orcid.org/0000-0000-0000-0000") in agents

    def test_agent_is_not_invented_when_absent(self, tmp_path: Path) -> None:
        g = build_graph(_write_bundle(tmp_path), agent=None)
        assert list(g.subjects(RDF.type, PROV.Person)) == []


# ---------------------------------------------------------------------------
# --dataspec
# ---------------------------------------------------------------------------


class TestDataspecOption:
    def test_dataspec_produces_image_and_label_members(self, tmp_path: Path) -> None:
        img = tmp_path / "sub-01_T1w.nii.gz"
        lbl = tmp_path / "sub-01_aseg.nii.gz"
        img.write_bytes(b"image-bytes")
        lbl.write_bytes(b"label-bytes")
        bundle = _write_bundle(tmp_path)
        spec_path = _write_dataspec(tmp_path, img, lbl)

        g = build_graph(bundle, dataspec_path=spec_path)
        run = next(g.subjects(RDF.type, PROV.Activity))
        dataset = next(g.objects(run, PROV.used))
        members = list(g.objects(dataset, PROV.hadMember))
        assert len(members) == 2

    def test_without_dataspec_falls_back_to_croissant_source_datasets(
        self, tmp_path: Path
    ) -> None:
        g = build_graph(_write_bundle(tmp_path), dataspec_path=None)
        run = next(g.subjects(RDF.type, PROV.Activity))
        dataset = next(g.objects(run, PROV.used))
        members = list(g.objects(dataset, PROV.hadMember))
        assert len(members) == 1


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_block_shape_emits_no_triple(self, tmp_path: Path) -> None:
        from rdflib.namespace import Namespace

        nb = Namespace("https://neuronets.dev/ns/nobrainer#")
        g = build_graph(_write_bundle(tmp_path, block_shape=[]))
        run = next(g.subjects(RDF.type, PROV.Activity))
        assert (run, nb.blockShape, None) not in g

    def test_none_n_classes_emits_no_triple(self, tmp_path: Path) -> None:
        from rdflib.namespace import Namespace

        nb = Namespace("https://neuronets.dev/ns/nobrainer#")
        g = build_graph(_write_bundle(tmp_path, n_classes=None))
        run = next(g.subjects(RDF.type, PROV.Activity))
        assert (run, nb.numberOfClasses, None) not in g

    def test_missing_sha256_falls_back_to_recompute(self, tmp_path: Path) -> None:
        from rdflib.namespace import Namespace

        nb = Namespace("https://neuronets.dev/ns/nobrainer#")
        bundle = _write_bundle(tmp_path, model_sha256="", write_weights=True)
        g = build_graph(bundle)
        run = next(g.subjects(RDF.type, PROV.Activity))
        model = next(g.objects(run, PROV.generated))
        status = list(g.objects(model, nb.checksumStatus))
        assert str(status[0]) == "recomputed"

    def test_missing_sha256_and_no_file_is_unavailable(self, tmp_path: Path) -> None:
        from rdflib.namespace import Namespace

        nb = Namespace("https://neuronets.dev/ns/nobrainer#")
        bundle = _write_bundle(tmp_path, model_sha256="", write_weights=False)
        g = build_graph(bundle)
        run = next(g.subjects(RDF.type, PROV.Activity))
        model = next(g.objects(run, PROV.generated))
        status = list(g.objects(model, nb.checksumStatus))
        assert str(status[0]) == "unavailable"

    def test_strict_raises_on_unavailable_checksum(self, tmp_path: Path) -> None:
        bundle = _write_bundle(tmp_path, model_sha256="", write_weights=False)
        with pytest.raises(ProvenanceError):
            build_graph(bundle, strict=True)

    def test_non_finite_loss_is_omitted_not_raised(self, tmp_path: Path) -> None:
        from rdflib.namespace import Namespace

        nb = Namespace("https://neuronets.dev/ns/nobrainer#")
        bundle = _write_bundle(tmp_path, final_loss=float("nan"))
        g = build_graph(bundle, strict=False)
        run = next(g.subjects(RDF.type, PROV.Activity))
        assert (run, nb.finalLoss, None) not in g
        assert (run, nb.lossStatus, None) in g

    def test_non_finite_loss_raises_under_strict(self, tmp_path: Path) -> None:
        bundle = _write_bundle(tmp_path, final_loss=float("nan"))
        with pytest.raises(ProvenanceError):
            build_graph(bundle, strict=True)

    def test_memory_address_poisoned_hyperparameter_is_elided(
        self, tmp_path: Path
    ) -> None:
        from rdflib.namespace import Namespace

        nb = Namespace("https://neuronets.dev/ns/nobrainer#")
        bundle = _write_bundle(
            tmp_path, model_args={"callback": "<function foo at 0x10a3b2c00>"}
        )
        g = build_graph(bundle, strict=False)
        elided = list(g.subjects(nb.hyperparameterElided, None))
        assert len(elided) == 1

    def test_poisoned_hyperparameter_raises_under_strict(self, tmp_path: Path) -> None:
        bundle = _write_bundle(
            tmp_path, model_args={"callback": "<function foo at 0x10a3b2c00>"}
        )
        with pytest.raises(ProvenanceError):
            build_graph(bundle, strict=True)

    def test_missing_croissant_json_raises(self, tmp_path: Path) -> None:
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with pytest.raises(ProvenanceError):
            build_graph(empty_dir)

    def test_dataset_flavor_croissant_is_rejected(self, tmp_path: Path) -> None:
        bundle_dir = tmp_path / "dsflavor"
        bundle_dir.mkdir()
        (bundle_dir / "croissant.json").write_text(
            json.dumps(
                {
                    "@type": "sc:Dataset",
                    "nobrainer:dataset_info": {"n_volumes": 3},
                }
            )
        )
        with pytest.raises(ProvenanceError):
            build_graph(bundle_dir)

    def test_checkpoint_flavor_architecture_is_not_tagged_registry(
        self, tmp_path: Path
    ) -> None:
        """write_checkpoint_croissant's subset omits source_datasets/model_args/
        n_classes/block_shape and puts a torch class name in model_architecture."""
        from rdflib.namespace import Namespace

        nb = Namespace("https://neuronets.dev/ns/nobrainer#")
        bundle_dir = tmp_path / "checkpoint_flavor"
        bundle_dir.mkdir()
        doc = {
            "name": "nobrainer-MeshNet",
            "description": "Trained MeshNet checkpoint via nobrainer",
            "distribution": [
                {
                    "name": "best_model.pth",
                    "contentUrl": "best_model.pth",
                    "sha256": "aabbcc",
                }
            ],
            "nobrainer:provenance": {
                "training_date": "2026-01-01T00:00:00+00:00",
                "nobrainer_version": "2.0.0a17.dev6+gb85a1ca5b",
                "pytorch_version": "2.9.0",
                "optimizer": {"class": "Adam", "args": {}},
                "loss_function": "CrossEntropyLoss",
                "epochs_trained": 5,
                "final_loss": 0.2,
                "best_loss": 0.2,
                "model_architecture": "MeshNet",
                "gpu_count": 0,
            },
        }
        (bundle_dir / "croissant.json").write_text(json.dumps(doc))
        g = build_graph(bundle_dir)
        run = next(g.subjects(RDF.type, PROV.Activity))
        vocab = list(g.objects(run, nb.architectureVocabulary))
        assert str(vocab[0]) == "torch-class-name"


# ---------------------------------------------------------------------------
# The DOMAIN-only boundary
# ---------------------------------------------------------------------------


class TestDomainOnlyBoundary:
    def test_default_export_passes_boundary_check(self, tmp_path: Path) -> None:
        g = build_graph(_write_bundle(tmp_path))
        _assert_domain_only(g)  # must not raise

    def test_positive_control_second_activity_is_rejected(self, tmp_path: Path) -> None:
        """Without this test, the boundary checks above could pass vacuously."""
        from rdflib import RDF, URIRef

        g = build_graph(_write_bundle(tmp_path))
        g.add(
            (URIRef("https://example.org/ingestion-activity"), RDF.type, PROV.Activity)
        )
        with pytest.raises(ProvenanceError):
            _assert_domain_only(g)

    def test_no_brainkb_term_anywhere(self, tmp_path: Path) -> None:
        g = build_graph(_write_bundle(tmp_path))
        for s, p, o in g:
            assert "brainkb" not in str(s).lower()
            assert "brainkb" not in str(p).lower()
            assert "brainkb" not in str(o).lower()

    def test_module_docstring_states_brainkb_is_separate_step(self) -> None:
        from nobrainer.provenance import rdf_export

        doc = (rdf_export.__doc__ or "").lower()
        assert "brainkb" in doc
        assert "separate" in doc


# ---------------------------------------------------------------------------
# Serialization: turtle/json-ld equivalence (required by the plan)
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_turtle_reparses(self, tmp_path: Path) -> None:
        g = build_graph(_write_bundle(tmp_path))
        ttl = to_turtle(g)
        reparsed = Graph().parse(data=ttl, format="turtle")
        assert len(reparsed) == len(g)

    def test_jsonld_reparses(self, tmp_path: Path) -> None:
        g = build_graph(_write_bundle(tmp_path))
        jsonld = to_jsonld(g)
        reparsed = Graph().parse(data=jsonld, format="json-ld")
        assert len(reparsed) == len(g)

    def test_jsonld_is_strict_valid_json(self, tmp_path: Path) -> None:
        g = build_graph(_write_bundle(tmp_path))
        text = to_jsonld(g)

        def _boom(x):
            raise ValueError(f"non-JSON constant: {x}")

        json.loads(text, parse_constant=_boom)  # must not raise

    def test_turtle_and_jsonld_have_same_triple_count(self, tmp_path: Path) -> None:
        bundle = _write_bundle(tmp_path)
        ttl = export_provenance(bundle, fmt="turtle")
        jsonld = export_provenance(bundle, fmt="json-ld")
        g_ttl = Graph().parse(data=ttl, format="turtle")
        g_jsonld = Graph().parse(data=jsonld, format="json-ld")
        assert len(g_ttl) == len(g_jsonld)

    def test_export_provenance_rejects_bad_format(self, tmp_path: Path) -> None:
        bundle = _write_bundle(tmp_path)
        with pytest.raises(ValueError):
            export_provenance(bundle, fmt="xml")


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_repeated_export_is_byte_identical(self, tmp_path: Path) -> None:
        bundle = _write_bundle(tmp_path)
        a = export_provenance(bundle, fmt="turtle")
        b = export_provenance(bundle, fmt="turtle")
        assert a == b

    def test_run_iri_unchanged_after_directory_move(self, tmp_path: Path) -> None:
        import shutil

        bundle = _write_bundle(tmp_path, name="orig")
        moved = tmp_path / "moved"
        shutil.copytree(bundle, moved)

        g1 = build_graph(bundle)
        g2 = build_graph(moved)
        run1 = next(g1.subjects(RDF.type, PROV.Activity))
        run2 = next(g2.subjects(RDF.type, PROV.Activity))
        assert run1 == run2

    def test_run_iri_changes_when_epochs_trained_changes(self, tmp_path: Path) -> None:
        """Negative control: without this, the identity payload could be constant."""
        bundle_a = _write_bundle(tmp_path, name="a")
        bundle_b = tmp_path / "b"
        bundle_b.mkdir()
        doc = json.loads((bundle_a / "croissant.json").read_text())
        doc["nobrainer:provenance"]["epochs_trained"] = 999
        (bundle_b / "croissant.json").write_text(json.dumps(doc))
        (bundle_b / "model.pth").write_bytes(b"dummy-weights")

        g_a = build_graph(bundle_a)
        g_b = build_graph(bundle_b)
        run_a = next(g_a.subjects(RDF.type, PROV.Activity))
        run_b = next(g_b.subjects(RDF.type, PROV.Activity))
        assert run_a != run_b

    def test_base_iri_changes_instance_iris(self, tmp_path: Path) -> None:
        bundle = _write_bundle(tmp_path)
        g1 = build_graph(bundle, base_iri="https://neuronets.dev/nobrainer/")
        g2 = build_graph(bundle, base_iri="https://example.org/nb/")
        run1 = next(g1.subjects(RDF.type, PROV.Activity))
        run2 = next(g2.subjects(RDF.type, PROV.Activity))
        assert str(run1).startswith("https://neuronets.dev/nobrainer/")
        assert str(run2).startswith("https://example.org/nb/")


# ---------------------------------------------------------------------------
# No-torch-import constraint
# ---------------------------------------------------------------------------


class TestImportIsolation:
    def test_provenance_module_imports_without_torch_being_required(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys\n"
                "import nobrainer.provenance\n"
                "assert 'torch' not in sys.modules, "
                "'nobrainer.provenance must not import torch'\n"
                "print('OK')",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "OK" in result.stdout


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestCli:
    def test_export_help_exits_zero(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "nobrainer.cli.main",
                "provenance",
                "export",
                "--help",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr

    def test_export_turtle_to_stdout(self, tmp_path: Path) -> None:
        bundle = _write_bundle(tmp_path)
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "nobrainer.cli.main",
                "provenance",
                "export",
                "--bundle",
                str(bundle),
                "--format",
                "turtle",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        g = Graph().parse(data=result.stdout, format="turtle")
        assert len(list(g.subjects(RDF.type, PROV.Activity))) == 1

    def test_export_writes_to_out_file(self, tmp_path: Path) -> None:
        bundle = _write_bundle(tmp_path)
        out_path = tmp_path / "out.ttl"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "nobrainer.cli.main",
                "provenance",
                "export",
                "--bundle",
                str(bundle),
                "--format",
                "turtle",
                "--out",
                str(out_path),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert out_path.exists()
        Graph().parse(str(out_path), format="turtle")  # must not raise

    def test_export_missing_bundle_dir_fails_cleanly(self, tmp_path: Path) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "nobrainer.cli.main",
                "provenance",
                "export",
                "--bundle",
                str(tmp_path / "does-not-exist"),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0

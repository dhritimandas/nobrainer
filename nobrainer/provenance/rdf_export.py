"""PROV-O RDF provenance export for nobrainer training runs.

Reads a saved model bundle (a ``Segmentation.save()`` directory: ``model.pth``
+ ``croissant.json``) and, optionally, a ``DataSpec`` dataset manifest
(:mod:`nobrainer.data.spec`), and emits a PROV-O graph describing the
training run as a ``prov:Activity`` that ``prov:used`` a dataset entity and
``prov:generated`` a model entity, both ``prov:wasAssociatedWith`` one or
more agents.

Scope: DOMAIN provenance only
------------------------------
This module emits facts about what nobrainer itself did -- the run, its
inputs, its outputs, its agents. It does **not** emit BrainKB
ingestion-activity triples (for example, an activity describing "this
graph was loaded into BrainKB at time T"). Named-graph registration and
POSTing this module's output to BrainKB's ingestion API are separate steps
performed by the caller against BrainKB's own API, entirely outside this
module's scope. ``_assert_domain_only`` enforces this boundary on every
call to :func:`build_graph`.

Determinism
-----------
Every IRI minted by this module is derived from content -- a sha256 digest
of a file, or a stable hash of a canonical-JSON payload -- never from a
timestamp, a random UUID, or an absolute filesystem path. Re-exporting the
same ``--bundle``/``--dataspec`` input from a different location on disk
produces a byte-identical graph. There are no blank nodes anywhere in the
emitted graph.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any

from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import DCTERMS, PROV, RDF, RDFS, XSD

import nobrainer
from nobrainer.data.spec import DataSpec, FileStatus, check_file_presence

__all__ = [
    "ProvenanceError",
    "build_graph",
    "export_provenance",
    "to_jsonld",
    "to_turtle",
]

SCHEMA = Namespace("https://schema.org/")

DEFAULT_BASE_IRI = "https://neuronets.dev/nobrainer/"
DEFAULT_VOCAB_IRI = "https://neuronets.dev/ns/nobrainer#"

# NOTE (needs owner sign-off before this is used as a permanent identifier):
# neuronets.dev currently serves the nobrainer book, not a term dereferencer.
# Both constants above are overridable via --base-iri specifically so this
# is a one-line change, not a re-mint of every IRI this module has produced.

_MEMORY_ADDRESS_RE = re.compile(r"<[^>]* at 0x[0-9a-fA-F]+>")
_GIT_SHA_RE = re.compile(r"\+.*?g([0-9a-fA-F]{7,40})")
_EMAIL_RE = re.compile(r"[^<\s]+@[^>\s]+")

# The PROV-O predicates/types this module allows itself to emit. Adding a
# predicate here is the reviewable seam that keeps the DOMAIN-only boundary
# from drifting silently -- see _assert_domain_only.
_FORBIDDEN_PREDICATES = frozenset(
    {
        PROV.wasInformedBy,
        PROV.wasStartedBy,
        PROV.wasEndedBy,
        PROV.qualifiedAssociation,
        PROV.hadPlan,
    }
)


class ProvenanceError(RuntimeError):
    """Raised when a bundle or dataspec cannot be turned into a provenance graph."""


# ---------------------------------------------------------------------------
# Hashing / canonicalization -- the basis of every minted IRI
# ---------------------------------------------------------------------------


def canonical_json_bytes(obj: Any) -> bytes:
    """Serialize an object to canonical, sorted, compact UTF-8 JSON bytes.

    Parameters
    ----------
    obj : Any
        A JSON-serializable object (dict, list, str, int, float, bool, None).

    Returns
    -------
    bytes
        UTF-8 encoded JSON with sorted keys and no incidental whitespace, so
        that the same logical payload always produces the same bytes
        regardless of dict insertion order.

    Raises
    ------
    ValueError
        If ``obj`` contains a non-finite float (``allow_nan=False``): a NaN
        or Infinity must never silently enter a content hash.
    """
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def digest(kind: str, payload: Any, length: int = 32) -> str:
    """Domain-separated, content-derived hex digest of a JSON payload.

    Parameters
    ----------
    kind : str
        A short tag identifying what is being hashed (e.g. ``"run"``,
        ``"dataset"``). Mixed into the hash ahead of an ASCII Unit
        Separator byte (``\\x1f``, which cannot occur in the JSON output),
        so a run and a dataset built from identical payloads never collide.
    payload : Any
        JSON-serializable payload to hash.
    length : int, optional
        Number of hex characters to keep from the SHA-256 digest. The
        default, 32 hex characters (128 bits), is birthday-safe for any
        plausible corpus size and should not be lowered.

    Returns
    -------
    str
        A stable, deterministic hex digest.
    """
    h = hashlib.sha256()
    h.update(kind.encode("utf-8"))
    h.update(b"\x1f")
    h.update(canonical_json_bytes(payload))
    return h.hexdigest()[:length]


def sha256_file(path: str | Path) -> str:
    """Stream a file in 64 KiB chunks and return its hex SHA-256 digest.

    Parameters
    ----------
    path : str or Path
        Path to an existing, readable file.

    Returns
    -------
    str
        Lowercase hex SHA-256 digest.
    """
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_float(value: float | None) -> str | None:
    """Convert a float to a stable string form for hashing, rejecting non-finite.

    Parameters
    ----------
    value : float or None
        A value read from run metadata (e.g. a training loss).

    Returns
    -------
    str or None
        ``None`` if ``value`` is ``None``; otherwise ``repr(float(value))``,
        a string, so the exact decimal representation is pinned independent
        of any future change to Python's own float-repr algorithm.

    Raises
    ------
    ProvenanceError
        If ``value`` is NaN or +/-Infinity. rdflib serializes a non-finite
        float literal to bare ``NaN``/``Infinity`` in JSON-LD, which is not
        valid JSON (verified) -- such a value must never reach a hash or a
        literal, silently or otherwise.
    """
    if value is None:
        return None
    fv = float(value)
    if fv != fv or fv in (float("inf"), float("-inf")):
        raise ProvenanceError(f"non-finite float cannot be normalized: {value!r}")
    return repr(fv)


def _elide_memory_addresses(value: Any) -> tuple[Any, bool]:
    """Detect and elide a Python ``repr(obj)`` memory-address string.

    ``write_model_croissant`` serializes with ``json.dumps(..., default=str)``,
    so a value without a stable ``__str__`` (a callable, an unpickleable
    object) can land in ``model_args``/``optimizer.args`` as
    ``"<function foo at 0x10a3b2c00>"`` -- a string that differs on every
    process. If such a value entered a content hash, run IRIs would stop
    being deterministic across machines.

    Parameters
    ----------
    value : Any
        A value read from croissant.json's hyperparameter dicts.

    Returns
    -------
    tuple[Any, bool]
        ``(value, False)`` unchanged, or ``(None, True)`` if ``value`` (or
        any string it contains) matched the memory-address pattern.
    """
    if isinstance(value, str) and _MEMORY_ADDRESS_RE.search(value):
        return None, True
    return value, False


def _parse_git_sha(version: str) -> str | None:
    """Extract a short git commit SHA from a hatch-vcs local version segment.

    Parameters
    ----------
    version : str
        A version string such as ``"2.0.0a17.dev6+gb85a1ca5b"``.

    Returns
    -------
    str or None
        The commit SHA (e.g. ``"b85a1ca5b"``), or ``None`` if the version
        string has no ``+g<sha>`` local segment.
    """
    match = _GIT_SHA_RE.search(version)
    return match.group(1) if match else None


# ---------------------------------------------------------------------------
# IRI minting -- every subject in the emitted graph is one of these
# ---------------------------------------------------------------------------


def _content_iri(
    base_iri: str, kind: str, sha256: str | None, nosha_payload: dict
) -> URIRef:
    """Mint a content-addressed IRI, or a visibly-unaddressed fallback.

    Parameters
    ----------
    base_iri : str
        The export's base IRI (trailing slash expected).
    kind : str
        ``"model"`` or ``"file"`` -- the path segment used for both the
        addressed and unaddressed forms.
    sha256 : str or None
        The file's SHA-256 digest, if known.
    nosha_payload : dict
        A JSON payload identifying the entity when no checksum is
        available (e.g. its recorded path). Hashed under ``"{kind}-nosha"``
        so it cannot collide with a real checksum.

    Returns
    -------
    URIRef
        ``{base}{kind}/sha256/{sha256}`` if a checksum is known, else
        ``{base}{kind}/x-nosha/{digest}`` -- the ``x-nosha`` segment is
        deliberately visible so any consumer can tell at a glance that the
        entity is not content-addressed.
    """
    if sha256:
        return URIRef(f"{base_iri}{kind}/sha256/{sha256}")
    return URIRef(f"{base_iri}{kind}/x-nosha/{digest(f'{kind}-nosha', nosha_payload)}")


def _dataset_iri(base_iri: str, member_iris: list[URIRef]) -> URIRef:
    """Mint a dataset-collection IRI from the sorted set of its member IRIs."""
    payload = {"members": sorted(str(m) for m in member_iris)}
    return URIRef(f"{base_iri}dataset/{digest('dataset', payload)}")


def _run_iri(base_iri: str, payload: dict) -> URIRef:
    """Mint a training-run IRI from its canonical identity payload."""
    return URIRef(f"{base_iri}run/{digest('run', payload)}")


def _software_agent_iri(base_iri: str, name: str, version: str) -> URIRef:
    """Mint a software-agent IRI from its name and version."""
    payload = {"name": name, "version": version}
    return URIRef(f"{base_iri}agent/software/{digest('agent-software', payload)}")


def _human_agent_iri(base_iri: str, agent_text: str) -> URIRef:
    """Mint a caller-supplied agent IRI from the raw --agent text."""
    return URIRef(
        f"{base_iri}agent/caller/{digest('agent-caller', {'text': agent_text})}"
    )


def _hparam_iri(
    base_iri: str, run_iri: URIRef, scope: str, name: str, value_payload: Any
) -> URIRef:
    """Mint a hyperparameter-node IRI scoped to its owning run."""
    payload = {"scope": scope, "name": name, "value": value_payload}
    return URIRef(f"{run_iri}/hparam/{scope}/{digest('hparam', payload, length=16)}")


# ---------------------------------------------------------------------------
# Reading the bundle / dataspec inputs
# ---------------------------------------------------------------------------


def _read_bundle(bundle_dir: Path) -> dict[str, Any]:
    """Read and validate a ``Segmentation.save()`` directory's croissant.json.

    Reads ``croissant.json`` with plain :func:`json.load` -- deliberately
    **not** via :mod:`nobrainer.processing.croissant`, because importing
    that module executes ``nobrainer/processing/__init__.py``, which pulls
    in torch (verified). This module stays stdlib + rdflib only.

    Parameters
    ----------
    bundle_dir : Path
        Directory expected to contain ``croissant.json`` (and normally
        ``model.pth``, though this function does not require the weights
        file to exist).

    Returns
    -------
    dict[str, Any]
        The full parsed croissant.json document.

    Raises
    ------
    ProvenanceError
        If ``croissant.json`` is missing, malformed, or describes a
        dataset (``nobrainer:dataset_info``) rather than a training run
        (``nobrainer:provenance``).
    """
    croissant_path = bundle_dir / "croissant.json"
    if not croissant_path.is_file():
        raise ProvenanceError(f"No croissant.json found under {bundle_dir}")
    try:
        doc = json.loads(croissant_path.read_text())
    except json.JSONDecodeError as exc:
        raise ProvenanceError(
            f"Malformed croissant.json at {croissant_path}: {exc}"
        ) from exc

    if "nobrainer:provenance" not in doc:
        if "nobrainer:dataset_info" in doc:
            raise ProvenanceError(
                f"{croissant_path} describes a dataset (nobrainer:dataset_info), "
                "not a training run. --bundle must point at a Segmentation.save() "
                "directory, whose croissant.json carries nobrainer:provenance."
            )
        raise ProvenanceError(f"{croissant_path} has no 'nobrainer:provenance' key.")
    return doc


def _model_distribution(doc: dict[str, Any]) -> dict[str, Any]:
    """Return the first distribution entry (the model weights file)."""
    dist = doc.get("distribution") or []
    if not dist:
        raise ProvenanceError("croissant.json has an empty 'distribution' list.")
    return dist[0]


def _resolve_model_sha256(
    bundle_dir: Path, distribution: dict[str, Any]
) -> tuple[str | None, str]:
    """Resolve a model checksum via the recorded-value / recompute / unavailable ladder.

    Parameters
    ----------
    bundle_dir : Path
        The bundle directory (used to locate the weights file for
        recomputation).
    distribution : dict[str, Any]
        The ``distribution[0]`` entry from croissant.json.

    Returns
    -------
    tuple[str or None, str]
        ``(sha256, status)`` where ``status`` is one of ``"present"``
        (recorded in croissant.json), ``"recomputed"`` (recorded value was
        empty but the file exists on disk), or ``"unavailable"`` (neither).
    """
    recorded = distribution.get("sha256") or ""
    if recorded:
        return recorded, "present"
    content_url = distribution.get("contentUrl", "model.pth")
    candidate = bundle_dir / content_url
    if candidate.is_file():
        return sha256_file(candidate), "recomputed"
    return None, "unavailable"


def _read_dataspec(path: Path) -> DataSpec:
    """Load a DataSpec manifest, raising ProvenanceError on failure.

    Parameters
    ----------
    path : Path
        Path to a DataSpec JSON manifest (:meth:`DataSpec.to_json` format).

    Returns
    -------
    DataSpec
        The parsed dataset specification.

    Raises
    ------
    ProvenanceError
        If the file is missing or malformed.
    """
    try:
        return DataSpec.from_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        raise ProvenanceError(
            f"Could not read DataSpec manifest {path}: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def build_graph(
    bundle_dir: str | Path,
    dataspec_path: str | Path | None = None,
    base_iri: str = DEFAULT_BASE_IRI,
    agent: str | None = None,
    strict: bool = False,
) -> Graph:
    """Build a DOMAIN-only PROV-O graph for one saved nobrainer training run.

    Parameters
    ----------
    bundle_dir : str or Path
        A ``Segmentation.save()`` directory (``model.pth`` + ``croissant.json``).
    dataspec_path : str, Path, or None, optional
        Optional ``DataSpec`` manifest JSON. When given, dataset provenance
        is built from its ``entries`` (both image and label paths, each
        checksummed here) instead of croissant's ``source_datasets``, which
        only ever records image paths. When omitted, falls back to
        croissant's ``source_datasets``.
    base_iri : str, optional
        Base IRI for minted instance identifiers. Default
        ``"https://neuronets.dev/nobrainer/"``.
    agent : str or None, optional
        An optional caller-supplied human or organization identity (a free
        text ``"Name <email>"`` string, or an IRI). If given, adds a
        ``prov:Person`` (free text) or a plain ``prov:Agent`` (an IRI --
        this module cannot tell whether an opaque IRI denotes a person or
        an organization) ``prov:wasAssociatedWith`` the run, **in addition
        to** the always-emitted software agents. This is caller-asserted,
        never inferred: nothing in the underlying data identifies a human,
        so this module never invents one on its own.
    strict : bool, optional
        If ``True``, raise :class:`ProvenanceError` on conditions that
        would otherwise be silently elided (an unavailable checksum, a
        non-finite loss value, a memory-address-poisoned hyperparameter).
        Default ``False``.

    Returns
    -------
    rdflib.Graph
        A graph containing exactly one ``prov:Activity`` (the run), a
        ``prov:Entity`` for the model it ``prov:generated``, a
        ``prov:Entity``/``prov:Collection`` for the dataset it
        ``prov:used`` (omitted if there are zero dataset members), and one
        or more ``prov:Agent`` nodes it is ``prov:wasAssociatedWith``. No
        blank nodes. Verified DOMAIN-only via ``_assert_domain_only``
        before being returned.
    """
    bundle_dir = Path(bundle_dir)
    doc = _read_bundle(bundle_dir)
    prov_meta = doc["nobrainer:provenance"]
    distribution = _model_distribution(doc)

    g = Graph()
    g.bind("prov", PROV)
    g.bind("dcterms", DCTERMS)
    g.bind("rdfs", RDFS)
    g.bind("schema", SCHEMA)
    g.bind("xsd", XSD)
    nb = Namespace(DEFAULT_VOCAB_IRI)
    g.bind("nb", nb)

    # -- Dataset members -----------------------------------------------------
    dataset_members: list[URIRef] = []
    checksum_coverage = "none"
    if dataspec_path is not None:
        spec = _read_dataspec(Path(dataspec_path))
        checksum_coverage = "images-and-labels"
        for entry in spec.entries:
            for key in ("image", "label"):
                if key not in entry:
                    continue
                file_path = entry[key]
                status = check_file_presence(file_path)
                sha = sha256_file(file_path) if status == FileStatus.PRESENT else None
                fiu = _content_iri(base_iri, "file", sha, {"path": file_path})
                g.add((fiu, RDF.type, PROV.Entity))
                g.add((fiu, RDF.type, nb.SourceFile))
                g.add((fiu, nb.sourcePath, Literal(file_path)))
                g.add((fiu, nb.sourceRole, Literal(key)))
                if sha:
                    g.add((fiu, nb.sha256, Literal(sha, datatype=XSD.hexBinary)))
                    g.add((fiu, nb.checksumStatus, Literal("present")))
                else:
                    g.add((fiu, nb.checksumStatus, Literal("unavailable")))
                    if strict:
                        raise ProvenanceError(
                            f"Cannot checksum {key} file: {file_path}"
                        )
                dataset_members.append(fiu)
    else:
        source_datasets = prov_meta.get("source_datasets") or []
        if source_datasets:
            checksum_coverage = "images-only"
        for item in source_datasets:
            path = item.get("path", "")
            sha = item.get("sha256") or None
            fiu = _content_iri(base_iri, "file", sha, {"path": path})
            g.add((fiu, RDF.type, PROV.Entity))
            g.add((fiu, RDF.type, nb.SourceFile))
            g.add((fiu, nb.sourcePath, Literal(path)))
            g.add((fiu, nb.sourceRole, Literal("image")))
            if sha:
                g.add((fiu, nb.sha256, Literal(sha, datatype=XSD.hexBinary)))
                g.add((fiu, nb.checksumStatus, Literal("present")))
            else:
                g.add((fiu, nb.checksumStatus, Literal("unavailable")))
            dataset_members.append(fiu)

    dataset_iri: URIRef | None = None
    if dataset_members:
        dataset_iri = _dataset_iri(base_iri, dataset_members)
        g.add((dataset_iri, RDF.type, PROV.Entity))
        g.add((dataset_iri, RDF.type, PROV.Collection))
        g.add((dataset_iri, RDF.type, nb.TrainingDataset))
        g.add(
            (
                dataset_iri,
                nb.memberCount,
                Literal(len(dataset_members), datatype=XSD.nonNegativeInteger),
            )
        )
        g.add((dataset_iri, nb.checksumCoverage, Literal(checksum_coverage)))
        for m in dataset_members:
            g.add((dataset_iri, PROV.hadMember, m))

    # -- Model entity ---------------------------------------------------------
    model_sha, checksum_status = _resolve_model_sha256(bundle_dir, distribution)
    if checksum_status == "unavailable" and strict:
        raise ProvenanceError(
            "Model weights checksum is unavailable and --strict was set."
        )
    model_iri = _content_iri(
        base_iri,
        "model",
        model_sha,
        {"contentUrl": distribution.get("contentUrl", "model.pth")},
    )
    g.add((model_iri, RDF.type, PROV.Entity))
    g.add((model_iri, RDF.type, nb.TrainedModel))
    g.add((model_iri, SCHEMA.name, Literal(distribution.get("name", "model.pth"))))
    g.add(
        (
            model_iri,
            nb.relativePath,
            Literal(distribution.get("contentUrl", "model.pth")),
        )
    )
    if distribution.get("encodingFormat"):
        g.add(
            (model_iri, SCHEMA.encodingFormat, Literal(distribution["encodingFormat"]))
        )
    if model_sha:
        g.add((model_iri, nb.sha256, Literal(model_sha, datatype=XSD.hexBinary)))
    g.add((model_iri, nb.checksumStatus, Literal(checksum_status)))
    training_date = prov_meta.get("training_date")
    if training_date:
        g.add(
            (
                model_iri,
                PROV.generatedAtTime,
                Literal(training_date, datatype=XSD.dateTime),
            )
        )
        g.add(
            (model_iri, DCTERMS.created, Literal(training_date, datatype=XSD.dateTime))
        )

    # -- Architecture vocabulary discrimination -------------------------------
    architecture = prov_meta.get("model_architecture")
    is_model_flavor = any(
        prov_meta.get(k)
        for k in ("source_datasets", "model_args", "n_classes", "block_shape")
    )
    architecture_vocab = "unknown"
    if architecture:
        architecture_vocab = (
            "nobrainer-model-registry" if is_model_flavor else "torch-class-name"
        )

    # -- Hyperparameters -------------------------------------------------------
    hparam_iris: list[URIRef] = []

    def _add_hparams(
        run_iri_placeholder: URIRef, scope: str, values: dict[str, Any]
    ) -> None:
        for name, value in (values or {}).items():
            clean, elided = _elide_memory_addresses(value)
            if elided:
                if strict:
                    raise ProvenanceError(
                        f"Hyperparameter {scope}.{name} is memory-address-poisoned."
                    )
                hiu = _hparam_iri(base_iri, run_iri_placeholder, scope, name, None)
                g.add((hiu, RDF.type, nb.Hyperparameter))
                g.add((hiu, SCHEMA.name, Literal(name)))
                g.add((hiu, nb.hyperparameterScope, Literal(scope)))
                g.add(
                    (hiu, nb.hyperparameterElided, Literal(True, datatype=XSD.boolean))
                )
                hparam_iris.append(hiu)
                continue
            hiu = _hparam_iri(base_iri, run_iri_placeholder, scope, name, clean)
            g.add((hiu, RDF.type, nb.Hyperparameter))
            g.add((hiu, SCHEMA.name, Literal(name)))
            g.add((hiu, nb.hyperparameterScope, Literal(scope)))
            if isinstance(clean, bool):
                g.add((hiu, SCHEMA.value, Literal(clean, datatype=XSD.boolean)))
            elif isinstance(clean, int):
                g.add((hiu, SCHEMA.value, Literal(clean, datatype=XSD.integer)))
            elif isinstance(clean, float):
                g.add(
                    (
                        hiu,
                        SCHEMA.value,
                        Literal(normalize_float(clean), datatype=XSD.double),
                    )
                )
            elif isinstance(clean, str):
                g.add((hiu, SCHEMA.value, Literal(clean)))
            else:
                g.add((hiu, nb.valueJson, Literal(json.dumps(clean, sort_keys=True))))
            hparam_iris.append(hiu)

    # -- Run identity payload + IRI -------------------------------------------
    optimizer = prov_meta.get("optimizer") or {}
    final_loss = prov_meta.get("final_loss")
    best_loss = prov_meta.get("best_loss")
    loss_status = "finite"
    try:
        final_loss_norm = normalize_float(final_loss)
        best_loss_norm = normalize_float(best_loss)
    except ProvenanceError:
        if strict:
            raise
        loss_status = "non-finite"
        final_loss_norm = None
        best_loss_norm = None

    run_payload = {
        "schema": 1,
        "model": str(model_iri),
        "dataset": str(dataset_iri) if dataset_iri else None,
        "training_date": training_date,
        "nobrainer_version": prov_meta.get("nobrainer_version"),
        "pytorch_version": prov_meta.get("pytorch_version"),
        "model_architecture": architecture,
        "architecture_vocab": architecture_vocab,
        "loss_function": prov_meta.get("loss_function"),
        "optimizer": optimizer,
        "epochs_trained": prov_meta.get("epochs_trained"),
        "final_loss": final_loss_norm,
        "best_loss": best_loss_norm,
        "n_classes": prov_meta.get("n_classes"),
        "block_shape": list(prov_meta.get("block_shape") or []),
        "model_args": prov_meta.get("model_args") or {},
        "gpu_count": prov_meta.get("gpu_count"),
    }
    run_iri = _run_iri(base_iri, run_payload)

    _add_hparams(run_iri, "optimizer", optimizer.get("args") or {})
    _add_hparams(run_iri, "model", prov_meta.get("model_args") or {})

    g.add((run_iri, RDF.type, PROV.Activity))
    g.add((run_iri, RDF.type, nb.TrainingRun))
    if doc.get("name"):
        g.add((run_iri, RDFS.label, Literal(doc["name"])))
    if doc.get("description"):
        g.add((run_iri, DCTERMS.description, Literal(doc["description"])))
    if doc.get("conformsTo"):
        g.add(
            (
                run_iri,
                nb.croissantConformsTo,
                Literal(doc["conformsTo"], datatype=XSD.anyURI),
            )
        )
    g.add((run_iri, nb.provenanceScope, Literal("domain")))
    g.add(
        (
            run_iri,
            nb.provenanceSchemaVersion,
            Literal(1, datatype=XSD.nonNegativeInteger),
        )
    )
    g.add((run_iri, nb.runIdentifierSource, Literal("derived")))
    if dataset_iri is not None:
        g.add((run_iri, PROV.used, dataset_iri))
    g.add((run_iri, PROV.generated, model_iri))
    g.add((model_iri, PROV.wasGeneratedBy, run_iri))
    if training_date:
        g.add(
            (
                run_iri,
                nb.metadataRecordedAt,
                Literal(training_date, datatype=XSD.dateTime),
            )
        )
    if prov_meta.get("nobrainer_version"):
        g.add((run_iri, nb.nobrainerVersion, Literal(prov_meta["nobrainer_version"])))
        git_sha = _parse_git_sha(prov_meta["nobrainer_version"])
        if git_sha:
            g.add((run_iri, nb.gitCommitId, Literal(git_sha)))
    if prov_meta.get("pytorch_version"):
        g.add((run_iri, nb.pytorchVersion, Literal(prov_meta["pytorch_version"])))
    if optimizer.get("class"):
        g.add((run_iri, nb.optimizerClass, Literal(optimizer["class"])))
    if prov_meta.get("loss_function"):
        g.add((run_iri, nb.lossFunction, Literal(prov_meta["loss_function"])))
    if prov_meta.get("epochs_trained") is not None:
        g.add(
            (
                run_iri,
                nb.epochsTrained,
                Literal(prov_meta["epochs_trained"], datatype=XSD.nonNegativeInteger),
            )
        )
    if final_loss_norm is not None:
        g.add((run_iri, nb.finalLoss, Literal(float(final_loss), datatype=XSD.double)))
    if best_loss_norm is not None:
        g.add((run_iri, nb.bestLoss, Literal(float(best_loss), datatype=XSD.double)))
    if loss_status == "non-finite":
        g.add((run_iri, nb.lossStatus, Literal("non-finite")))
    if architecture:
        g.add((run_iri, nb.modelArchitecture, Literal(architecture)))
        g.add((run_iri, nb.architectureVocabulary, Literal(architecture_vocab)))
        if architecture_vocab == "nobrainer-model-registry":
            g.add(
                (run_iri, nb.modelArchitectureNormalized, Literal(architecture.lower()))
            )
    if prov_meta.get("n_classes") is not None:
        g.add(
            (
                run_iri,
                nb.numberOfClasses,
                Literal(prov_meta["n_classes"], datatype=XSD.nonNegativeInteger),
            )
        )
    block_shape = prov_meta.get("block_shape") or []
    if block_shape:
        g.add((run_iri, nb.blockShape, Literal(json.dumps(list(block_shape)))))
        g.add(
            (
                run_iri,
                nb.blockShapeRank,
                Literal(len(block_shape), datatype=XSD.nonNegativeInteger),
            )
        )
    if prov_meta.get("gpu_count") is not None:
        g.add(
            (
                run_iri,
                nb.gpuCount,
                Literal(prov_meta["gpu_count"], datatype=XSD.nonNegativeInteger),
            )
        )
    g.add((run_iri, nb.sourceChecksumCoverage, Literal(checksum_coverage)))
    for hiu in hparam_iris:
        g.add((run_iri, nb.hasHyperparameter, hiu))

    # -- Agents -----------------------------------------------------------------
    nobrainer_version = prov_meta.get("nobrainer_version") or nobrainer.__version__
    nb_agent_iri = _software_agent_iri(base_iri, "nobrainer", nobrainer_version)
    g.add((nb_agent_iri, RDF.type, PROV.Agent))
    g.add((nb_agent_iri, RDF.type, PROV.SoftwareAgent))
    g.add((nb_agent_iri, SCHEMA.name, Literal("nobrainer")))
    g.add((nb_agent_iri, SCHEMA.softwareVersion, Literal(nobrainer_version)))
    g.add((nb_agent_iri, RDFS.label, Literal(f"nobrainer {nobrainer_version}")))
    g.add((run_iri, PROV.wasAssociatedWith, nb_agent_iri))

    pytorch_version = prov_meta.get("pytorch_version")
    if pytorch_version:
        pt_agent_iri = _software_agent_iri(base_iri, "pytorch", pytorch_version)
        g.add((pt_agent_iri, RDF.type, PROV.Agent))
        g.add((pt_agent_iri, RDF.type, PROV.SoftwareAgent))
        g.add((pt_agent_iri, SCHEMA.name, Literal("pytorch")))
        g.add((pt_agent_iri, SCHEMA.softwareVersion, Literal(pytorch_version)))
        g.add((pt_agent_iri, RDFS.label, Literal(f"pytorch {pytorch_version}")))
        g.add((run_iri, PROV.wasAssociatedWith, pt_agent_iri))

    if agent:
        if agent.startswith("http://") or agent.startswith("https://"):
            agent_iri = URIRef(agent)
            g.add((agent_iri, RDF.type, PROV.Agent))
        else:
            agent_iri = _human_agent_iri(base_iri, agent)
            g.add((agent_iri, RDF.type, PROV.Agent))
            g.add((agent_iri, RDF.type, PROV.Person))
            g.add((agent_iri, RDFS.label, Literal(agent)))
            email_match = _EMAIL_RE.search(agent)
            if email_match:
                g.add((agent_iri, SCHEMA.email, Literal(email_match.group(0))))
        g.add((run_iri, PROV.wasAssociatedWith, agent_iri))

    _assert_domain_only(g)
    return g


def _assert_domain_only(g: Graph) -> None:
    """Verify a graph contains DOMAIN provenance only -- no BrainKB ingestion triples.

    Parameters
    ----------
    g : rdflib.Graph
        The graph to check.

    Raises
    ------
    ProvenanceError
        If any of the following hold: more than one ``prov:Activity``
        exists; any ``prov:Agent`` is not a ``prov:SoftwareAgent``,
        ``prov:Person``, or plain ``prov:Agent`` associated with the run;
        a forbidden PROV predicate (``wasInformedBy``, ``wasStartedBy``,
        ``wasEndedBy``, ``qualifiedAssociation``, ``hadPlan``) is present;
        any term (subject, predicate, or object) contains ``"brainkb"``
        case-insensitively; or a blank node is present anywhere.
    """
    from rdflib import BNode

    activities = set(g.subjects(RDF.type, PROV.Activity))
    if len(activities) != 1:
        raise ProvenanceError(
            f"Expected exactly one prov:Activity (the training run); found {len(activities)}."
        )

    for s, p, o in g:
        if isinstance(s, BNode) or isinstance(o, BNode):
            raise ProvenanceError(
                "Graph contains a blank node; all IRIs must be content-derived."
            )
        if p in _FORBIDDEN_PREDICATES:
            raise ProvenanceError(f"Forbidden predicate present: {p}")
        for term in (s, p, o):
            if "brainkb" in str(term).lower():
                raise ProvenanceError(
                    f"Term contains 'brainkb': {term!r}. This module emits DOMAIN "
                    "provenance only; BrainKB ingestion-activity triples are added "
                    "by the caller against BrainKB's own API, not by this module."
                )

    agents = set(g.subjects(RDF.type, PROV.Agent))
    for agent_iri in agents:
        associated_with_run = any(
            (run, PROV.wasAssociatedWith, agent_iri) in g for run in activities
        )
        if not associated_with_run:
            raise ProvenanceError(
                f"Agent {agent_iri} is not associated with the training run."
            )


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def to_turtle(g: Graph) -> str:
    """Serialize a graph to Turtle.

    Parameters
    ----------
    g : rdflib.Graph
        The graph to serialize.

    Returns
    -------
    str
        Turtle-formatted text.
    """
    return g.serialize(format="turtle")


def to_jsonld(g: Graph) -> str:
    """Serialize a graph to JSON-LD.

    Parameters
    ----------
    g : rdflib.Graph
        The graph to serialize.

    Returns
    -------
    str
        JSON-LD formatted text.
    """
    return g.serialize(format="json-ld")


def _reparse_or_raise(text: str, fmt: str) -> Graph:
    """Parse serialized RDF text with a fresh Graph and raise on failure.

    Parameters
    ----------
    text : str
        Serialized RDF (Turtle or JSON-LD).
    fmt : str
        ``"turtle"`` or ``"json-ld"``.

    Returns
    -------
    rdflib.Graph
        The re-parsed graph.

    Raises
    ------
    ProvenanceError
        If the text does not parse as valid RDF in the given format. This
        is a correctness self-check performed on every export, not just in
        tests: an export this module cannot re-parse must never be handed
        to a caller.
    """
    try:
        return Graph().parse(data=text, format=fmt)
    except Exception as exc:  # noqa: BLE001 - any parse failure is a real bug here
        raise ProvenanceError(
            f"Serialized {fmt} output failed to re-parse: {exc}"
        ) from exc


def export_provenance(
    bundle_dir: str | Path,
    dataspec_path: str | Path | None = None,
    fmt: str = "turtle",
    base_iri: str = DEFAULT_BASE_IRI,
    agent: str | None = None,
    strict: bool = False,
) -> str:
    """Build a provenance graph and return it serialized, verified re-parseable.

    Parameters
    ----------
    bundle_dir : str or Path
        A ``Segmentation.save()`` directory.
    dataspec_path : str, Path, or None, optional
        Optional DataSpec manifest JSON for image+label dataset provenance.
    fmt : {"turtle", "json-ld"}, optional
        Output serialization. Default ``"turtle"``.
    base_iri : str, optional
        Base IRI for minted identifiers.
    agent : str or None, optional
        Optional caller-supplied human/organization agent identity.
    strict : bool, optional
        If ``True``, raise on conditions this module would otherwise elide.

    Returns
    -------
    str
        The serialized graph, already verified to re-parse with a fresh
        ``rdflib.Graph().parse()``.

    Raises
    ------
    ProvenanceError
        On any input, construction, boundary, or serialization failure.
    ValueError
        If ``fmt`` is not ``"turtle"`` or ``"json-ld"``.
    """
    if fmt not in ("turtle", "json-ld"):
        raise ValueError(f"fmt must be 'turtle' or 'json-ld', got {fmt!r}")

    g = build_graph(
        bundle_dir,
        dataspec_path=dataspec_path,
        base_iri=base_iri,
        agent=agent,
        strict=strict,
    )
    text = to_turtle(g) if fmt == "turtle" else to_jsonld(g)
    _reparse_or_raise(text, fmt)
    return text

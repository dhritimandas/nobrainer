"""BIDS discovery (``from_bids``) and reshaping (``to_bids``).

Self-contained module — imports only stdlib, matching the import discipline
of :mod:`nobrainer.data.spec` (its module docstring: "Does NOT import from
``nobrainer.*`` so it can be used by external CI scripts without installing
torch/monai"). ``pybids`` is imported lazily, only inside the pybids backend.

Entity grammar and directory rules are pinned to BIDS **1.11.1**
(``https://bids-specification.readthedocs.io/en/stable/``, schema 1.2.1).
"""

from __future__ import annotations

from collections import Counter, defaultdict
import dataclasses
import enum
import json
import logging
from pathlib import Path
import re
from typing import Any, Callable

logger = logging.getLogger(__name__)

BIDS_VERSION = "1.11.1"

# ---------------------------------------------------------------------------
# Entity grammar (BIDS 1.11.1, src/schema/objects/formats.yaml)
# ---------------------------------------------------------------------------

LABEL_PATTERN = re.compile(r"^[0-9a-zA-Z+]+$")
INDEX_PATTERN = re.compile(r"^[0-9]+$")

# Entities whose value type is "index" (non-negative integer); every other
# entity is a "label" ([0-9a-zA-Z+]+). src/schema/rules/entities.yaml.
INDEX_ENTITIES = frozenset({"run", "echo", "flip", "inv", "split", "chunk"})

# Ordered, allowed entity sets per suffix. An entity absent from a suffix's
# list is MUST NOT, not merely unused (BIDS entity-table legend).
ANAT_RAW_ORDER = (
    "sub",
    "ses",
    "task",
    "acq",
    "ce",
    "rec",
    "run",
    "echo",
    "part",
    "chunk",
)
DSEG_ORDER = (
    "sub",
    "ses",
    "task",
    "acq",
    "ce",
    "rec",
    "run",
    "echo",
    "part",
    "space",
    "chunk",
    "atlas",
    "seg",
    "scale",
    "res",
    "desc",
)
MASK_ORDER = (
    "sub",
    "ses",
    "task",
    "acq",
    "ce",
    "rec",
    "run",
    "echo",
    "part",
    "space",
    "chunk",
    "res",
    "label",
    "desc",
)
PROBSEG_ORDER = (
    "sub",
    "ses",
    "task",
    "acq",
    "ce",
    "rec",
    "run",
    "echo",
    "part",
    "space",
    "chunk",
    "atlas",
    "seg",
    "scale",
    "res",
    "label",
    "desc",
)

STRUCTURAL_SUFFIXES = ("T1w", "T2w", "FLAIR", "T2starw", "PDw")
SEG_SUFFIXES = ("dseg", "probseg", "mask", "defacemask")
_KNOWN_SUFFIXES = frozenset(STRUCTURAL_SUFFIXES) | frozenset(SEG_SUFFIXES)

_SUFFIX_ORDER = {"dseg": DSEG_ORDER, "mask": MASK_ORDER, "probseg": PROBSEG_ORDER}

# A derivative reuses its source's entities and appends these; stripping them
# is how a derivative file is joined back to its raw source (see
# "Image <-> label matching" in the plan).
_DERIVATIVE_ONLY_ENTITIES = frozenset(
    {"space", "res", "den", "atlas", "seg", "scale", "label", "desc"}
)

_KNOWN_EXTENSIONS = (".nii.gz", ".nii", ".json", ".tsv")

_ENTITY_TOKEN = re.compile(r"^([a-zA-Z]+)-(.+)$")


def _allowed_entities_for_suffix(suffix: str) -> tuple[str, ...]:
    """Return the ordered, allowed entity set for a suffix."""
    return _SUFFIX_ORDER.get(suffix, ANAT_RAW_ORDER)


def _split_extension(name: str) -> tuple[str, str]:
    """Split a filename into ``(stem, extension)``.

    Treats ``.nii.gz`` as a single unit -- ``str.rpartition(".")`` is wrong
    for it.
    """
    for ext in _KNOWN_EXTENSIONS:
        if name.endswith(ext):
            return name[: -len(ext)], ext
    return name, ""


@dataclasses.dataclass(frozen=True)
class ParsedName:
    """A parsed BIDS filename.

    Attributes
    ----------
    entities : dict of str to str
        Entity key -> value (values are always strings; index entities such
        as ``run`` keep their original zero-padding).
    suffix : str
        The filename suffix (e.g. ``"T1w"``, ``"dseg"``).
    extension : str
        File extension, including the leading dot (``.nii.gz`` is one unit).
    """

    entities: dict[str, str]
    suffix: str
    extension: str


def parse_bids_filename(name: str) -> ParsedName | None:
    """Parse a single BIDS filename into entities, suffix, and extension.

    Grammar is the BIDS 1.11.1 value grammar: entity values match
    ``[0-9a-zA-Z+]+`` (label) or ``[0-9]+`` (index); no entity may repeat.
    This function checks grammar only -- it does not check that an entity is
    *allowed* for the parsed suffix (see :func:`_allowed_entities_for_suffix`
    for that).

    Parameters
    ----------
    name : str
        A bare filename (no directory components).

    Returns
    -------
    ParsedName or None
        ``None`` if the name does not match the BIDS grammar, including
        repeated entities, malformed entity tokens, or an unrecognized
        extension.
    """
    if not name or name.startswith(".") or len(name) > 255:
        return None
    stem, ext = _split_extension(name)
    if not ext or not stem:
        return None
    parts = stem.split("_")
    suffix = parts[-1]
    if not suffix or not LABEL_PATTERN.match(suffix) or "-" in suffix:
        return None
    entities: dict[str, str] = {}
    for token in parts[:-1]:
        match = _ENTITY_TOKEN.match(token)
        if not match:
            return None
        key, value = match.group(1), match.group(2)
        if key in entities:
            return None  # BIDS: an entity MUST NOT appear more than once
        pattern = INDEX_PATTERN if key in INDEX_ENTITIES else LABEL_PATTERN
        if not pattern.match(value):
            return None
        entities[key] = value
    return ParsedName(entities=entities, suffix=suffix, extension=ext)


# ---------------------------------------------------------------------------
# Skip reporting
# ---------------------------------------------------------------------------


class SkipReason(enum.Enum):
    """Why a candidate file was excluded from a scan or write."""

    UNPARSEABLE_NAME = "unparseable_name"
    NOT_A_DATATYPE_DIR = "not_a_datatype_dir"
    UNKNOWN_SUFFIX = "unknown_suffix"
    NO_MATCHING_LABEL = "no_matching_label"
    AMBIGUOUS_LABEL = "ambiguous_label"
    DUPLICATE_EXTENSION = "duplicate_extension"
    ENTITY_NOT_ALLOWED = "entity_not_allowed"
    UNREADABLE_FILE = "unreadable_file"
    UNSUPPORTED_EXTENSION = "unsupported_extension"


@dataclasses.dataclass(frozen=True)
class SkippedFile:
    """A single file excluded from a scan or write, and why."""

    path: str
    reason: SkipReason
    detail: str


@dataclasses.dataclass
class BidsScan:
    """Result of scanning a BIDS dataset.

    Attributes
    ----------
    entries : list of dict
        ``{"image": path}`` or ``{"image": path, "label": path}`` records,
        matching the schema :func:`nobrainer.dataset.get_dataset` expects.
        Either every entry has a ``"label"`` key or none do -- never a mix
        (see ``processing/dataset.py``'s parallel-list construction, which
        raises on a length mismatch between images and labels).
    skipped : list of SkippedFile
        Every file that was found but not included, with a reason.
    subject_ids : list of str
        BIDS subject labels (without the ``sub-`` prefix) present in
        ``entries``, in the same order.
    """

    entries: list[dict[str, str]]
    skipped: list[SkippedFile]
    subject_ids: list[str]


# ---------------------------------------------------------------------------
# Small helpers duplicated (not imported) from nobrainer.datasets.openneuro
# ---------------------------------------------------------------------------
#
# nobrainer.data keeps a stdlib-only import boundary (see module docstring).
# nobrainer.datasets.openneuro is lightweight today, but importing across
# that boundary would tie nobrainer.data's contract to it staying that way.
# These mirror openneuro._extract_subject_id / _file_ok exactly.


def _extract_subject_id(path: Path) -> str:
    """Extract ``sub-XX`` from a BIDS-style path."""
    for part in path.parts[:-1]:
        if part.startswith("sub-"):
            return part
    name = path.name
    if name.startswith("sub-"):
        return name.split("_")[0]
    return name


def _file_ok(p: Path) -> bool:
    """True if *p* is a real file with nonzero size (follows symlinks)."""
    try:
        return p.stat().st_size > 0
    except OSError:
        return False


# ---------------------------------------------------------------------------
# Directory walking
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class _Candidate:
    path: Path
    parsed: ParsedName
    sub_from_dir: str | None
    ses_from_dir: str | None


def _is_candidate_file(f: Path) -> bool:
    """True if *f* should be considered, including broken symlinks.

    ``Path.is_file()`` follows symlinks and returns False for a broken one
    -- exactly the git-annex placeholder case ``from_bids`` must still
    surface, so that :func:`nobrainer.data.spec.validate` can report
    ``ANNEX_MISSING`` on it (see the module docstring).
    """
    return (f.is_file() or f.is_symlink()) and not f.name.startswith(".")


def _iter_datatype_files(root: Path, datatype: str = "anat"):
    """Yield files under ``sub-*/[ses-*/]<datatype>/``, with dir context.

    Yields
    ------
    tuple of (Path, str or None, str or None)
        ``(file_path, subject_label_from_dir, session_label_from_dir)``.
    """
    if not root.is_dir():
        return
    for sub_dir in sorted(p for p in root.glob("sub-*") if p.is_dir()):
        sub_label = sub_dir.name[len("sub-") :]
        ses_dirs = sorted(p for p in sub_dir.glob("ses-*") if p.is_dir())
        if ses_dirs:
            for ses_dir in ses_dirs:
                ses_label = ses_dir.name[len("ses-") :]
                dt_dir = ses_dir / datatype
                if dt_dir.is_dir():
                    for f in sorted(dt_dir.iterdir()):
                        if _is_candidate_file(f):
                            yield f, sub_label, ses_label
        else:
            dt_dir = sub_dir / datatype
            if dt_dir.is_dir():
                for f in sorted(dt_dir.iterdir()):
                    if _is_candidate_file(f):
                        yield f, sub_label, None


def _iter_derivative_files(derivatives_dir: Path | None, datatype: str = "anat"):
    """Yield derivative files, searching every pipeline under *derivatives_dir*.

    If *derivatives_dir* itself is a single pipeline (has its own
    ``dataset_description.json``), only that pipeline is searched.
    """
    if derivatives_dir is None or not derivatives_dir.is_dir():
        return
    if (derivatives_dir / "dataset_description.json").is_file():
        pipeline_dirs = [derivatives_dir]
    else:
        pipeline_dirs = sorted(
            p
            for p in derivatives_dir.iterdir()
            if p.is_dir() and not p.name.startswith(".")
        )
    for pdir in pipeline_dirs:
        yield from _iter_datatype_files(pdir, datatype)


def _join_key(entities: dict[str, str]) -> tuple[tuple[str, str], ...]:
    """Strip derivative-only entities, producing a key comparable across
    a raw image and the derivative(s) built from it."""
    return tuple(
        sorted(
            (k, v) for k, v in entities.items() if k not in _DERIVATIVE_ONLY_ENTITIES
        )
    )


def _read_sources(json_sidecar: Path) -> list[str]:
    """Read the ``Sources`` list from a JSON sidecar, if present and valid."""
    try:
        data = json.loads(json_sidecar.read_text())
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return []
    sources = data.get("Sources", [])
    return [s for s in sources if isinstance(s, str)]


def _source_join_key(uri: str) -> tuple[tuple[str, str], ...] | None:
    """Parse a BIDS-URI ``Sources`` entry into a join key, if possible."""
    name = uri.rsplit("/", 1)[-1]
    parsed = parse_bids_filename(name)
    if parsed is None:
        return None
    return _join_key(parsed.entities)


def _sidecar_path(path: Path) -> Path:
    """Return the ``.json`` sidecar path for a data file."""
    stem, _ = _split_extension(path.name)
    return path.with_name(stem + ".json")


def _resolve_candidates(
    candidates: list[_Candidate], ambiguous_reason: SkipReason
) -> tuple[Path | None, list[SkippedFile]]:
    """Pick one file among candidates sharing a join key, or report why not.

    Candidates that differ only by extension (same entities + suffix) are
    a "duplicate extension" case: BIDS requires uniqueness there, so unless
    exactly one candidate is a real, nonempty file (the others being bogus
    empty stubs), neither is chosen and both are reported.

    Candidates that differ in their entities (not just extension) sharing
    the same join key are a genuine ambiguity (e.g. two derivative pipelines
    both producing a ``dseg`` for the same source image) and are reported
    with *ambiguous_reason*.
    """
    if len(candidates) == 1:
        return candidates[0].path, []

    by_identity: dict[tuple, list[Path]] = defaultdict(list)
    for c in candidates:
        identity = (tuple(sorted(c.parsed.entities.items())), c.parsed.suffix)
        by_identity[identity].append(c.path)

    if len(by_identity) == 1:
        paths = next(iter(by_identity.values()))
        ok_paths = [p for p in paths if _file_ok(p)]
        if len(ok_paths) == 1:
            return ok_paths[0], []
        return None, [
            SkippedFile(
                str(p),
                SkipReason.DUPLICATE_EXTENSION,
                f"multiple extensions for identical entities: {[str(x) for x in paths]}",
            )
            for p in paths
        ]

    return None, [
        SkippedFile(
            str(c.path),
            ambiguous_reason,
            f"{len(candidates)} candidates share the same join key",
        )
        for c in candidates
    ]


def _collect(
    root: Path,
    derivatives_dir: Path | None,
    suffix: str,
    label_suffix: str,
    session: str | None,
    require_labels: bool,
    parse_fn: Callable[[Path], ParsedName | None],
) -> BidsScan:
    """Shared discovery core for both the walker and pybids backends."""
    skipped: list[SkippedFile] = []
    image_candidates: dict[tuple, list[_Candidate]] = defaultdict(list)
    label_candidates: dict[tuple, list[_Candidate]] = defaultdict(list)
    label_sources: dict[tuple, tuple[tuple[str, str], ...]] = {}

    for path, sub_from_dir, ses_from_dir in _iter_datatype_files(root, "anat"):
        _classify(
            path,
            sub_from_dir,
            ses_from_dir,
            suffix,
            session,
            parse_fn,
            skipped,
            image_candidates,
        )

    deriv_root = (
        derivatives_dir if derivatives_dir is not None else root / "derivatives"
    )
    for path, sub_from_dir, ses_from_dir in _iter_derivative_files(deriv_root, "anat"):
        candidate = _classify(
            path,
            sub_from_dir,
            ses_from_dir,
            label_suffix,
            session,
            parse_fn,
            skipped,
            label_candidates,
        )
        if candidate is not None:
            sidecar = _sidecar_path(path)
            if sidecar.is_file():
                for uri in _read_sources(sidecar):
                    source_key = _source_join_key(uri)
                    if source_key is not None:
                        label_sources[_join_key(candidate.parsed.entities)] = source_key

    resolved_images: dict[tuple, Path] = {}
    for key, candidates in image_candidates.items():
        chosen, dup_skips = _resolve_candidates(
            candidates, SkipReason.DUPLICATE_EXTENSION
        )
        skipped.extend(dup_skips)
        if chosen is not None:
            resolved_images[key] = chosen

    resolved_labels: dict[tuple, Path] = {}
    for key, candidates in label_candidates.items():
        chosen, dup_skips = _resolve_candidates(candidates, SkipReason.AMBIGUOUS_LABEL)
        skipped.extend(dup_skips)
        if chosen is not None:
            resolved_labels[key] = chosen

    # A Sources sidecar is authoritative: if it points at a *different*
    # image than the plain entity join would have, prefer it.
    for label_key, source_key in label_sources.items():
        if label_key in resolved_labels and source_key in resolved_images:
            resolved_labels[source_key] = resolved_labels[label_key]

    entries: list[dict[str, str]] = []
    subject_ids: list[str] = []
    for key in sorted(resolved_images, key=lambda k: str(resolved_images[k])):
        image_path = resolved_images[key]
        sub_id = dict(key).get("sub")
        if require_labels:
            label_path = resolved_labels.get(key)
            if label_path is None:
                skipped.append(
                    SkippedFile(
                        str(image_path),
                        SkipReason.NO_MATCHING_LABEL,
                        f"no '{label_suffix}' file found for entities {dict(key)}",
                    )
                )
                continue
            entries.append({"image": str(image_path), "label": str(label_path)})
        else:
            entries.append({"image": str(image_path)})
        if sub_id:
            subject_ids.append(sub_id)

    return BidsScan(entries=entries, skipped=skipped, subject_ids=subject_ids)


def _classify(
    path: Path,
    sub_from_dir: str | None,
    ses_from_dir: str | None,
    target_suffix: str,
    session: str | None,
    parse_fn: Callable[[Path], ParsedName | None],
    skipped: list[SkippedFile],
    bucket: dict[tuple, list[_Candidate]],
) -> _Candidate | None:
    """Parse and validate one file; append it to *bucket* or *skipped*."""
    parsed = parse_fn(path)
    if parsed is None:
        skipped.append(
            SkippedFile(
                str(path),
                SkipReason.UNPARSEABLE_NAME,
                "filename does not match the BIDS entity-suffix-extension grammar",
            )
        )
        return None

    if parsed.extension not in (".nii", ".nii.gz"):
        # JSON sidecars and TSV lookup tables sit next to data files with the
        # same entities/suffix -- BIDS's uniqueness rule explicitly excludes
        # them ("This limitation does not apply to metadata files"). Not an
        # error, just not a data-file candidate.
        return None

    if (
        parsed.entities.get("sub") != sub_from_dir
        or parsed.entities.get("ses") != ses_from_dir
    ):
        skipped.append(
            SkippedFile(
                str(path),
                SkipReason.NOT_A_DATATYPE_DIR,
                "sub/ses entities in the filename do not match the directory "
                f"structure (dir: sub={sub_from_dir!r} ses={ses_from_dir!r}, "
                f"filename: sub={parsed.entities.get('sub')!r} "
                f"ses={parsed.entities.get('ses')!r})",
            )
        )
        return None

    if parsed.suffix != target_suffix:
        if parsed.suffix not in _KNOWN_SUFFIXES:
            skipped.append(
                SkippedFile(
                    str(path),
                    SkipReason.UNKNOWN_SUFFIX,
                    f"suffix '{parsed.suffix}' is not a recognized BIDS anat suffix",
                )
            )
        return None  # a different, known suffix (e.g. T2w when we want T1w) -- not an error

    if session is not None and parsed.entities.get("ses") != session:
        return None  # filtered out by request, not an error

    allowed = set(_allowed_entities_for_suffix(parsed.suffix))
    disallowed = set(parsed.entities) - allowed
    if disallowed:
        skipped.append(
            SkippedFile(
                str(path),
                SkipReason.ENTITY_NOT_ALLOWED,
                f"entities not allowed for suffix '{parsed.suffix}': {sorted(disallowed)}",
            )
        )
        return None

    candidate = _Candidate(
        path=path, parsed=parsed, sub_from_dir=sub_from_dir, ses_from_dir=ses_from_dir
    )
    bucket[_join_key(parsed.entities)].append(candidate)
    return candidate


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


def _pybids_available() -> bool:
    try:
        import bids  # noqa: F401
    except ImportError:
        return False
    return True


def _pybids_import_error() -> ImportError:
    return ImportError(
        "backend='pybids' requires the optional 'pybids' package. "
        "Install with: uv pip install -e '.[bids]'"
    )


# pybids uses long entity names from bids.json/derivatives.json that differ
# from the short BIDS schema keys used throughout this module.
_PYBIDS_TO_BIDS = {
    "subject": "sub",
    "session": "ses",
    "task": "task",
    "acquisition": "acq",
    "ceagent": "ce",
    "reconstruction": "rec",
    "run": "run",
    "echo": "echo",
    "flip": "flip",
    "inversion": "inv",
    "part": "part",
    "chunk": "chunk",
    "space": "space",
    "resolution": "res",
    "density": "den",
    "label": "label",
    "atlas": "atlas",
    "segmentation": "seg",
    "description": "desc",
    "modality": "mod",
    "direction": "dir",
    "hemisphere": "hemi",
    "staining": "stain",
    "volume": "voi",
    "nucleus": "nuc",
    "mtransfer": "mt",
    "processing": "proc",
    "sample": "sample",
    "recording": "recording",
    "tracer": "trc",
    "scale": "scale",
    "split": "split",
    "cohort": "cohort",
    "tracksys": "tracksys",
}


def _pybids_parse(path: Path) -> ParsedName | None:
    """Parse a filename with pybids, normalized to our entity vocabulary.

    Two documented pybids traps are guarded here: (1) its ``subject`` and
    ``datatype`` patterns require a preceding path separator and silently
    drop those entities for a bare filename -- always pass an absolute
    path; (2) ``run`` is coerced to ``int``, destroying zero-padding -- cast
    back to ``str``.
    """
    from bids.layout import parse_file_entities

    raw = parse_file_entities(str(path.resolve()))
    if "suffix" not in raw:
        return None
    entities: dict[str, str] = {}
    for pybids_key, value in raw.items():
        if pybids_key in ("suffix", "extension", "datatype"):
            continue
        short = _PYBIDS_TO_BIDS.get(pybids_key)
        if short is not None:
            entities[short] = str(value)
    extension = raw.get("extension", "")
    if extension and not extension.startswith("."):
        extension = "." + extension
    return ParsedName(entities=entities, suffix=raw["suffix"], extension=extension)


def scan_bids(
    root: str | Path,
    *,
    suffix: str = "T1w",
    label_suffix: str = "dseg",
    derivatives_dir: str | Path | None = None,
    session: str | None = None,
    require_labels: bool = True,
    backend: str = "auto",
) -> BidsScan:
    """Discover image/label pairs in a BIDS (or BIDS-Derivatives) dataset.

    Never raises on a per-file problem -- every unusable file is recorded in
    ``result.skipped`` with a reason and the scan continues. Only two things
    are fatal: *root* does not exist, or nothing at all was found.

    Parameters
    ----------
    root : str or Path
        BIDS dataset root (containing ``sub-*`` directories).
    suffix : str
        Raw anat suffix to discover as the image (default ``"T1w"``).
    label_suffix : str
        Derivative suffix to discover as the label (default ``"dseg"``).
    derivatives_dir : str or Path, or None
        Where to look for labels. Defaults to ``root / "derivatives"``. May
        point at one pipeline directly, or at a parent containing several --
        all immediate subdirectories are then searched.
    session : str or None
        If given, restrict to this session label (without the ``ses-``
        prefix).
    require_labels : bool
        If True (default), only subjects with both an image and a matching
        label are returned, and image-only subjects are recorded in
        ``skipped`` with reason ``NO_MATCHING_LABEL``. If False, every
        found image is returned and no entry carries a ``"label"`` key --
        entries are never a mix of labelled and unlabelled (see
        ``processing/dataset.py``'s parallel-list construction).
    backend : {"auto", "pybids", "walker"}
        ``"auto"`` (default) uses pybids when it is importable, else the
        zero-dependency walker. ``"walker"`` always uses the internal
        implementation. ``"pybids"`` requires the optional ``pybids``
        package and raises ``ImportError`` naming the extra if it is
        missing.

    Returns
    -------
    BidsScan

    Raises
    ------
    FileNotFoundError
        If *root* is not an existing directory.
    ValueError
        If *backend* is invalid, or if nothing matched -- the message
        includes a histogram of why files were skipped.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(
            f"BIDS root does not exist or is not a directory: {root}"
        )

    if backend == "auto":
        backend = "pybids" if _pybids_available() else "walker"

    if backend == "walker":
        parse_fn: Callable[[Path], ParsedName | None] = lambda p: parse_bids_filename(
            p.name
        )
    elif backend == "pybids":
        if not _pybids_available():
            raise _pybids_import_error()
        parse_fn = _pybids_parse
    else:
        raise ValueError(
            f"Unknown backend {backend!r}. Expected 'auto', 'pybids', or 'walker'."
        )

    deriv_dir = Path(derivatives_dir) if derivatives_dir is not None else None
    result = _collect(
        root, deriv_dir, suffix, label_suffix, session, require_labels, parse_fn
    )

    if not result.entries:
        histogram = dict(Counter(s.reason.value for s in result.skipped))
        raise ValueError(
            f"No BIDS entries found under {root} (suffix={suffix!r}, "
            f"label_suffix={label_suffix!r}, require_labels={require_labels}). "
            f"Skip reasons: {histogram or 'no candidate files were found at all'}."
        )

    for s in result.skipped:
        logger.warning(
            "scan_bids: skipped %s (%s): %s", s.path, s.reason.value, s.detail
        )

    return result


# ---------------------------------------------------------------------------
# to_bids: reshape arbitrary image/label pairs into a BIDS tree
# ---------------------------------------------------------------------------

_TO_BIDS_EXTENSIONS = (".nii.gz", ".nii", ".mgz")


@dataclasses.dataclass(frozen=True)
class SubjectMapping:
    """How one input entry's subject label was derived."""

    entry_index: int
    subject_label: str
    strategy: str  # "explicit" | "sanitized" | "sequential"


@dataclasses.dataclass
class BidsWriteReport:
    """Result of :func:`to_bids`."""

    out_root: str
    n_written: int
    skipped: list[SkippedFile]
    subject_mapping: list[SubjectMapping]


def _sanitize_label(text: str) -> str | None:
    """Sanitize an arbitrary string to a BIDS label, or None if empty."""
    cleaned = re.sub(r"[^0-9a-zA-Z+]", "", text)
    return cleaned or None


def _strip_known_extension(name: str) -> str:
    for ext in _TO_BIDS_EXTENSIONS:
        if name.endswith(ext):
            return name[: -len(ext)]
    return name


def _derive_subject_label(
    image_path: Path, index: int, explicit: str | None
) -> tuple[str, str]:
    """Derive a BIDS subject label for one input entry.

    Order: an explicit label (sanitized) -> an existing ``sub-<label>``
    token already in the input path -> the sanitized filename stem ->
    a sequential fallback. Returns ``(label, strategy)``.
    """
    if explicit is not None:
        label = _sanitize_label(str(explicit))
        if label:
            return label, "explicit"

    dir_token = _extract_subject_id(image_path)
    if dir_token.startswith("sub-"):
        label = _sanitize_label(dir_token[len("sub-") :])
        if label:
            return label, "sanitized"

    label = _sanitize_label(_strip_known_extension(image_path.name))
    if label:
        return label, "sanitized"

    return f"{index + 1:03d}", "sequential"


def _symlink(src: Path, dest: Path) -> None:
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    dest.symlink_to(src.resolve())


def _write_dataset_description(root: Path, name: str, *, derivative: bool) -> None:
    payload: dict[str, Any] = {"Name": name, "BIDSVersion": BIDS_VERSION}
    if derivative:
        payload["DatasetType"] = "derivative"
        payload["GeneratedBy"] = [{"Name": "nobrainer"}]
    (root / "dataset_description.json").write_text(json.dumps(payload, indent=2) + "\n")


def _write_dseg_tsv(path: Path, label_names: dict[int, str] | list[str]) -> None:
    items = (
        sorted(label_names.items())
        if isinstance(label_names, dict)
        else list(enumerate(label_names))
    )
    lines = ["index\tname"] + [f"{idx}\t{name}" for idx, name in items]
    path.write_text("\n".join(lines) + "\n")


def to_bids(
    entries: list[dict[str, str]],
    out_root: str | Path,
    *,
    dataset_name: str = "nobrainer dataset",
    subject_ids: list[str] | None = None,
    suffix: str = "T1w",
    desc: str = "nobrainer",
    label_names: dict[int, str] | list[str] | None = None,
    overwrite: bool = False,
) -> BidsWriteReport:
    """Reshape ``{"image", "label"}`` entries into a BIDS(-Derivatives) tree.

    Writes symlinks only, never copies -- input datasets are large, and
    symlinks keep :func:`nobrainer.data.spec.check_file_presence` meaningful
    on the result.

    Labels are written under ``<out_root>/derivatives/nobrainer/`` as
    ``_dseg`` files, not as raw files: ``dseg``/``probseg``/``mask`` are not
    valid raw-BIDS anat suffixes (BIDS 1.11.1, ``rules.files.deriv.imaging``).

    Never fails on a single bad input -- unreadable files, unsupported
    extensions, and unsanitizable names are recorded in the report's
    ``skipped`` list and the rest of the entries are still written.

    Parameters
    ----------
    entries : list of dict
        ``{"image": path}`` or ``{"image": path, "label": path}`` records.
    out_root : str or Path
        Output BIDS root. Created if missing.
    dataset_name : str
        ``Name`` field for the root ``dataset_description.json``.
    subject_ids : list of str, or None
        Explicit subject labels, same length and order as *entries*. When
        None, a label is derived per entry (see :func:`_derive_subject_label`).
    suffix : str
        Raw anat suffix to write images as (default ``"T1w"``).
    desc : str
        ``desc-`` entity value for written label files (default
        ``"nobrainer"``).
    label_names : dict of int to str, list of str, or None
        Written to ``derivatives/nobrainer/dseg.tsv`` as the ``index``/
        ``name`` lookup table, when any label was written.
    overwrite : bool
        Required to write into a non-empty *out_root*.

    Returns
    -------
    BidsWriteReport

    Raises
    ------
    FileExistsError
        If *out_root* is non-empty and *overwrite* is False.
    ValueError
        If nothing was written -- see the raised message and, for a partial
        failure, ``report.skipped``.
    """
    out_root = Path(out_root)
    if out_root.exists() and any(out_root.iterdir()) and not overwrite:
        raise FileExistsError(
            f"{out_root} already exists and is not empty; pass overwrite=True."
        )
    out_root.mkdir(parents=True, exist_ok=True)
    deriv_root = out_root / "derivatives" / "nobrainer"

    skipped: list[SkippedFile] = []
    mapping: list[SubjectMapping] = []
    used_casefold: set[str] = set()
    n_written = 0
    any_label_written = False

    for i, entry in enumerate(entries):
        image = entry.get("image")
        if not image:
            skipped.append(
                SkippedFile(
                    f"<entry {i}>", SkipReason.UNPARSEABLE_NAME, "missing 'image' key"
                )
            )
            continue

        image_path = Path(image)
        image_ext = next((e for e in _TO_BIDS_EXTENSIONS if image.endswith(e)), None)
        if image_ext is None:
            skipped.append(
                SkippedFile(
                    image,
                    SkipReason.UNSUPPORTED_EXTENSION,
                    f"extension not in {_TO_BIDS_EXTENSIONS}",
                )
            )
            continue
        if not _file_ok(image_path):
            skipped.append(
                SkippedFile(
                    image, SkipReason.UNREADABLE_FILE, "file does not exist or is empty"
                )
            )
            continue

        explicit = subject_ids[i] if subject_ids is not None else None
        sub_label, strategy = _derive_subject_label(image_path, i, explicit)
        if sub_label.casefold() in used_casefold:
            sub_label = f"{i + 1:03d}"
            strategy = "sequential"
        used_casefold.add(sub_label.casefold())

        anat_dir = out_root / f"sub-{sub_label}" / "anat"
        anat_dir.mkdir(parents=True, exist_ok=True)
        _symlink(image_path, anat_dir / f"sub-{sub_label}_{suffix}{image_ext}")

        label = entry.get("label")
        if label:
            label_path = Path(label)
            label_ext = next(
                (e for e in _TO_BIDS_EXTENSIONS if label.endswith(e)), None
            )
            if label_ext is None:
                skipped.append(
                    SkippedFile(
                        label,
                        SkipReason.UNSUPPORTED_EXTENSION,
                        f"extension not in {_TO_BIDS_EXTENSIONS}",
                    )
                )
            elif not _file_ok(label_path):
                skipped.append(
                    SkippedFile(
                        label,
                        SkipReason.UNREADABLE_FILE,
                        "file does not exist or is empty",
                    )
                )
            else:
                deriv_anat_dir = deriv_root / f"sub-{sub_label}" / "anat"
                deriv_anat_dir.mkdir(parents=True, exist_ok=True)
                _symlink(
                    label_path,
                    deriv_anat_dir
                    / f"sub-{sub_label}_space-orig_desc-{desc}_dseg{label_ext}",
                )
                any_label_written = True

        mapping.append(
            SubjectMapping(entry_index=i, subject_label=sub_label, strategy=strategy)
        )
        n_written += 1

    if n_written == 0:
        raise ValueError(
            f"No entries were written to {out_root}; see the skipped files for reasons."
        )

    _write_dataset_description(out_root, dataset_name, derivative=False)
    if any_label_written:
        _write_dataset_description(
            deriv_root, f"{dataset_name} (nobrainer derivatives)", derivative=True
        )
        if label_names:
            _write_dseg_tsv(deriv_root / "dseg.tsv", label_names)

    return BidsWriteReport(
        out_root=str(out_root),
        n_written=n_written,
        skipped=skipped,
        subject_mapping=mapping,
    )

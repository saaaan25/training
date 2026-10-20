"""Auditoría de metadata.csv y de los WAV en disco."""

import hashlib
import json
from pathlib import Path

import pandas as pd

from common import LABELS, SPLITS, sha256_file

SPLIT_ALIASES = {
    "train": "train", "training": "train",
    "val": "val", "validation": "val", "valid": "val",
    "test": "test", "testing": "test",
}
AUGMENTED = {"augmented", "augmented_existing", "augmented_generated"}
ALLOWED_ORIGINS = {"real_clean"} | AUGMENTED
PROFILES = {
    "semi_dataset": {"train": 400, "val": 50, "test": 50},
    "final_dataset": {"train": 800, "val": 100, "test": 100},
}


def load_metadata(path, source_column):
    meta = pd.read_csv(path)

    if (source_column == "source_group" and source_column not in meta.columns
            and "source_parent" in meta.columns):
        source_column = "source_parent"
    if "parent_file" not in meta.columns and "clean_parent" in meta.columns:
        meta["parent_file"] = meta["clean_parent"]

    missing = {"file_name", "label", "split", "origin", source_column} - set(meta.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas en metadata: {sorted(missing)}")

    raw_split = meta["split"].astype(str).str.strip().str.lower()
    unknown = sorted(set(raw_split) - set(SPLIT_ALIASES))
    if unknown:
        raise ValueError(f"Splits no reconocidos en metadata: {unknown}")

    meta["file_name"] = meta["file_name"].astype(str).map(lambda v: Path(v).name)
    meta["label"] = meta["label"].astype(str).str.strip()
    meta["origin"] = meta["origin"].astype(str).str.strip()
    meta["split"] = raw_split.map(SPLIT_ALIASES)
    meta[source_column] = meta[source_column].fillna("").astype(str).str.strip()
    meta["relative_path"] = meta["split"] + "/" + meta["label"] + "/" + meta["file_name"]
    return meta, source_column


def check_metadata(meta, source_column):
    if meta[source_column].eq("").any():
        raise ValueError(f"La columna {source_column!r} contiene valores vacíos.")

    invalid = sorted(set(meta["label"]) - set(LABELS))
    if invalid:
        raise ValueError(f"Etiquetas no permitidas: {invalid}")

    invalid = sorted(set(meta["origin"]) - ALLOWED_ORIGINS)
    if invalid:
        raise ValueError(f"Valores no permitidos en origin: {invalid}. "
                         f"Admitidos: {sorted(ALLOWED_ORIGINS)}")

    for column in ("file_name", "relative_path"):
        duplicated = meta.loc[meta[column].duplicated(keep=False), column]
        if not duplicated.empty:
            raise ValueError(f"{column} duplicado en metadata: {duplicated.tolist()[:10]}")

    leaked = meta.groupby(source_column)["split"].nunique()
    leaked = leaked[leaked > 1].index.tolist()
    if leaked:
        raise ValueError(f"Fuentes presentes en más de un split. Ejemplos: {leaked[:10]}")

    augmented = meta["origin"].isin(AUGMENTED)
    if (augmented & meta["split"].ne("train")).any():
        raise ValueError("Existen audios aumentados fuera de train.")
    if (meta["split"].isin(["val", "test"]) & meta["origin"].ne("real_clean")).any():
        raise ValueError("Validation y test deben contener solo origin=real_clean.")

    if not augmented.any():
        return
    if "parent_file" not in meta.columns:
        raise ValueError("metadata requiere parent_file/clean_parent para registros augmented.")

    children = meta[augmented].copy()
    children["parent"] = (children["parent_file"].fillna("").astype(str)
                          .str.strip().map(lambda v: Path(v).name))
    parents = meta.set_index("file_name")[["origin", "split", "label", source_column]]
    joined = children.join(parents, on="parent", rsuffix="_parent")

    invalid = (
        joined["origin_parent"].isna()
        | joined["origin_parent"].ne("real_clean")
        | joined["split_parent"].ne("train")
        | joined["label_parent"].ne(joined["label"])
        | joined[f"{source_column}_parent"].ne(joined[source_column])
    )
    if invalid.any():
        examples = joined.loc[invalid, ["file_name", "parent"]].head(5).values.tolist()
        raise ValueError("Padre inexistente o inválido (debe ser real_clean, de train, "
                         f"misma clase y misma fuente). Ejemplos: {examples}")


def scan_disk(dataset_dir):
    files = []
    for split in SPLITS:
        for label in LABELS:
            folder = dataset_dir / split / label
            if not folder.is_dir():
                raise FileNotFoundError(f"No existe la carpeta esperada: {folder}")
            for path in sorted(folder.glob("*.wav")):
                files.append({
                    "relative_path": path.relative_to(dataset_dir).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                })
    return files


def check_counts(meta, dataset_name, user_expected):
    counts = (pd.crosstab(meta["split"], meta["label"])
              .reindex(index=list(SPLITS), columns=LABELS, fill_value=0))

    expected = dict(PROFILES.get(dataset_name, {}))
    for split, value in (user_expected or {}).items():
        if value is None:
            continue
        if expected.get(split, value) != value:
            raise ValueError(f"Conteo esperado incompatible con {dataset_name}: {split}")
        expected[split] = value

    for split in SPLITS:
        if not (counts.loc[split] > 0).all():
            raise ValueError(f"Todas las clases deben contener audios en {split}.")
        if split in expected and not (counts.loc[split] == expected[split]).all():
            raise ValueError(f"Conteos inesperados en {split}. Esperado por clase: "
                             f"{expected[split]}; observado: {counts.loc[split].to_dict()}")
    return counts, expected


def audit_dataset(dataset_dir, metadata_path=None, source_column="source_group", expected=None):
    dataset_dir = Path(dataset_dir).resolve()
    metadata_path = (Path(metadata_path).resolve() if metadata_path
                     else dataset_dir / "metadata.csv")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"No se encontró el metadata: {metadata_path}")

    meta, source_column = load_metadata(metadata_path, source_column)
    check_metadata(meta, source_column)

    files = scan_disk(dataset_dir)
    on_disk = {f["relative_path"] for f in files}
    listed = set(meta["relative_path"])
    if listed - on_disk:
        raise ValueError(f"Registros del metadata ausentes en disco: {sorted(listed - on_disk)[:10]}")
    if on_disk - listed:
        raise ValueError(f"WAV en disco ausentes del metadata: {sorted(on_disk - listed)[:10]}")

    counts, expected = check_counts(meta, dataset_dir.name, expected)

    columns = ["relative_path", "label", "split", "origin", source_column]
    if "parent_file" in meta.columns:
        columns.append("parent_file")
    records = meta[columns].sort_values("relative_path").to_dict(orient="records")

    metadata_sha = sha256_file(metadata_path)
    fingerprint = hashlib.sha256(json.dumps(
        {"metadata_sha256": metadata_sha, "files": files, "records": records},
        sort_keys=True, ensure_ascii=False,
    ).encode("utf-8")).hexdigest()

    return {
        "dataset_dir": str(dataset_dir),
        "metadata_path": str(metadata_path),
        "metadata_sha256": metadata_sha,
        "dataset_fingerprint": fingerprint,
        "source_group_column": source_column,
        "records": int(len(meta)),
        "counts_per_split_and_label": counts.to_dict(),
        "expected_counts": expected,
    }

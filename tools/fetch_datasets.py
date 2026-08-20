#!/usr/bin/env python3
"""
Phase 1 dataset acquisition. Sections 1.1 (ReID) and 1.4 (objects).

Credentials come from the ENVIRONMENT and are never printed, never written to
a file, and never passed on a command line where they would land in shell
history and `ps` output:

    KAGGLE_USERNAME / KAGGLE_KEY   or  ~/.kaggle/kaggle.json
    ROBOFLOW_API_KEY

Every download is skipped when its target already looks complete, so this is
safe to re-run after a partial failure without re-fetching 145 MB.

WHY FORKLIFT IS FETCHED NOW AND THEN LEFT ALONE
    Phase 9 has to demonstrate introducing a class the base model has never
    seen, in under one working day, with zero training steps. Downloading it
    now and refusing to touch it until then is what makes that demonstration
    honest rather than staged. Do not add it to edge/classes.yaml.

USAGE
    python tools/fetch_datasets.py            # everything
    python tools/fetch_datasets.py --reid     # Market-1501 only
    python tools/fetch_datasets.py --objects  # PPE + forklift only
    python tools/fetch_datasets.py --verify   # count what is already on disk
"""
import argparse
import collections
import os
import pathlib
import sys

ROOT = pathlib.Path(os.environ.get("VI_ROOT", pathlib.Path(__file__).resolve().parent.parent))
DATA = pathlib.Path(os.environ.get("DATA", ROOT / "data"))
DATASETS = DATA / "datasets"

MARKET_SLUG = "pengcw1/market-1501"
MARKET_DIR = DATASETS / "Market-1501-v15.09.15"
# The published split. Deviating from these means the Rank-1 you report is not
# comparable to any number in the literature, which is the entire reason for
# using a standard benchmark.
MARKET_EXPECTED = {"bounding_box_train": 12936, "query": 3368,
                   "bounding_box_test": 19732}

OBJECTS_DIR = DATASETS / "objects"
ROBOFLOW_JOBS = [
    # (workspace, project, version, local dir, why)
    ("roboflow-universe-projects", "construction-site-safety", 30, "ppe",
     "base 20-class PPE inventory: the classes every customer wants"),
    ("roboflow-100", "forklift-rzcxl", 2, "forklift",
     "HELD OUT for Phase 9. Do not add to the base model."),
]


def die(msg, code=2):
    print(f"\nERROR: {msg}", file=sys.stderr)
    sys.exit(code)


# --------------------------------------------------------------------------
def count_market(root=MARKET_DIR):
    """Counts and, more usefully, the identity/camera structure. The filename
    IS the label source: 0002_c1s1_000451_03.jpg is identity 0002, camera c1,
    sequence s1, frame 000451, box 03. Identity -1 is a distractor and must be
    excluded from evaluation or Rank-1 is meaningless."""
    if not root.exists():
        return None
    out = {}
    for split in ("bounding_box_train", "query", "bounding_box_test"):
        d = root / split
        if not d.exists():
            out[split] = None
            continue
        files = list(d.glob("*.jpg"))
        ids = {p.name.split("_")[0] for p in files}
        cams = {p.name.split("_")[1][:2] for p in files if "_" in p.name}
        out[split] = {"images": len(files), "identities": len(ids),
                      "cameras": sorted(cams),
                      "distractors": sum(1 for p in files
                                         if p.name.startswith("-1"))}
    return out


def fetch_market():
    if MARKET_DIR.exists() and (MARKET_DIR / "query").exists():
        n = len(list((MARKET_DIR / "query").glob("*.jpg")))
        if n >= MARKET_EXPECTED["query"]:
            print(f"  Market-1501 already present ({n} query images), skipping")
            return
        print(f"  Market-1501 present but incomplete ({n} query images), refetching")

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        die("kaggle is not installed.  pip install kaggle")
    except OSError as e:
        die(f"kaggle imported but could not authenticate: {e}")

    DATASETS.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()                      # env vars or ~/.kaggle/kaggle.json
    print(f"  downloading {MARKET_SLUG} -> {DATASETS} (~145 MB)")
    api.dataset_download_files(MARKET_SLUG, path=str(DATASETS), unzip=True, quiet=False)

    if not MARKET_DIR.exists():
        # Mirrors vary in how deeply they nest the archive.
        cands = [p for p in DATASETS.rglob("bounding_box_test") if p.is_dir()]
        if cands:
            found = cands[0].parent
            print(f"  archive nested differently; found split root at {found}")
            if found != MARKET_DIR:
                found.rename(MARKET_DIR)
        else:
            die(f"downloaded, but no bounding_box_test/ under {DATASETS}. "
                f"Inspect the extracted tree and move it to {MARKET_DIR}.")


# --------------------------------------------------------------------------
def count_objects(root=OBJECTS_DIR):
    out = {}
    for _, _, _, name, _ in ROBOFLOW_JOBS:
        d = root / name
        if not d.exists():
            out[name] = None
            continue
        yaml_path = d / "data.yaml"
        classes = None
        if yaml_path.exists():
            try:
                import yaml
                classes = (yaml.safe_load(yaml_path.read_text()) or {}).get("names")
            except Exception:
                classes = "unparseable"
        out[name] = {"data_yaml": yaml_path.exists(),
                     "images": len(list(d.rglob("*.jpg"))),
                     "labels": len(list(d.rglob("*.txt"))),
                     "classes": classes,
                     "n_classes": len(classes) if isinstance(classes, list) else None}
    return out


def fetch_objects():
    key = os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        die("ROBOFLOW_API_KEY is not set in this environment.")
    try:
        from roboflow import Roboflow
    except ImportError:
        die("roboflow is not installed.  pip install roboflow")

    OBJECTS_DIR.mkdir(parents=True, exist_ok=True)
    rf = Roboflow(api_key=key)
    cwd = pathlib.Path.cwd()
    os.chdir(OBJECTS_DIR)                   # the SDK writes relative to cwd
    try:
        for workspace, project, version, name, why in ROBOFLOW_JOBS:
            if (OBJECTS_DIR / name / "data.yaml").exists():
                print(f"  {name}: already present, skipping")
                continue
            print(f"  {name}: {workspace}/{project} v{version}  ({why})")
            rf.workspace(workspace).project(project).version(version) \
              .download("yolov8", location=name)
    finally:
        os.chdir(cwd)


# --------------------------------------------------------------------------
def report():
    print("\n" + "=" * 72)
    print("SECTION 1.1  Market-1501")
    print("=" * 72)
    m = count_market()
    if not m:
        print(f"  ABSENT at {MARKET_DIR}")
    else:
        for split, exp in MARKET_EXPECTED.items():
            s = m.get(split)
            if not s:
                print(f"  {split:22s} MISSING (expected {exp})")
                continue
            ok = "OK  " if s["images"] == exp else "DIFF"
            print(f"  {split:22s} {s['images']:6d} imgs (expect {exp:6d}) {ok}  "
                  f"{s['identities']:5d} ids  cams={s['cameras']}  "
                  f"distractors={s['distractors']}")

    print("\n" + "=" * 72)
    print("SECTION 1.4  object datasets")
    print("=" * 72)
    o = count_objects()
    for name, s in o.items():
        if not s:
            print(f"  {name:10s} ABSENT")
            continue
        print(f"  {name:10s} data.yaml={s['data_yaml']}  images={s['images']:6d}  "
              f"labels={s['labels']:6d}  classes={s['n_classes']}")
        if isinstance(s["classes"], list):
            print(f"             {s['classes']}")

    ok = (m and all(m.get(k) and m[k]["images"] == v
                    for k, v in MARKET_EXPECTED.items())
          and all(s and s["data_yaml"] for s in o.values()))
    print("\n  GATE 1 dataset checks: " + ("PASS" if ok else "NOT MET"))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reid", action="store_true")
    ap.add_argument("--objects", action="store_true")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    do_all = not (a.reid or a.objects or a.verify)

    if a.reid or do_all:
        print("\n--- Section 1.1: Market-1501 ---")
        fetch_market()
    if a.objects or do_all:
        print("\n--- Section 1.4: object datasets ---")
        fetch_objects()
    return report()


if __name__ == "__main__":
    sys.exit(main())

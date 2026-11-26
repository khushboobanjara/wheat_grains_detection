#!/usr/bin/env python3
"""
Train the wheat-grain quality model from the terminal.  The dashboard (server.py) then only PREDICTS.

1. Sort your photos into one folder per grain type. Each photo should show grains of ONE type:

       dataset/
         Sound/                *.jpg  *.png ...
         Shrivelled/
         Immature/
         Broken/
         Damaged_Discoloured/

   (`python train.py --make-folders` creates the empty folders. Folder names are matched loosely.)

2. Train:                 python train.py
3. Start the dashboard:   python server.py       -> Predict tab -> upload any new photo

Useful options
--------------
   python train.py --data my_photos                 another dataset folder
   python train.py --test-dir my_test_photos        score the model on photos it never saw (same folder layout)
   python train.py --preset "Pale / light wheat"    detection colour preset (see wheat_detector.PRESETS)
   python train.py --normalize photo                photos mix good and bad grains (see grain_trainer.py)
   python train.py --from-samples                   re-use grains labelled in the old in-browser trainer

Without --test-dir the score is a photo-level cross-validation: the model is always tested on photos it was
not trained on, because grains from the same photo look alike and would give an over-optimistic score.
"""

from __future__ import annotations

import argparse
import io
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

import grain_trainer as gt
from wheat_detector import PRESETS, Params, detect, resize_for_processing

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
BASE = Path(__file__).resolve().parent
LINE = "-" * 78


# --------------------------------------------------------------------------- #
# Dataset reading
# --------------------------------------------------------------------------- #
def read_image(path: Path) -> np.ndarray:
    """Decode an image (honours EXIF rotation, same as the dashboard)."""
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(path.read_bytes()))).convert("RGB")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def scan_dataset(root: Path) -> dict:
    """{class name: [photo paths]} from the sub-folders of `root`."""
    if not root.is_dir():
        sys.exit(f"\nDataset folder '{root}' not found.\n"
                 f"Create it with the grain-type sub-folders using:  python train.py --make-folders\n"
                 f"then copy your photos into them (one grain type per folder).")
    found: dict = {}
    for sub in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")):
        cls = gt.class_from_folder(sub.name)
        photos = sorted(p for p in sub.rglob("*") if p.suffix.lower() in IMG_EXT)
        if cls is None:
            print(f"  ! skipping folder '{sub.name}': not one of {', '.join(gt.CLASSES)}")
        elif photos:
            found.setdefault(cls, []).extend(photos)
    return found


def build_params(a: argparse.Namespace) -> Params:
    p = Params()
    if a.preset:
        h, s, v = (PRESETS[a.preset][k] for k in ("h", "s", "v"))
        p.h_min, p.h_max, p.s_min, p.s_max, p.v_min, p.v_max = h[0], h[1], s[0], s[1], v[0], v[1]
    if a.min_area is not None:
        p.min_area = a.min_area
    if a.max_area is not None:
        p.max_area = a.max_area
    if a.rel_min_pct is not None:
        p.rel_min_pct = a.rel_min_pct
    if a.no_split:
        p.split = False
    return p


def detect_photos(found: dict, params: Params, title: str) -> list:
    """Run the OpenCV detector on every photo. Returns one record per photo that contains grains."""
    records, total = [], sum(len(v) for v in found.values())
    print(f"\n{title}: detecting grains in {total} photos")
    print(LINE)
    k = 0
    for cls, photos in found.items():
        for path in photos:
            k += 1
            try:
                orig = read_image(path)
                work, scale = resize_for_processing(orig)
                grains = detect(work, params).grains
            except Exception as e:                                             # a bad file must not stop training
                print(f"  [{k:>3}/{total}] {cls:<20} {path.name:<32} unreadable ({type(e).__name__})")
                continue
            flag = "" if grains else "   <- NO GRAINS FOUND (check detection settings)"
            print(f"  [{k:>3}/{total}] {cls:<20} {path.name:<32} {len(grains):>4} grains{flag}")
            if grains:
                records.append({"cls": cls, "photo": f"{cls}/{path.name}", "grains": grains, "scale": scale,
                                "raw": gt.collect(orig, grains, scale)})
    return records


def feature_table(records: list, ref):
    """Stack the features of all photos. ref=None -> every photo is measured against its own median grain."""
    Xs, y, groups = [], [], []
    for r in records:
        rf = ref if ref else gt.photo_reference(r["raw"])
        Xs.append(gt.features_from(r["raw"], r["grains"], r["scale"], rf))
        y += [r["cls"]] * len(r["grains"])
        groups += [r["photo"]] * len(r["grains"])
    return (np.vstack(Xs) if Xs else np.zeros((0, len(gt.FEATURE_NAMES)))), y, groups


def make_reference(records: list) -> dict:
    """Median Sound grain of the training photos (all photos if no Sound photo exists)."""
    pool = [r for r in records if r["cls"] == "Sound"] or records
    return gt.merge_references([gt.photo_reference(r["raw"]) for r in pool])


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def metrics(y_true, y_pred, kind: str, label: str, n_photos: int, note: str = "") -> dict:
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
    labels = [c for c in gt.CLASSES if c in set(y_true) | set(y_pred)]
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return {"kind": kind, "label": label, "note": note, "n_grains": int(len(y_true)), "n_photos": int(n_photos),
            "accuracy": float(accuracy_score(y_true, y_pred)), "macro_f1": float(np.mean(f)),
            "per_class": {c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
                          for i, c in enumerate(labels)},
            "confusion": {"labels": labels, "matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist()}}


def cross_validate(X, y, groups, a) -> dict:
    """Photo-level cross-validation (falls back to grain-level when there are too few photos)."""
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_predict
    ya, ga = np.array(y), np.array(groups)
    photos_per_class = {c: len({g for g, lab in zip(groups, y) if lab == c}) for c in set(y)}
    n_photos, min_photos = len(set(groups)), min(photos_per_class.values())
    clf = gt.make_classifier(max(100, a.trees // 3), a.seed)                   # fewer trees: this is only a check
    if min_photos >= 2:
        k = int(min(5, min_photos))
        pred = cross_val_predict(clf, X, ya, groups=ga, cv=StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=a.seed))
        return metrics(ya, pred, "photo_cv", f"{k}-fold cross-validation, split by photo", n_photos)
    counts = Counter(y)
    k = int(min(5, min(counts.values())))
    if k < 2:
        return {}
    pred = cross_val_predict(clf, X, ya, cv=StratifiedKFold(n_splits=k, shuffle=True, random_state=a.seed))
    return metrics(ya, pred, "grain_cv", f"{k}-fold cross-validation, split by grain", n_photos,
                   "Some grain type has only one photo, so grains of the same photo were in both the training and the "
                   "test part. The real accuracy on new photos is probably LOWER. Add more photos per type.")


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def report_text(meta: dict, n_photos: int, seconds: float) -> str:
    ev = meta.get("evaluation") or {}
    out = [LINE, "TRAINING REPORT", LINE,
           f"Trained      : {meta['trained_at']}  ({seconds:.0f} s, {meta['trees']} trees)",
           f"Grains       : {meta['n_samples']} from {n_photos} photos",
           f"Normalisation: {meta['normalize']}" + ("  (fixed reference grain stored in the model)" if meta["normalize"] == "reference" else "  (per-photo median grain)"),
           "", f"{'Grain type':<22}{'training grains':>16}"]
    out += [f"  {c:<20}{meta['counts'].get(c, 0):>16}" for c in gt.CLASSES]
    if ev:
        out += ["", f"EVALUATION: {ev['label']}", f"  {ev['n_grains']} grains from {ev['n_photos']} photos",
                f"  accuracy {ev['accuracy'] * 100:.1f} %    macro F1 {ev['macro_f1'] * 100:.1f} %"]
        if ev.get("note"):
            out.append(f"  NOTE: {ev['note']}")
        out += ["", f"  {'Grain type':<22}{'precision':>10}{'recall':>9}{'F1':>8}{'grains':>8}"]
        out += [f"  {c:<22}{v['precision'] * 100:>9.1f}%{v['recall'] * 100:>8.1f}%{v['f1'] * 100:>7.1f}%{v['support']:>8}"
                for c, v in ev["per_class"].items()]
        labels = ev["confusion"]["labels"]
        out += ["", "  Confusion matrix (rows = true type, columns = predicted):",
                "  " + " " * 22 + "".join(f"{(c[:9] + '.') if len(c) > 9 else c:>11}" for c in labels)]
        out += ["  " + f"{c:<22}" + "".join(f"{v:>11}" for v in row) for c, row in zip(labels, ev["confusion"]["matrix"])]
    out += ["", "Most useful features: " + ", ".join(f"{n} ({v:.2f})" for n, v in meta["top_features"][:6]), LINE]
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args():
    ap = argparse.ArgumentParser(description="Train the grain-quality model (see the docstring at the top of train.py).",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--data", type=Path, default=BASE / "dataset", help="folder with one sub-folder per grain type")
    ap.add_argument("--test-dir", type=Path, help="optional folder (same layout) of photos used ONLY to score the model")
    ap.add_argument("--model", type=Path, default=gt.MODEL_FILE, help="where to save the trained model")
    ap.add_argument("--normalize", choices=["reference", "photo"], default="reference",
                    help="reference = fixed reference grain (photos hold one type each); photo = per-photo median")
    ap.add_argument("--preset", choices=list(PRESETS), help="detection colour preset")
    ap.add_argument("--min-area", type=int, help="smallest grain area in px (detection)")
    ap.add_argument("--max-area", type=int, help="largest grain area in px (detection)")
    ap.add_argument("--rel-min-pct", type=int, help="drop blobs below this %% of the median grain (0 keeps fragments)")
    ap.add_argument("--no-split", action="store_true", help="do not split touching grains")
    ap.add_argument("--trees", type=int, default=300, help="trees in the Random Forest")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-eval", action="store_true", help="skip the evaluation")
    ap.add_argument("--from-samples", action="store_true", help="train from training_data/samples.json (legacy labels)")
    ap.add_argument("--make-folders", action="store_true", help="create the empty dataset folders and exit")
    return ap.parse_args()


def main():
    a = parse_args()
    if a.make_folders:
        for cls in gt.CLASSES:
            (a.data / gt.FOLDER_NAMES[cls]).mkdir(parents=True, exist_ok=True)
        print(f"Created {a.data}/ with one folder per grain type. Put photos of ONE grain type in each folder,")
        print("then run:  python train.py")
        return
    t0 = time.time()
    params = build_params(a)
    detect_cfg = asdict(params)

    # ---- legacy path: labelled grains from the old in-browser trainer -------------------------------------
    if a.from_samples:
        if not gt.LEGACY_SAMPLES.is_file():
            sys.exit(f"\n{gt.LEGACY_SAMPLES} not found. --from-samples only works if you labelled grains with the "
                     f"old in-browser trainer on this computer.\nOtherwise sort photos into folders and run:  python train.py")
        X, y, sources = gt.load_legacy_samples()
        if not len(y):
            sys.exit("training_data/samples.json holds no labelled grains.")
        print(f"Loaded {len(y)} labelled grains from {len(set(sources))} photos (photo-normalised features).")
        records, ref, normalize, groups = [], None, "photo", sources
        extra = {"source": "legacy-samples"}
        test_X, test_y, test_photos = None, None, 0
    else:
        found = scan_dataset(a.data)
        if len(found) < 2:
            sys.exit(f"\nFound photos for {len(found)} grain type(s) in '{a.data}'. At least 2 types are needed.\n"
                     f"Expected sub-folders like: {', '.join(gt.FOLDER_NAMES.values())}")
        records = detect_photos(found, params, "Training photos")
        if not records:
            sys.exit("\nNo grains were detected in any photo. Open one in the dashboard (python server.py) and tune the "
                     "detection settings, then pass them here (--preset / --min-area / --rel-min-pct ...).")
        normalize = a.normalize
        ref = make_reference(records) if normalize == "reference" else None
        X, y, groups = feature_table(records, ref)
        extra = {"source": "terminal", "n_photos": len(records)}
        test_X = test_y = None
        test_photos = 0
        if a.test_dir and not a.no_eval:
            test_records = detect_photos(scan_dataset(a.test_dir), params, "Test photos")
            if test_records:
                test_X, test_y, tg = feature_table(test_records, ref)
                test_photos = len(test_records)
            else:
                print("  ! no grains found in the test folder; falling back to cross-validation")

    counts = Counter(y)
    print(f"\nTraining grains: " + ", ".join(f"{c} {counts[c]}" for c in gt.CLASSES if c in counts))
    thin = [c for c in counts if counts[c] < 30]
    if thin:
        print(f"  ! few grains for: {', '.join(thin)}. Results for these types will be unreliable; add more photos.")
    if len(counts) < len(gt.CLASSES):
        print(f"  ! no data for: {', '.join(c for c in gt.CLASSES if c not in counts)}. The model can never predict these types.")

    # ---- evaluation ---------------------------------------------------------------------------------------
    evaluation = None
    if not a.no_eval:
        print("\nEvaluating (this is the honest accuracy: the model never sees the photos it is scored on)...")
        try:
            if test_X is not None:
                clf = gt.make_classifier(a.trees, a.seed).fit(X, np.array(y))
                evaluation = metrics(np.array(test_y), clf.predict(test_X), "test_set",
                                     f"held-out test photos ({a.test_dir})", test_photos)
            else:
                evaluation = cross_validate(X, y, groups, a) or None
                if evaluation is None:
                    print("  ! too few photos/grains for an evaluation (need at least 2 per grain type).")
        except Exception as e:                                             # never block training on the check
            print(f"  ! evaluation skipped ({type(e).__name__}: {e})")

    # ---- final model on ALL training grains ---------------------------------------------------------------
    print(f"Training the final model on all {len(y)} grains...")
    model = gt.GrainModel(a.model)
    n_photos = extra.get("n_photos") or len(set(groups))
    meta = model.fit(X, y, ref=ref, normalize=normalize, detect=detect_cfg, evaluation=evaluation,
                     trees=a.trees, seed=a.seed, extra={**extra, "n_photos": n_photos})
    text = report_text(meta, n_photos, time.time() - t0)
    print("\n" + text)
    report_file = Path(a.model).with_name("training_report.txt")
    report_file.write_text(text + "\n", encoding="utf-8")
    print(f"\nModel saved  : {a.model}\nReport saved : {report_file}")
    print("\nNext:  python server.py   ->  open the Predict tab  ->  upload a new photo")


if __name__ == "__main__":
    main()

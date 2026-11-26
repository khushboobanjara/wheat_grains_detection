# 🌾 Wheat Grain Analysis Dashboard

A professional web dashboard (plain **HTML + CSS + JavaScript**, served by a small **Flask** backend) that
**detects, counts, measures and (optionally) grades wheat grains** from a photo. Detection is classical
computer vision (HSV segmentation, morphology, contours, watershed, shape filters) and runs 100 % locally with
OpenCV: no dataset and no paid service needed. Grain **quality grading** (Sound / Shrivelled / Immature / Broken /
Damaged-Discoloured) uses a model you **train once in the terminal** (`python train.py`); the dashboard then only
**predicts** new photos with it. The Groq classification and written report are **optional** and use the free Groq
Vision API only when you add a key and click the button.

> This version replaces the earlier Streamlit front end. The detection and Groq modules are unchanged.

## Quick start
**Windows:** double-click `run_windows.bat`   **macOS / Linux:** `./run_mac_linux.sh`

Or manually (Python 3.9+):
```bash
python -m venv venv
venv\Scripts\activate          # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
python server.py               # open http://localhost:8000
```
Use another port with `PORT=9000 python server.py` (Windows: `set PORT=9000`).

**Workflow:** 1) sort photos into folders  2) `python train.py` (terminal)  3) `python server.py`, open **Predict**, upload a new photo.

## Dashboard tour
| View | What you get |
|---|---|
| **Overview** | KPI cards (grain count, average size, size variation, processing time), view switcher (side-by-side / processed / original / mask), size-distribution histogram, summary statistics |
| **Detection** | Segmentation mask, active settings, sortable per-grain table with small/large outlier flag |
| **Predict** | Shows the model you trained in the terminal (date, grains, test accuracy, per-type scores, confusion matrix) and grades the uploaded photo grain by grain, with a close-up and the model's confidence per type |
| **Quality** | Full report table with the grain types (Sound / Shrivelled / Immature / Broken / Damaged-Discoloured) as rows: count, % by count, % by area, average size, confidence, review flags. Source: your trained model, or optional Groq |
| **AI report** | *(optional, Groq)* Written visual analysis of colour, shape, uniformity and defects |
| **Export** | Processed PNG, mask PNG, grain CSV, classification PDF + CSVs, AI report (MD/TXT) |
| **Batch** | Appears when you upload several images: grains per image, totals, chart, CSV |

The right-hand **Detection settings** panel holds the colour preset, HSV range, clean-up, shape filters,
touching-grain splitting and display options, plus **Reset all settings**. Results update live as you move a slider.
Other touches: drag-and-drop upload, click-to-enlarge images, light/dark theme, responsive layout.

## Train your own model in the terminal (no API key)
Training happens **outside the browser**. The dashboard never trains; it loads `models/grain_model.joblib` and predicts.

**1. Put your photos in one folder per grain type.** Each photo should show grains of **one** type only:
```
dataset/
  Sound/                 photo1.jpg  photo2.jpg ...
  Shrivelled/
  Immature/
  Broken/
  Damaged_Discoloured/
```
`python train.py --make-folders` creates the empty folders. Folder names are matched loosely (case, `_`, `-`, "Shriveled" all work). Use several photos per type, ideally 5+ and 100+ grains per type, taken at the **same distance and in similar light**.

**2. Train.**
```bash
python train.py
```
It detects the grains with OpenCV, builds the features, trains a Random Forest and prints a report: accuracy, per-type precision / recall, confusion matrix. Accuracy is measured on photos the model did **not** train on (photo-level cross-validation), because grains from the same photo look alike and would give a flattering score. Saved to `models/grain_model.joblib` and `models/training_report.txt`.

**3. Predict.** `python server.py` → **Predict** tab → upload any new photo. The **Quality** tab then fills in the grain-type table automatically. After re-training, the server picks up the new model by itself (or press *Reload model*).

Useful options:
| Command | Purpose |
|---|---|
| `python train.py --test-dir dataset_test` | Score the model on separate photos (same folder layout) instead of cross-validation: the most honest number for your report |
| `python train.py --data my_photos` | Use another dataset folder |
| `python train.py --preset "Pale / light wheat"` | Detection colour preset (also `--min-area`, `--max-area`, `--rel-min-pct`, `--no-split`) |
| `python train.py --normalize photo` | For photos that mix good and bad grains: judge sizes against each photo's own median grain instead of a fixed reference |
| `python train.py --from-samples` | Re-use grains you labelled earlier with the old in-browser trainer (`training_data/samples.json`) |
| `python train.py --trees 500 --seed 1` | Random Forest size / reproducibility |

Check that every photo reports a sensible grain count while training (a photo with `NO GRAINS FOUND` needs different detection settings). Broken fragments can be filtered out by the default shape filters; if that happens try `--rel-min-pct 0` and a smaller `--min-area`.

## Optional: Groq key
1. Create a free key at https://console.groq.com/keys
2. Copy `.env.example` to `.env` and set `GROQ_API_KEY=...`
3. Reload the page. The sidebar status turns green.

The key stays on the server in `.env` (git-ignored) and is never sent to the browser. Images go to Groq only when
you click *Classify grains with Groq* or *Generate AI report*. Never share or commit your `.env`.

## Tuning cheat sheet
| Problem | Fix |
|---|---|
| Grains missing | Check the mask (Detection view). Widen Hue, lower min Saturation, or try another preset |
| Background / text detected | Raise min Saturation, narrow Hue, tighten area and aspect filters |
| Touching grains counted as one | Lower *Split sensitivity* (e.g. 0.4) |
| One grain split in two | Raise *Split sensitivity* (e.g. 0.65) or *Min size vs median* |
| Specks counted | Raise minimum area or *Min size vs median* |
| Pale wheat on white | Use the *Pale / light wheat* preset or enable Otsu auto-tune |

## Files
| File | Purpose |
|---|---|
| `server.py` | Flask backend: upload, detection, batch, model prediction, Groq endpoints, downloads |
| `static/index.html` | Page shell (sidebar, top bar, settings panel) |
| `static/style.css` | Design system, light/dark themes, responsive rules |
| `static/app.js` | Dashboard logic, charts (inline SVG), tables, API calls. No external libraries |
| `wheat_detector.py` | Detection pipeline (unchanged) |
| `train.py` | **Terminal training**: reads `dataset/`, evaluates, saves the model |
| `grain_trainer.py` | Feature extraction + model wrapper (load / predict) used by `train.py` and `server.py` |
| `groq_analyzer.py`, `grain_report.py` | Optional Groq classification, report, PDF/CSV (unchanged) |

## Limitations
Colour-based detection depends on lighting and background. Heavily stacked grains, strong colour casts or a
background the same colour as the wheat need manual slider tuning. Measurements are in pixels (no physical
calibration). The trained model's confidence is the share of decision trees that voted for a type, not a calibrated probability, and it only knows what your training photos showed (other camera, lighting or wheat variety → re-train). Groq confidence values are the model's own estimate, not a measured accuracy. Uploaded images are
kept in server memory only (the 40 most recent) and are lost when the server restarts.

# 🌾 Wheat Grain Analysis Dashboard

A professional Streamlit dashboard that **detects, counts, measures and (optionally) grades wheat grains**
from a photo. Detection is classical computer vision (HSV segmentation, morphology, contours, watershed,
shape filters) and runs 100 % locally with OpenCV: no dataset, no training, no paid service.
Grain classification and the written report are **optional** and use the free Groq Vision API only when you
add a key and click the button.

## Quick start
**Windows:** double-click `run_windows.bat`   **macOS / Linux:** `./run_mac_linux.sh`

Or manually (Python 3.9+):
```bash
python -m venv venv
venv\Scripts\activate          # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py           # opens http://localhost:8501
```

## Dashboard tour
| Tab | What you get |
|---|---|
| **📊 Overview** | KPI cards (grain count, avg length/width, size variation, processing time), view switcher (side-by-side / processed / original / mask), size-distribution histogram, summary statistics |
| **🔍 Detection** | Segmentation mask, active settings, sortable per-grain table (length, width, aspect, solidity, area, small/large outlier flag) |
| **🔬 Quality classification** | *(optional, Groq)* Sound / Shrivelled / Immature / Broken / Damaged-Discoloured per grain, charts, manual-inspection flags, observations, detailed report |
| **🧠 AI report** | *(optional, Groq)* Written visual analysis of colour, shape, uniformity and defects |
| **📦 Export** | Processed PNG, mask PNG, grain CSV, classification PDF + CSVs, AI report (MD/TXT) |
| **🗂 Batch** | Appears when you upload several images: grains per image, totals, chart, CSV |

The sidebar holds all detection settings (colour preset, HSV range, clean-up, shape filters, touching-grain
splitting, display options) plus a **Reset all settings** button.

## Optional: Groq key
1. Create a free key at https://console.groq.com/keys
2. Copy `.env.example` to `.env` and set `GROQ_API_KEY=...`
3. Reload the page. The header chip turns green.

The key stays in `.env` (git-ignored) and is never shown in the UI. Images are sent to Groq only when you click
*Classify grains with Groq* or *Generate AI report*. Never share or commit your `.env`.

## Tuning cheat sheet
| Problem | Fix |
|---|---|
| Grains missing | Check the mask (Detection tab). Widen Hue, lower min Saturation, or try another preset |
| Background / text detected | Raise min Saturation, narrow Hue, tighten area and aspect filters |
| Touching grains counted as one | Lower *Split sensitivity* (e.g. 0.4) |
| One grain split in two | Raise *Split sensitivity* (e.g. 0.65) or *Min size vs median* |
| Specks counted | Raise minimum area or *Min size vs median* |
| Pale wheat on white | Use the *Pale / light wheat* preset or enable Otsu auto-tune |

## Files
| File | Purpose |
|---|---|
| `app.py` | Dashboard layout, tabs, sidebar |
| `ui_theme.py` | CSS theme and reusable components (header, KPI cards, callouts) |
| `wheat_detector.py` | Detection pipeline (unchanged) |
| `groq_analyzer.py`, `grain_report.py` | Optional Groq classification, report, PDF/CSV (unchanged) |
| `.streamlit/config.toml` | Colour theme, upload limit (50 MB), telemetry off |

## Limitations
Colour-based detection depends on lighting and background. Heavily stacked grains, strong colour casts or a
background the same colour as the wheat need manual slider tuning. Measurements are in pixels (no physical
calibration). Groq confidence values are the model's own estimate, not a measured accuracy.

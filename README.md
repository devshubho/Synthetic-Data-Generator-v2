# Project Synthesis (Synthetic Data Generator v2)

Streamlit app that generates high-quality synthetic tabular data from built-in templates or from your own sample upload.

**Version:** 2.3.9  
**UI name:** Project Synthesis / Project Synthesi

## Features

- **Template generation** — personal/customer, sales, employee, time series, logs, metrics, IoT, healthcare, finance, and more
- **Custom generation (upload sample)** — Faker-driven synthesis that preserves schema and closed enums from your CSV/Excel/JSON sample
- **India-first defaults** — `en_IN` locale, Indian cities/phones, and format helpers for Aadhaar, PAN, IFSC, UPI, pincode, GSTIN
- **Semantic column roles** — person names (including doctor), emails (name-coherent), cities, IDs, dates, products, couriers/vendors as closed labels
- **Quality report** — role-aware score (completeness, ID uniqueness, date diversity, open-field diversity, enum fidelity, name↔email coherence, numeric fidelity)
- **Export** — CSV, JSON, Parquet, Excel
- **Dashboard** — Plotly quality visualization and generation history

## Requirements

- Python 3.10+ (tested with newer interpreters; pinned older wheels in historical notes may fail on 3.14)
- See [`requirements.txt`](requirements.txt)

## Quick start

```powershell
cd Synthetic-Data-Generator-v2
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run src/app.py
```

Open **http://localhost:8501**

Alternative entry (adds `src` to path):

```powershell
python run.py
```

## Custom (upload) workflow

1. **Upload Sample** — provide a CSV, Excel, or JSON file with at least 2 rows  
2. **Generate** — choose *User-Defined (Upload Sample)*, set record count, generate  
3. **Review** — preview data and the quality breakdown (overall score + component metrics)  
4. **Export** — download in your preferred format  

### What custom generation does

| Column kind | Behavior |
|-------------|----------|
| Person-like (`name`, `doctor`, `employee_name`, …) | Novel Faker names (`en_IN` when India context is detected) |
| Email | Derived from the same person so local-part matches the name |
| City / phone | Novel realistic values; India-aware when seed looks Indian |
| Closed labels (`status`, `courier`, `department`, …) | Resampled from seed vocabulary only (no invented jargon) |
| IDs | Unique minted IDs following seed prefix/width patterns |
| Dates | Sampled across seed min–max range |
| Product groups | Jointly sampled (`product_name` + `category` + `unit_price`) |

Privacy anonymization is **skipped** for custom synthetic output so name↔email coherence is preserved (values are already synthetic).

## Project layout

```
src/
  app.py                 # Streamlit UI
  config.py              # Limits, templates, export formats
  generators/
    custom.py            # Sample-based CustomGenerator
    template.py          # Built-in template generator
    factory.py
  engine/
    pipeline.py          # End-to-end generation workflow
    data_processor.py
    validator.py
  analytics/
    quality_report.py    # Role-aware quality scoring
  privacy/               # Anonymizer (templates / optional paths)
  export/                # CSV, JSON, Parquet, Excel
  visualization/         # Dashboard
  database/              # Local generation history
tests/                   # Generator and analytics tests
```

## Quality score (custom path)

`overall_score` averages metrics that matter for synthetic data:

- Completeness  
- ID uniqueness  
- Date diversity  
- Open-field diversity (names, cities, emails, … — not closed enums)  
- Enum fidelity vs seed (when a sample is available)  
- Name↔email coherence (when both exist)  
- Numeric fidelity vs seed  

Legitimate low-cardinality fields like `payment_method` or `courier` no longer drag the score down.

## Development

```powershell
.\.venv\Scripts\Activate.ps1
# run a quick custom generation smoke test from repo root
python -c "import sys; sys.path.insert(0,'src'); ..."
```

Suggested: do not commit `logs/`, `datasets/history.db`, or `__pycache__/` — they are local runtime artifacts.

## License

Use according to your repository / organization policy.

# 🇪🇺 EU Parliamentary Speeches — NLP Data Engineering Pipeline

A multilingual data engineering pipeline that ingests, processes, and analyses EU Parliamentary speeches from the Europarl corpus using NLP techniques. Built with Python, spaCy, PostgreSQL, MongoDB, and Streamlit.

🚀 **Live Demo:** https://europarl-nlp-pipeline-i8h56zc47zgt39wgfqfxmt.streamlit.app/

---

## Prerequisites

Before running the project, make sure you have the following installed on your machine:

- Python 3.11 or higher
- Docker Desktop (for PostgreSQL, MongoDB, and Redis)
- Git

---

## 1. Clone the Repository

```bash
git clone https://github.com/Osron01/europarl-nlp-pipeline.git
cd europarl-nlp-pipeline
```

---

## 2. Set Up a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

---

## 3. Install Dependencies

```bash
pip3 install pandas numpy spacy streamlit plotly sqlalchemy psycopg2-binary pymongo redis python-dotenv langdetect
```

Then download the spaCy language model:

```bash
python3 -m spacy download en_core_web_sm
```

---

## 4. Download the Dataset

Go to: https://www.statmt.org/europarl/v10/

Download the following files and save them to `data/raw/europarl/`:

| File | Language Pair |
|------|--------------|
| `europarl-v10.de-en.tsv.gz` | German - English |
| `europarl-v10.fr-en.tsv.gz` | French - English |
| `europarl-v10.es-en.tsv.gz` | Spanish - English |
| `europarl-v10.bg-en.tsv.gz` | Bulgarian - English |

Your folder structure should look like this:

```
data/
└── raw/
    └── europarl/
        ├── europarl-v10.de-en.tsv.gz
        ├── europarl-v10.fr-en.tsv.gz
        ├── europarl-v10.es-en.tsv.gz
        └── europarl-v10.bg-en.tsv.gz
```

---

## 5. Start the Database Services

Make sure Docker Desktop is running, then:

```bash
docker-compose up -d
```

This starts PostgreSQL (port 5432), MongoDB (port 27017), and Redis (port 6379).

Verify services are running:

```bash
docker-compose ps
```

---

## 6. Run the Pipeline

Run each step in order:

**Step 1 — Ingest data into bronze layer:**
```bash
python3 src/ingestion/europarl_ingestion.py
```

**Step 2 — Run NLP processing into silver layer:**
```bash
python3 src/nlp/nlp_processor.py
```

**Step 3 — Aggregate data into gold layer:**
```bash
python3 src/processing/data_processor.py
```

---

## 7. Launch the Dashboard

```bash
python3 -m streamlit run dashboard.py
```

Open your browser and go to: **http://localhost:8501**

---

## Project Structure

```
europarl-nlp-pipeline/
├── data/
│   ├── raw/europarl/        # Downloaded source files
│   ├── bronze/              # Ingested JSON files
│   ├── silver/              # NLP-enriched JSON files
│   └── gold/                # Aggregated analytics JSON files
├── src/
│   ├── ingestion/
│   │   └── europarl_ingestion.py
│   ├── nlp/
│   │   └── nlp_processor.py
│   └── processing/
│       └── data_processor.py
├── dashboard.py
├── docker-compose.yml
└── requirements.txt
```

---

## Troubleshooting

**`python` command not found on Mac:**
Use `python3` instead of `python` for all commands.

**`streamlit` command not found:**
Use `python3 -m streamlit run dashboard.py` instead.

**Docker services not starting:**
Make sure Docker Desktop is open and running before running `docker-compose up -d`.

**spaCy model not found:**
Run `python3 -m spacy download en_core_web_sm` again inside your activated virtual environment.

**No gold data found error in dashboard:**
Make sure you have run all three pipeline steps (ingestion, NLP, processing) before launching the dashboard.

---

## Dataset

- **Name:** Europarl Parallel Corpus v10
- **Source:** https://www.statmt.org/europarl/v10/
- **License:** Free for non-commercial use
- **Languages:** German, French, Spanish, Bulgarian (translated to English)
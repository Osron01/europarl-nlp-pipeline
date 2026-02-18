import gzip
import json
import logging
import os
from datetime import datetime
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class EuroparlIngestion:

    LANGUAGE_PAIRS = {
        "de-en": "German-English",
        "fr-en": "French-English",
        "es-en": "Spanish-English",
        "bg-en": "Bulgarian-English",
    }

    def __init__(self, raw_data_path: str = "data/raw/europarl", bronze_path: str = "data/bronze"):
        self.raw_data_path = Path(raw_data_path)
        self.bronze_path = Path(bronze_path)
        self.bronze_path.mkdir(parents=True, exist_ok=True)

    def _find_local_files(self) -> dict:
        found = {}
        for lang_pair in self.LANGUAGE_PAIRS:
            file_path = self.raw_data_path / f"europarl-v10.{lang_pair}.tsv.gz"
            if file_path.exists():
                found[lang_pair] = file_path
                logger.info(f"Found: {file_path}")
            else:
                logger.warning(f"Not found: {file_path}")
        return found

    def _load_tsv_gz(self, file_path: Path, lang_pair: str, sample_size: int = 5000) -> pd.DataFrame:
        logger.info(f"Loading {lang_pair} from {file_path}")
        rows = []
        source_lang, target_lang = lang_pair.split("-")

        with gzip.open(file_path, "rt", encoding="utf-8", errors="ignore") as f:
            for i, line in enumerate(f):
                if i >= sample_size:
                    break
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    source_text = parts[0].strip()
                    target_text = parts[1].strip()
                    if source_text and target_text:
                        rows.append({
                        "source_lang": source_lang,
                        "target_lang": target_lang,
                        "lang_pair": lang_pair,
                        "source_text": parts[0].strip(),
                        "target_text": parts[1].strip(),
                        "line_id": i,
                        "ingested_at": datetime.utcnow().isoformat(),
                    })

        df = pd.DataFrame(rows)
        logger.info(f"Loaded {len(df)} records for {lang_pair}")
        return df

    def _save_to_bronze(self, df: pd.DataFrame, lang_pair: str) -> str:
        date_str = datetime.utcnow().strftime("%Y%m%d")
        output_file = self.bronze_path / f"europarl_{lang_pair}_{date_str}.json"
        records = df.to_dict(orient="records")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved {len(records)} records to {output_file}")
        return str(output_file)

    def run(self, sample_size: int = 5000) -> dict:
        logger.info("Starting Europarl ingestion")
        local_files = self._find_local_files()

        if not local_files:
            raise FileNotFoundError(f"No Europarl files found in {self.raw_data_path}")

        results = {}
        for lang_pair, file_path in local_files.items():
            df = self._load_tsv_gz(file_path, lang_pair, sample_size)
            if not df.empty:
                output_file = self._save_to_bronze(df, lang_pair)
                results[lang_pair] = {
                    "records_loaded": len(df),
                    "output_file": output_file,
                    "lang_pair": lang_pair,
                    "language_name": self.LANGUAGE_PAIRS[lang_pair],
                }

        logger.info(f"Ingestion complete. Processed {len(results)} language pairs.")
        return results


if __name__ == "__main__":
    ingestion = EuroparlIngestion()
    results = ingestion.run(sample_size=5000)
    for lang_pair, info in results.items():
        print(f"{info['language_name']}: {info['records_loaded']} records → {info['output_file']}")
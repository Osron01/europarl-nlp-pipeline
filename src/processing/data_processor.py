import json
import logging
from collections import Counter
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class DataProcessor:

    def __init__(self, silver_path: str = "data/silver", gold_path: str = "data/gold"):
        self.silver_path = Path(silver_path)
        self.gold_path = Path(gold_path)
        self.gold_path.mkdir(parents=True, exist_ok=True)

    def _load_silver_file(self, file_path: Path) -> list:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _compute_sentiment_distribution(self, records: list) -> dict:
        counts = Counter(r["nlp"]["sentiment"]["label"] for r in records if "nlp" in r)
        total = sum(counts.values())
        return {
            "positive": counts.get("positive", 0),
            "neutral": counts.get("neutral", 0),
            "negative": counts.get("negative", 0),
            "total": total,
            "positive_pct": round(counts.get("positive", 0) / total * 100, 1) if total else 0,
            "neutral_pct": round(counts.get("neutral", 0) / total * 100, 1) if total else 0,
            "negative_pct": round(counts.get("negative", 0) / total * 100, 1) if total else 0,
        }

    def _compute_top_entities(self, records: list, top_n: int = 20) -> list:
        entity_counter = Counter()
        for r in records:
            if "nlp" in r:
                for ent in r["nlp"].get("entities", []):
                    entity_counter[(ent["text"], ent["label"])] += 1
        return [
            {"text": text, "label": label, "count": count}
            for (text, label), count in entity_counter.most_common(top_n)
        ]

    def _compute_top_keywords(self, records: list, top_n: int = 30) -> list:
        keyword_counter = Counter()
        for r in records:
            if "nlp" in r:
                for kw in r["nlp"].get("keywords", []):
                    keyword_counter[kw] += 1
        return [
            {"keyword": kw, "count": count}
            for kw, count in keyword_counter.most_common(top_n)
        ]

    def _compute_avg_sentiment_score(self, records: list) -> float:
        scores = [r["nlp"]["sentiment"]["score"] for r in records if "nlp" in r]
        return round(sum(scores) / len(scores), 3) if scores else 0.0

    def _process_lang_pair(self, lang_pair: str, records: list) -> dict:
        return {
            "lang_pair": lang_pair,
            "total_records": len(records),
            "sentiment_distribution": self._compute_sentiment_distribution(records),
            "avg_sentiment_score": self._compute_avg_sentiment_score(records),
            "top_entities": self._compute_top_entities(records),
            "top_keywords": self._compute_top_keywords(records),
            "processed_at": datetime.now().isoformat(),
        }

    def _save_to_gold(self, data: dict, lang_pair: str) -> str:
        date_str = datetime.now().strftime("%Y%m%d")
        output_file = self.gold_path / f"europarl_{lang_pair}_gold_{date_str}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved gold data to {output_file}")
        return str(output_file)

    def _save_summary(self, all_results: dict) -> str:
        summary = {
            "generated_at": datetime.now().isoformat(),
            "language_pairs": list(all_results.keys()),
            "total_records_processed": sum(v["total_records"] for v in all_results.values()),
            "sentiment_by_lang": {
                lp: data["sentiment_distribution"]
                for lp, data in all_results.items()
            },
            "avg_sentiment_by_lang": {
                lp: data["avg_sentiment_score"]
                for lp, data in all_results.items()
            },
        }
        output_file = self.gold_path / "summary.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved summary to {output_file}")
        return str(output_file)

    def run(self) -> dict:
        logger.info("Starting data processing (silver -> gold)")
        silver_files = list(self.silver_path.glob("europarl_*_nlp_*.json"))

        if not silver_files:
            raise FileNotFoundError(f"No silver files found in {self.silver_path}")

        all_results = {}
        for silver_file in silver_files:
            parts = silver_file.stem.split("_")
            lang_pair = parts[1] + "-" + parts[2]
            logger.info(f"Processing {lang_pair} from {silver_file.name}")

            records = self._load_silver_file(silver_file)
            gold_data = self._process_lang_pair(lang_pair, records)
            output_file = self._save_to_gold(gold_data, lang_pair)
            all_results[lang_pair] = gold_data
            logger.info(f"  {lang_pair}: {gold_data['total_records']} records aggregated")

        self._save_summary(all_results)
        logger.info("Data processing complete.")
        return all_results


if __name__ == "__main__":
    processor = DataProcessor()
    results = processor.run()
    for lang_pair, data in results.items():
        dist = data["sentiment_distribution"]
        print(f"{lang_pair}: {data['total_records']} records | "
              f"pos={dist['positive_pct']}% neu={dist['neutral_pct']}% neg={dist['negative_pct']}%")
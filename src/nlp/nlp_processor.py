import json
import logging
from datetime import datetime
from pathlib import Path

import spacy

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class NLPProcessor:

    SPACY_MODELS = {
        "en": "en_core_web_sm",
        "de": "de_core_news_sm",
        "fr": "fr_core_news_sm",
        "es": "es_core_news_sm",
        "bg": "en_core_web_sm",
    }

    POSITIVE_WORDS = {
        "good", "great", "excellent", "support", "agree", "positive", "success",
        "improve", "benefit", "progress", "cooperation", "welcome", "strong",
        "effective", "important", "growth", "opportunity", "solution", "achieve",
    }

    NEGATIVE_WORDS = {
        "bad", "poor", "reject", "oppose", "negative", "fail", "problem", "crisis",
        "concern", "risk", "threat", "disagree", "difficult", "against", "lack",
        "weak", "failure", "issue", "conflict", "dangerous",
    }

    def __init__(self, bronze_path: str = "data/bronze", silver_path: str = "data/silver"):
        self.bronze_path = Path(bronze_path)
        self.silver_path = Path(silver_path)
        self.silver_path.mkdir(parents=True, exist_ok=True)
        self.nlp_models = {}

    def _load_model(self, lang: str) -> spacy.Language:
        if lang not in self.nlp_models:
            model_name = self.SPACY_MODELS.get(lang, "en_core_web_sm")
            try:
                self.nlp_models[lang] = spacy.load(model_name)
                logger.info(f"Loaded spaCy model: {model_name}")
            except OSError:
                logger.warning(f"Model {model_name} not found, falling back to en_core_web_sm")
                self.nlp_models[lang] = spacy.load("en_core_web_sm")
        return self.nlp_models[lang]

    def _analyze_sentiment(self, text: str) -> dict:
        tokens = set(text.lower().split())
        positive_hits = tokens & self.POSITIVE_WORDS
        negative_hits = tokens & self.NEGATIVE_WORDS
        pos_score = len(positive_hits)
        neg_score = len(negative_hits)

        if pos_score > neg_score:
            label = "positive"
            score = round(pos_score / (pos_score + neg_score + 1), 3)
        elif neg_score > pos_score:
            label = "negative"
            score = round(-neg_score / (pos_score + neg_score + 1), 3)
        else:
            label = "neutral"
            score = 0.0

        return {"label": label, "score": score, "positive_hits": list(positive_hits), "negative_hits": list(negative_hits)}

    def _extract_entities(self, doc: spacy.tokens.Doc) -> list:
        return [
            {"text": ent.text, "label": ent.label_}
            for ent in doc.ents
            if ent.label_ in {"PERSON", "ORG", "GPE", "LOC", "NORP", "EVENT"}
        ]

    def _extract_keywords(self, doc: spacy.tokens.Doc) -> list:
        return list({
            token.lemma_.lower()
            for token in doc
            if not token.is_stop and not token.is_punct and token.is_alpha and len(token.text) > 3
        })[:20]

    def _process_record(self, record: dict) -> dict:
        target_text = record.get("target_text", "")
        source_lang = record.get("source_lang", "en")

        nlp = self._load_model("en")
        doc = nlp(target_text[:500])

        sentiment = self._analyze_sentiment(target_text)
        entities = self._extract_entities(doc)
        keywords = self._extract_keywords(doc)

        return {
            **record,
            "nlp": {
                "sentiment": sentiment,
                "entities": entities,
                "keywords": keywords,
                "token_count": len(doc),
                "processed_at": datetime.now().isoformat(),
            }
        }

    def _load_bronze_file(self, file_path: Path) -> list:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save_to_silver(self, records: list, lang_pair: str) -> str:
        date_str = datetime.now().strftime("%Y%m%d")
        output_file = self.silver_path / f"europarl_{lang_pair}_nlp_{date_str}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved {len(records)} NLP-processed records to {output_file}")
        return str(output_file)

    def run(self, sample_size: int = 500) -> dict:
        logger.info("Starting NLP processing")
        bronze_files = list(self.bronze_path.glob("europarl_*.json"))

        if not bronze_files:
            raise FileNotFoundError(f"No bronze files found in {self.bronze_path}")

        results = {}
        for bronze_file in bronze_files:
            lang_pair = bronze_file.stem.split("_")[1] + "-" + bronze_file.stem.split("_")[2]
            logger.info(f"Processing {lang_pair} from {bronze_file.name}")

            records = self._load_bronze_file(bronze_file)
            sample = records[:sample_size]

            processed = []
            for i, record in enumerate(sample):
                processed.append(self._process_record(record))
                if (i + 1) % 100 == 0:
                    logger.info(f"  Processed {i + 1}/{len(sample)} records")

            output_file = self._save_to_silver(processed, lang_pair)
            results[lang_pair] = {
                "records_processed": len(processed),
                "output_file": output_file,
            }

        logger.info(f"NLP processing complete. Processed {len(results)} language pairs.")
        return results


if __name__ == "__main__":
    processor = NLPProcessor()
    results = processor.run(sample_size=500)
    for lang_pair, info in results.items():
        print(f"{lang_pair}: {info['records_processed']} records → {info['output_file']}")
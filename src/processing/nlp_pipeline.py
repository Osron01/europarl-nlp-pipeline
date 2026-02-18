import json
import os
from pathlib import Path
from typing import Dict, List
import logging
import sys

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.nlp.spacy_nlp import SpacyNLPProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NLPPipeline:
    """Pipeline to process meetings with NLP and save to silver layer"""

    def __init__(self, bronze_path: str = "data/bronze", silver_path: str = "data/silver"):
        self.bronze_path = bronze_path
        self.silver_path = silver_path
        self.nlp_processor = SpacyNLPProcessor()

        # Create silver directory
        os.makedirs(silver_path, exist_ok=True)

    def process_all_meetings(self) -> List[Dict]:
        """Process all meetings in bronze layer with NLP"""
        bronze_files = list(Path(self.bronze_path).glob("*.json"))
        processed_meetings = []

        if not bronze_files:
            logger.warning("No bronze files found. Run data ingestion first.")
            return []

        logger.info(f"Processing {len(bronze_files)} meetings with NLP...")

        for file_path in bronze_files:
            try:
                # Load meeting data
                with open(file_path, 'r', encoding='utf-8') as f:
                    meeting = json.load(f)

                # Process with NLP
                transcript = meeting.get('transcript', '')
                if transcript and len(transcript.strip()) > 0:
                    logger.info(f"Processing NLP for {meeting['id']}...")
                    nlp_results = self.nlp_processor.process_text(transcript)
                    meeting['nlp_analysis'] = nlp_results

                    # Add some metadata about processing
                    meeting['processed_at'] = meeting.get('fetched_at')
                    meeting['transcript_length'] = len(transcript)
                    meeting['word_count'] = len(transcript.split())
                else:
                    logger.warning(f"No transcript found for {meeting['id']}")
                    meeting['nlp_analysis'] = {}

                # Save to silver layer
                silver_filename = f"processed_{meeting['id']}.json"
                silver_path = os.path.join(self.silver_path, silver_filename)

                with open(silver_path, 'w', encoding='utf-8') as f:
                    json.dump(meeting, f, indent=2, ensure_ascii=False)

                processed_meetings.append(meeting)
                logger.info(f"✅ Processed {meeting['id']} from {meeting.get('city', 'unknown')}")

            except Exception as e:
                logger.error(f"❌ Error processing {file_path}: {e}")

        logger.info(f"✅ Completed processing {len(processed_meetings)} meetings")
        return processed_meetings

    def get_processing_summary(self) -> Dict:
        """Get summary of processed meetings"""
        silver_files = list(Path(self.silver_path).glob("*.json"))

        if not silver_files:
            return {"error": "No processed files found"}

        total_meetings = len(silver_files)
        cities = set()
        total_entities = 0
        sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
        topic_totals = {}

        for file_path in silver_files:
            with open(file_path, 'r', encoding='utf-8') as f:
                meeting = json.load(f)

            cities.add(meeting.get('city', 'unknown'))

            nlp_analysis = meeting.get('nlp_analysis', {})

            # Count entities
            entities = nlp_analysis.get('entities', [])
            total_entities += len(entities)

            # Count sentiment
            sentiment = nlp_analysis.get('sentiment', {}).get('sentiment', 'neutral')
            sentiment_counts[sentiment] += 1

            # Sum topics
            topics = nlp_analysis.get('topics', {})
            for topic, score in topics.items():
                topic_totals[topic] = topic_totals.get(topic, 0) + score

        return {
            "total_meetings": total_meetings,
            "cities": list(cities),
            "total_entities": total_entities,
            "sentiment_distribution": sentiment_counts,
            "top_topics": dict(sorted(topic_totals.items(), key=lambda x: x[1], reverse=True)[:5])
        }


if __name__ == "__main__":
    # Make sure data directories exist
    os.makedirs("data/bronze", exist_ok=True)
    os.makedirs("data/silver", exist_ok=True)

    # Create and run pipeline
    pipeline = NLPPipeline()

    # Process all meetings
    results = pipeline.process_all_meetings()

    if results:
        print(f"\n✅ Processed {len(results)} meetings successfully")

        # Show processing summary
        summary = pipeline.get_processing_summary()
        print(f"\n📊 Processing Summary:")
        print(f"   Total meetings: {summary['total_meetings']}")
        print(f"   Cities: {', '.join(summary['cities'])}")
        print(f"   Total entities found: {summary['total_entities']}")
        print(f"   Sentiment distribution: {summary['sentiment_distribution']}")
        print(f"   Top topics: {summary['top_topics']}")
    else:
        print("❌ No meetings processed. Make sure to run data ingestion first:")
        print("   python3 src/ingestion/fcc_data_client.py")
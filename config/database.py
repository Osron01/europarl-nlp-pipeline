import os
from sqlalchemy import create_engine, MetaData, text
from pymongo import MongoClient
import redis
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# PostgreSQL configuration
POSTGRES_URL = f"postgresql://{os.getenv('POSTGRES_USER', 'postgres')}:{os.getenv('POSTGRES_PASSWORD', 'password123')}@{os.getenv('POSTGRES_HOST', 'localhost')}:{os.getenv('POSTGRES_PORT', '5432')}/{os.getenv('POSTGRES_DB', 'meetingdb')}"


def get_postgres_engine():
    """Create PostgreSQL engine"""
    return create_engine(POSTGRES_URL)


def get_mongo_client():
    """Create MongoDB client"""
    client = MongoClient(f"mongodb://{os.getenv('MONGO_HOST', 'localhost')}:{os.getenv('MONGO_PORT', '27017')}/")
    return client[os.getenv('MONGO_DB', 'meeting_intelligence')]


def get_redis_client():
    """Create Redis client"""
    return redis.Redis(
        host=os.getenv('REDIS_HOST', 'localhost'),
        port=int(os.getenv('REDIS_PORT', '6379')),
        decode_responses=True
    )


# Test connections
if __name__ == "__main__":
    try:
        print("Testing database connections...")

        # Test PostgreSQL
        engine = get_postgres_engine()
        with engine.connect() as conn:
            result = conn.execute(text("SELECT version();"))
            print(f"✅ PostgreSQL: Connected successfully")

        # Test MongoDB
        mongo_db = get_mongo_client()
        collections = mongo_db.list_collection_names()
        print(f"✅ MongoDB: Connected to {mongo_db.name}")

        # Test Redis
        redis_client = get_redis_client()
        redis_client.set("test", "connection_successful")
        result = redis_client.get("test")
        print(f"✅ Redis: {result}")

        print("\n🎉 All database connections successful!")

    except Exception as e:
        print(f"❌ Database connection error: {e}")
        print("Make sure Docker containers are running: docker-compose up -d")
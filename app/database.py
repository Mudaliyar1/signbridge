import logging
try:
    import dns.resolver
    # Ensure dnspython on Windows has reliable fallback public DNS for SRV lookups
    if dns.resolver.default_resolver is None:
        dns.resolver.default_resolver = dns.resolver.Resolver(configure=True)
    if not dns.resolver.default_resolver.nameservers:
        dns.resolver.default_resolver.nameservers = ['8.8.8.8', '1.1.1.1', '8.8.4.4']
except Exception:
    pass

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo import ASCENDING, DESCENDING
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

from app.config import settings

logger = logging.getLogger("signbridge.database")

class Database:
    client: AsyncIOMotorClient | None = None
    db: AsyncIOMotorDatabase | None = None
    _connected: bool = False
    _last_error: str | None = None

    @classmethod
    async def connect(cls):
        try:
            logger.info(f"Connecting to MongoDB Atlas database '{settings.DATABASE_NAME}'...")
            cls.client = AsyncIOMotorClient(
                settings.MONGODB_URL,
                serverSelectionTimeoutMS=8000,
                connectTimeoutMS=8000
            )
            cls.db = cls.client[settings.DATABASE_NAME]
            
            # Verify connectivity strictly
            await cls.client.admin.command("ping")
            cls._connected = True
            cls._last_error = None
            logger.info(f"Successfully connected to MongoDB Atlas database '{settings.DATABASE_NAME}'")
            
            # Ensure required indexes
            await cls.init_indexes()
        except (ConnectionFailure, ServerSelectionTimeoutError, Exception) as e:
            cls._connected = False
            cls._last_error = str(e)
            logger.error(f"CRITICAL: Failed to connect to MongoDB Atlas: {e}")

    @classmethod
    async def check_health(cls) -> bool:
        """Dynamically ping MongoDB Atlas to verify real-time status."""
        if cls.client is None:
            return False
        try:
            await cls.client.admin.command("ping")
            cls._connected = True
            cls._last_error = None
            return True
        except Exception as e:
            cls._connected = False
            cls._last_error = str(e)
            return False

    @classmethod
    def is_connected(cls) -> bool:
        return cls._connected

    @classmethod
    def get_last_error(cls) -> str | None:
        return cls._last_error

    @classmethod
    async def close(cls):
        if cls.client:
            cls.client.close()
            cls._connected = False
            logger.info("MongoDB connection closed.")

    @classmethod
    async def init_indexes(cls):
        """Create indexes required by specification."""
        if not cls._connected or cls.db is None:
            return
        
        try:
            # 1. models
            await cls.db.models.create_index([("name", ASCENDING)])
            await cls.db.models.create_index([("status", ASCENDING)])
            await cls.db.models.create_index([("version", ASCENDING)])
            
            # 2. sign_classes
            await cls.db.sign_classes.create_index([("model_id", ASCENDING), ("name", ASCENDING)])
            
            # 3. participants
            await cls.db.participants.create_index([("participant_id", ASCENDING)], unique=True)
            
            # 4. dataset_samples
            await cls.db.dataset_samples.create_index([("model_id", ASCENDING)])
            await cls.db.dataset_samples.create_index([("class_id", ASCENDING)])
            await cls.db.dataset_samples.create_index([("participant_id", ASCENDING)])
            await cls.db.dataset_samples.create_index([("media_type", ASCENDING)])
            await cls.db.dataset_samples.create_index([("status", ASCENDING)])
            await cls.db.dataset_samples.create_index([("storage_provider", ASCENDING)])
            await cls.db.dataset_samples.create_index([("cloudinary.public_id", ASCENDING)])
            await cls.db.dataset_samples.create_index([("created_at", DESCENDING)])
            await cls.db.dataset_samples.create_index([
                ("model_id", ASCENDING),
                ("class_id", ASCENDING),
                ("participant_id", ASCENDING)
            ])
            
            # 5. training_runs
            await cls.db.training_runs.create_index([("model_id", ASCENDING)])
            await cls.db.training_runs.create_index([("status", ASCENDING)])
            await cls.db.training_runs.create_index([("created_at", DESCENDING)])
            
            # 6. model_versions
            await cls.db.model_versions.create_index([("model_id", ASCENDING), ("version", ASCENDING)])
            
            # 7. testing_results
            await cls.db.testing_results.create_index([("model_id", ASCENDING), ("created_at", DESCENDING)])
            await cls.db.testing_results.create_index([("model_version_id", ASCENDING)])
            
            # 8. ml_experiments
            await cls.db.ml_experiments.create_index([("model_id", ASCENDING), ("created_at", DESCENDING)])
            await cls.db.ml_experiments.create_index([("algorithm", ASCENDING)])
            
            logger.info("MongoDB Atlas collections and indexes initialized successfully.")
        except Exception as e:
            logger.error(f"Error creating indexes on MongoDB Atlas: {e}")

# Helpers for collection access
def get_db() -> AsyncIOMotorDatabase:
    if Database.db is None or not Database.is_connected():
        raise RuntimeError(f"MongoDB Atlas is unavailable. Error: {Database.get_last_error() or 'Not connected'}")
    return Database.db

def models_col():
    return get_db().models

def classes_col():
    return get_db().sign_classes

def participants_col():
    return get_db().participants

def samples_col():
    return get_db().dataset_samples

def runs_col():
    return get_db().training_runs

def versions_col():
    return get_db().model_versions

def testing_results_col():
    return get_db().testing_results

def experiments_col():
    return get_db().ml_experiments

from dotenv import load_dotenv
import os

load_dotenv()

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
CRIC_API_KEY = os.getenv("CRIC_API_KEY")

# LLM settings
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.1-70b-versatile")
GROQ_TEMPERATURE = float(os.getenv("GROQ_TEMPERATURE", "0.3"))

# App settings
APP_TITLE = os.getenv("APP_TITLE", "Cricket Decision Intelligence System")
DEBUG = os.getenv("DEBUG", "False").lower() == "true"

# CricAPI base URL
CRIC_API_BASE_URL = "https://api.cricapi.com/v1"

# ChromaDB path
VECTOR_STORE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), 
    "data", 
    "vector_store"
)

# SQLite DB path
SQLITE_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "data",
    "predictions.db"
)

# Validate critical keys on startup
def validate_config():
    missing = []
    if not GROQ_API_KEY:
        missing.append("GROQ_API_KEY")
    if not CRIC_API_KEY:
        missing.append("CRIC_API_KEY")
    if missing:
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

validate_config()

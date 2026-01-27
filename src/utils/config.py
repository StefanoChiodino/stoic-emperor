import os
import yaml
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

def load_env():
    """Load environment variables from .env file"""
    load_dotenv()

def load_config(config_path: str = "config/settings.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file and substitute env vars"""
    load_env()
    
    if not os.path.exists(config_path):
        # Return default config if file doesn't exist yet
        return default_config()
        
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    return substitute_env_vars(config)

def substitute_env_vars(config: Any) -> Any:
    """Recursively substitute environment variables in config values"""
    if isinstance(config, dict):
        return {k: substitute_env_vars(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [substitute_env_vars(v) for v in config]
    elif isinstance(config, str) and config.startswith("${") and config.endswith("}"):
        env_var = config[2:-1]
        return os.getenv(env_var, config)
    return config

def default_config() -> Dict[str, Any]:
    """Return default configuration"""
    return {
        "models": {
            "main": os.getenv("LLM_MAIN_MODEL", "gpt-4o"),
            "reviewer": os.getenv("LLM_REVIEWER_MODEL", "claude-3-5-sonnet"),
        },
        "paths": {
            "stoic_texts": os.getenv("STOIC_TEXTS_FOLDER", "./data/stoic_texts"),
            "vector_db": os.getenv("VECTOR_DB_PATH", "./data/vector_db"),
            "sqlite_db": os.getenv("SQLITE_DB_PATH", "./data/stoic_emperor.db"),
        },
        "memory": {
            "max_context_tokens": 4000,
            "embedding_model": "text-embedding-3-small",
        },
        "rag": {
            "chunk_size": 500,
            "chunk_overlap": 50,
            "similarity_threshold": 0.6,
        },
        "aegean_consensus": {
            "enabled": True,
            "beta_threshold": 2,
            "alpha_quorum": 1.0,
            "sessions_between_analysis": 5,
        }
    }

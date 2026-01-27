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
        config = default_config()
    else:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        config = substitute_env_vars(config)
    
    prompts_path = Path("config/prompts.yaml")
    if prompts_path.exists():
        with open(prompts_path, 'r') as f:
            prompts = yaml.safe_load(f)
            config["prompts"] = prompts
    
    return config

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
        "database": {
            "url": os.getenv("DATABASE_URL", "sqlite:///./data/stoic_emperor.db"),
        },
        "paths": {
            "stoic_texts": os.getenv("STOIC_TEXTS_FOLDER", "./data/stoic_texts"),
        },
        "auth": {
            "jwt_secret": os.getenv("JWT_SECRET"),
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
        },
        "condensation": {
            "hot_buffer_tokens": 4000,
            "chunk_threshold_tokens": 8000,
            "summary_budget_tokens": 12000,
            "use_consensus": True,
        }
    }

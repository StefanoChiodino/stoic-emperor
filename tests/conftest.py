import os
import shutil
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def temp_dir():
    dirpath = tempfile.mkdtemp()
    yield Path(dirpath)
    shutil.rmtree(dirpath)


@pytest.fixture
def test_db_path(temp_dir):
    return str(temp_dir / "test_stoic_emperor.db")


@pytest.fixture
def test_vector_path(temp_dir):
    return str(temp_dir / "test_vector_db")


@pytest.fixture
def sample_journal_entry():
    return """Today was difficult. I argued with my father again about my career choices.
He thinks I'm wasting my potential, but I feel like I'm finally doing what matters to me.
The conflict is exhausting. I keep replaying our conversation in my head.
Why can't he just accept that I'm not him?"""


@pytest.fixture
def sample_psych_update():
    from src.models.schemas import PsychUpdate

    return PsychUpdate(
        detected_patterns=["conflict_avoidance", "external_validation_seeking"],
        emotional_state="frustrated, seeking approval",
        stoic_principle_applied="Dichotomy of Control",
        suggested_next_direction="Explore father relationship dynamics",
        confidence=0.85,
    )


@pytest.fixture
def integration_test_config(temp_dir):
    env_file = Path(__file__).parent.parent / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key, value = key.strip(), value.strip()
            if key and value and not os.environ.get(key, "").strip():
                os.environ[key] = value

    main_model = os.getenv("LLM_MAIN_MODEL", "gpt-4o-mini")
    reviewer_model = os.getenv("LLM_REVIEWER_MODEL", "gpt-4o-mini")
    base_url = os.getenv("LLM_BASE_URL")

    return {
        "db_path": str(temp_dir / "integration_test.db"),
        "vector_path": str(temp_dir / "integration_vector_db"),
        "models": {
            "main": os.getenv("TEST_LLM_MAIN_MODEL", main_model),
            "reviewer": os.getenv("TEST_LLM_REVIEWER_MODEL", reviewer_model),
        },
        "base_url": base_url,
    }


@pytest.fixture
def mock_emperor_response():
    return {
        "response_text": "You speak of your father's expectations as if they were chains upon your soul. But consider: whose life is it that you live? The Stoic knows that another's opinion, however forceful, remains outside our control. What is within your power is how you respond to his words.",
        "psych_update": {
            "detected_patterns": ["external_locus_of_control", "approval_seeking"],
            "emotional_state": "conflicted, frustrated",
            "stoic_principle_applied": "Dichotomy of Control",
            "suggested_next_direction": "Explore internalized parental expectations",
            "confidence": 0.8,
        },
    }

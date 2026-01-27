import pytest
from pathlib import Path

from src.infrastructure.database import Database
from src.infrastructure.vector_store import VectorStore
from src.infrastructure.ingestion_pipeline import IngestionPipeline, ingest_stoic_highlights
from src.models.schemas import User, Session, Message


class TestIngestionPipelineUnit:
    def test_chunk_text(self, test_vector_path):
        vectors = VectorStore(test_vector_path)
        pipeline = IngestionPipeline(vectors, llm=None)
        
        text = " ".join(["word"] * 1000)
        chunks = pipeline._chunk_text(text, source="test", author="Test", work="Test Work")
        
        assert len(chunks) > 1
        for chunk in chunks:
            word_count = len(chunk.content.split())
            assert word_count <= pipeline.chunk_size + 10


class TestIngestionPipelineWithHighlights:
    def test_ingest_stoic_highlights_structure(self, test_vector_path):
        vectors = VectorStore(test_vector_path)
        
        initial_count = vectors.count("stoic_wisdom")
        assert initial_count == 0


@pytest.mark.integration
class TestIngestionPipelineIntegration:
    def test_ingest_highlights_with_tagging(self, integration_test_config):
        from src.utils.llm_client import LLMClient
        
        vectors = VectorStore(integration_test_config["vector_path"])
        llm = LLMClient(base_url=integration_test_config.get("base_url"))
        
        count = ingest_stoic_highlights(vectors, llm)
        
        assert count > 0
        
        total = vectors.count("stoic_wisdom")
        assert total > 0

    def test_query_ingested_highlights(self, integration_test_config):
        from src.utils.llm_client import LLMClient
        
        vectors = VectorStore(integration_test_config["vector_path"])
        llm = LLMClient(base_url=integration_test_config.get("base_url"))
        
        ingest_stoic_highlights(vectors, llm)
        
        results = vectors.query(
            collection="stoic_wisdom",
            query_texts=["control over my thoughts"],
            n_results=3
        )
        
        assert results is not None
        assert "documents" in results
        assert len(results["documents"][0]) > 0


@pytest.mark.integration
class TestFullChatWorkflow:
    def test_chat_session_workflow(self, integration_test_config):
        from src.core.emperor_brain import EmperorBrain
        from src.utils.llm_client import LLMClient
        
        db = Database(integration_test_config["db_path"])
        vectors = VectorStore(integration_test_config["vector_path"])
        llm = LLMClient(base_url=integration_test_config.get("base_url"))
        
        config = {
            "models": integration_test_config["models"],
            "paths": {"sqlite_db": integration_test_config["db_path"]}
        }
        brain = EmperorBrain(llm_client=llm, config=config)
        
        user = db.get_or_create_user("test_workflow_user")
        session = Session(user_id=user.id)
        db.create_session(session)
        
        user_msg = Message(
            session_id=session.id,
            role="user",
            content="I feel like I'm not good enough for my job."
        )
        db.save_message(user_msg)
        
        response = brain.respond(
            user_message=user_msg.content,
            conversation_history=[user_msg],
            retrieved_context=None
        )
        
        emperor_msg = Message(
            session_id=session.id,
            role="emperor",
            content=response.response_text,
            psych_update=response.psych_update
        )
        db.save_message(emperor_msg)
        
        messages = db.get_session_messages(session.id)
        assert len(messages) == 2
        assert messages[1].psych_update is not None

    def test_multi_turn_conversation(self, integration_test_config):
        from src.core.emperor_brain import EmperorBrain
        from src.utils.llm_client import LLMClient
        
        db = Database(integration_test_config["db_path"])
        llm = LLMClient(base_url=integration_test_config.get("base_url"))
        
        config = {
            "models": integration_test_config["models"],
            "paths": {"sqlite_db": integration_test_config["db_path"]}
        }
        brain = EmperorBrain(llm_client=llm, config=config)
        
        user = db.get_or_create_user("test_multi_turn_user")
        session = Session(user_id=user.id)
        db.create_session(session)
        
        conversation = [
            "I've been procrastinating a lot lately.",
            "It's mostly about a difficult project at work.",
            "I'm afraid of failing."
        ]
        
        history = []
        
        for user_text in conversation:
            user_msg = Message(session_id=session.id, role="user", content=user_text)
            db.save_message(user_msg)
            history.append(user_msg)
            
            response = brain.respond(
                user_message=user_text,
                conversation_history=history,
                retrieved_context=None
            )
            
            emperor_msg = Message(
                session_id=session.id,
                role="emperor",
                content=response.response_text,
                psych_update=response.psych_update
            )
            db.save_message(emperor_msg)
            history.append(emperor_msg)
        
        all_messages = db.get_session_messages(session.id)
        assert len(all_messages) == 6
        
        emperor_messages = [m for m in all_messages if m.role == "emperor"]
        for msg in emperor_messages:
            assert msg.psych_update is not None


@pytest.mark.integration
class TestAegeanConsensusIntegration:
    def test_consensus_basic(self, integration_test_config):
        from src.core.aegean_consensus import AegeanConsensusProtocol
        
        prompts = {
            "test_analysis": """Analyze the following text and provide a psychological assessment:

{text}

Include: emotional state, cognitive patterns, and recommendations."""
        }
        
        consensus = AegeanConsensusProtocol(
            model_a=integration_test_config["models"]["main"],
            model_b=integration_test_config["models"]["reviewer"],
            prompts=prompts,
            beta_threshold=1,
            verbose=False
        )
        
        result = consensus.reach_consensus(
            prompt_name="test_analysis",
            variables={"text": "I feel overwhelmed by work. Everything seems impossible."},
            max_rounds=1
        )
        
        assert result is not None
        assert len(result.final_output) > 0
        assert result.stability_score >= 0

import json
from unittest.mock import MagicMock, patch

import pytest

from src.infrastructure.ingestion_pipeline import (
    IngestionPipeline,
    TaggedChunk,
    TextChunk,
    ingest_stoic_highlights,
)


class TestTextChunk:
    def test_chunk_initialization(self):
        chunk = TextChunk(
            id="chunk_1",
            content="Test content",
            source="/path/to/file.txt",
        )

        assert chunk.id == "chunk_1"
        assert chunk.content == "Test content"
        assert chunk.source == "/path/to/file.txt"
        assert chunk.author is None
        assert chunk.work is None

    def test_chunk_with_all_fields(self):
        chunk = TextChunk(
            id="chunk_2",
            content="Philosophical text",
            source="/path/to/meditations.txt",
            author="Marcus Aurelius",
            work="Meditations",
            chapter="Book I",
        )

        assert chunk.author == "Marcus Aurelius"
        assert chunk.work == "Meditations"
        assert chunk.chapter == "Book I"


class TestTaggedChunk:
    def test_tagged_chunk_initialization(self):
        chunk = TextChunk(id="c1", content="Test", source="test.txt")
        tagged = TaggedChunk(
            chunk=chunk,
            classical_tags=["Amor Fati", "Memento Mori"],
            modern_tags=["acceptance", "mortality"],
            themes=["death", "fate"],
        )

        assert tagged.chunk.id == "c1"
        assert "Amor Fati" in tagged.classical_tags
        assert "acceptance" in tagged.modern_tags
        assert "death" in tagged.themes


class TestIngestionPipeline:
    @pytest.fixture
    def mock_vector_store(self):
        mock = MagicMock()
        mock.add.return_value = None
        return mock

    @pytest.fixture
    def mock_llm_client(self):
        mock = MagicMock()
        mock.generate.return_value = json.dumps(
            {
                "classical_tags": ["Stoic Acceptance"],
                "modern_tags": ["mindfulness"],
                "themes": ["self-control"],
            }
        )
        return mock

    @pytest.fixture
    def pipeline(self, mock_vector_store, mock_llm_client):
        config = {
            "rag": {"chunk_size": 100, "chunk_overlap": 10},
            "models": {"main": "gpt-4"},
        }
        return IngestionPipeline(mock_vector_store, mock_llm_client, config)

    def test_initialization(self, mock_vector_store):
        config = {"rag": {"chunk_size": 200, "chunk_overlap": 20}, "models": {"main": "test-model"}}
        pipeline = IngestionPipeline(mock_vector_store, config=config)

        assert pipeline.chunk_size == 200
        assert pipeline.chunk_overlap == 20
        assert pipeline.vectors == mock_vector_store

    def test_initialization_defaults(self, mock_vector_store):
        pipeline = IngestionPipeline(mock_vector_store, config={})

        assert pipeline.chunk_size == 500
        assert pipeline.chunk_overlap == 50

    def test_chunk_text(self, pipeline):
        text = " ".join([f"word{i}" for i in range(250)])

        chunks = pipeline._chunk_text(text, source="test.txt", author="Test Author", work="Test Work")

        assert len(chunks) > 1
        assert all(isinstance(c, TextChunk) for c in chunks)
        assert chunks[0].author == "Test Author"
        assert chunks[0].work == "Test Work"

    def test_chunk_text_short(self, pipeline):
        text = "Short text with few words"

        chunks = pipeline._chunk_text(text, source="test.txt")

        assert len(chunks) == 1
        assert chunks[0].content == text

    def test_tag_chunk_with_template(self, pipeline):
        pipeline.prompts = {"concept_tagging": "Tag this: {passage}"}
        chunk = TextChunk(id="c1", content="Test passage about stoicism", source="test.txt")

        result = pipeline._tag_chunk(chunk)

        assert isinstance(result, TaggedChunk)
        assert result.chunk.id == "c1"

    def test_tag_chunk_no_template(self, pipeline):
        pipeline.prompts = {}
        chunk = TextChunk(id="c1", content="Test passage", source="test.txt")

        result = pipeline._tag_chunk(chunk)

        assert isinstance(result, TaggedChunk)
        assert result.classical_tags == []
        assert result.modern_tags == []

    def test_tag_chunk_llm_error(self, pipeline, mock_llm_client):
        pipeline.prompts = {"concept_tagging": "Tag: {passage}"}
        mock_llm_client.generate.side_effect = Exception("LLM error")
        chunk = TextChunk(id="c1", content="Test", source="test.txt")

        result = pipeline._tag_chunk(chunk)

        assert result.classical_tags == []
        assert result.modern_tags == []

    def test_store_chunks_empty(self, pipeline):
        result = pipeline._store_chunks([], "stoic_wisdom")

        assert result == 0
        pipeline.vectors.add.assert_not_called()

    def test_store_chunks(self, pipeline, mock_vector_store):
        chunk = TextChunk(id="c1", content="Test content", source="test.txt", author="Author", work="Work")
        tagged = TaggedChunk(
            chunk=chunk,
            classical_tags=["tag1"],
            modern_tags=["tag2"],
            themes=["theme1"],
        )

        result = pipeline._store_chunks([tagged], "stoic_wisdom")

        assert result == 1
        mock_vector_store.add.assert_called_once()
        call_args = mock_vector_store.add.call_args
        assert call_args.kwargs["collection"] == "stoic_wisdom"
        assert call_args.kwargs["ids"] == ["c1"]

    def test_ingest_stoic_text(self, pipeline, temp_dir):
        test_file = temp_dir / "stoic.txt"
        test_file.write_text("You have power over your mind, not outside events. " * 20)

        result = pipeline.ingest_stoic_text(
            str(test_file), author="Marcus Aurelius", work="Meditations", tag_with_llm=False
        )

        assert result > 0
        pipeline.vectors.add.assert_called()

    def test_ingest_stoic_text_file_not_found(self, pipeline):
        with pytest.raises(FileNotFoundError):
            pipeline.ingest_stoic_text("/nonexistent/file.txt", "Author", "Work")

    def test_ingest_stoic_text_with_tagging(self, pipeline, temp_dir, mock_llm_client):
        pipeline.prompts = {"concept_tagging": "Tag: {passage}"}
        test_file = temp_dir / "stoic.txt"
        test_file.write_text("Test content " * 50)

        result = pipeline.ingest_stoic_text(str(test_file), "Author", "Work", tag_with_llm=True)

        assert result > 0

    def test_ingest_psychoanalysis_text(self, pipeline, temp_dir):
        test_file = temp_dir / "psych.txt"
        test_file.write_text("Psychological concepts and analysis " * 20)

        result = pipeline.ingest_psychoanalysis_text(str(test_file), author="Freud", work="Studies", tag_with_llm=False)

        assert result > 0

    def test_ingest_psychoanalysis_text_file_not_found(self, pipeline):
        with pytest.raises(FileNotFoundError):
            pipeline.ingest_psychoanalysis_text("/nonexistent.txt", "Author", "Work")

    def test_ingest_directory(self, pipeline, temp_dir):
        subdir = temp_dir / "texts"
        subdir.mkdir()
        (subdir / "file1.txt").write_text("Content one " * 50)
        (subdir / "file2.txt").write_text("Content two " * 50)
        (subdir / "file3.md").write_text("Markdown content " * 50)

        result = pipeline.ingest_directory(
            str(subdir),
            collection="stoic_wisdom",
            author="Test",
            work="Collection",
            tag_with_llm=False,
        )

        assert result > 0

    def test_ingest_directory_not_a_directory(self, pipeline, temp_dir):
        test_file = temp_dir / "file.txt"
        test_file.write_text("content")

        with pytest.raises(NotADirectoryError):
            pipeline.ingest_directory(str(test_file), "stoic_wisdom", "Author", "Work")

    def test_ingest_directory_custom_extensions(self, pipeline, temp_dir):
        subdir = temp_dir / "texts"
        subdir.mkdir()
        (subdir / "file1.rst").write_text("RST content " * 50)
        (subdir / "file2.txt").write_text("TXT content " * 50)

        result = pipeline.ingest_directory(
            str(subdir),
            collection="psychoanalysis",
            author="Test",
            work="Work",
            extensions=[".rst"],
            tag_with_llm=False,
        )

        assert result > 0

    def test_ingest_directory_psychoanalysis_collection(self, pipeline, temp_dir):
        subdir = temp_dir / "psych_texts"
        subdir.mkdir()
        (subdir / "concepts.txt").write_text("Psychological concepts " * 50)

        result = pipeline.ingest_directory(
            str(subdir),
            collection="psychoanalysis",
            author="Jung",
            work="Archetypes",
            tag_with_llm=False,
        )

        assert result > 0


class TestIngestStoicHighlights:
    def test_ingest_stoic_highlights(self):
        mock_vectors = MagicMock()
        mock_llm = MagicMock()
        mock_llm.generate.return_value = json.dumps(
            {
                "classical_tags": ["Virtue"],
                "modern_tags": ["ethics"],
                "themes": ["character"],
            }
        )

        with patch("src.infrastructure.ingestion_pipeline.IngestionPipeline._load_prompts") as mock_prompts:
            mock_prompts.return_value = {"concept_tagging": "Tag: {passage}"}

            result = ingest_stoic_highlights(mock_vectors, mock_llm)

        assert result > 0
        assert mock_vectors.add.call_count > 0

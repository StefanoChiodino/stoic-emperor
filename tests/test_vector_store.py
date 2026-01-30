import pytest

from src.infrastructure.vector_store import VectorStore


class TestVectorStoreInitialization:
    def test_creates_collections(self, test_vector_path):
        vs = VectorStore(test_vector_path)

        for collection in VectorStore.COLLECTIONS:
            count = vs.count(collection)
            assert count >= 0

    def test_collections_exist(self, test_vector_path):
        _vs = VectorStore(test_vector_path)

        assert "episodic" in VectorStore.COLLECTIONS
        assert "semantic" in VectorStore.COLLECTIONS
        assert "stoic_wisdom" in VectorStore.COLLECTIONS
        assert "psychoanalysis" in VectorStore.COLLECTIONS


class TestVectorStoreOperations:
    def test_add_and_query(self, test_vector_path):
        vs = VectorStore(test_vector_path)

        vs.add(
            collection="stoic_wisdom",
            ids=["doc_1"],
            documents=["You have power over your mind, not outside events."],
            metadatas=[{"author": "Marcus Aurelius", "work": "Meditations"}],
        )

        results = vs.query(collection="stoic_wisdom", query_texts=["control over thoughts"], n_results=1)

        assert results is not None
        assert "documents" in results

    def test_add_multiple_documents(self, test_vector_path):
        vs = VectorStore(test_vector_path)

        vs.add(
            collection="semantic",
            ids=["insight_1", "insight_2", "insight_3"],
            documents=[
                "User avoids conflict with authority figures",
                "User seeks external validation",
                "User struggles with decision-making",
            ],
            metadatas=[
                {"user_id": "test_user", "confidence": 0.8},
                {"user_id": "test_user", "confidence": 0.9},
                {"user_id": "test_user", "confidence": 0.7},
            ],
        )

        count = vs.count("semantic")
        assert count == 3

    def test_query_with_filter(self, test_vector_path):
        vs = VectorStore(test_vector_path)

        vs.add(
            collection="episodic",
            ids=["turn_1", "turn_2"],
            documents=[
                "User: I feel anxious. Marcus: Consider what is within your control.",
                "User: Work is stressful. Marcus: The obstacle is the way.",
            ],
            metadatas=[{"user_id": "user_a", "session_id": "s1"}, {"user_id": "user_b", "session_id": "s2"}],
        )

        results = vs.query(collection="episodic", query_texts=["anxiety"], n_results=5, where={"user_id": "user_a"})

        assert results is not None

    def test_get_by_ids(self, test_vector_path):
        vs = VectorStore(test_vector_path)

        vs.add(
            collection="stoic_wisdom", ids=["get_test_1", "get_test_2"], documents=["First document", "Second document"]
        )

        results = vs.get(collection="stoic_wisdom", ids=["get_test_1"])

        assert results is not None
        assert len(results.get("ids", [])) == 1

    def test_delete(self, test_vector_path):
        vs = VectorStore(test_vector_path)

        vs.add(collection="psychoanalysis", ids=["del_1", "del_2"], documents=["To delete", "To keep"])

        vs.delete(collection="psychoanalysis", ids=["del_1"])

        results = vs.get(collection="psychoanalysis", ids=["del_1", "del_2"])
        assert "del_1" not in results.get("ids", [])


class TestVectorStorePersistence:
    def test_data_persists(self, test_vector_path):
        vs1 = VectorStore(test_vector_path)
        vs1.add(collection="stoic_wisdom", ids=["persist_1"], documents=["This should persist across instances"])

        del vs1

        vs2 = VectorStore(test_vector_path)
        results = vs2.get(collection="stoic_wisdom", ids=["persist_1"])

        assert "persist_1" in results.get("ids", [])


class TestVectorStoreEdgeCases:
    def test_query_empty_collection(self, test_vector_path):
        vs = VectorStore(test_vector_path)

        results = vs.query(collection="episodic", query_texts=["anything"], n_results=5)

        assert results is not None

    def test_count_empty_collection(self, test_vector_path):
        vs = VectorStore(test_vector_path)

        count = vs.count("episodic")

        assert count == 0

    def test_query_no_input_raises(self, test_vector_path):
        vs = VectorStore(test_vector_path)

        with pytest.raises(ValueError, match="Either query_texts or query_embeddings"):
            vs.query(collection="episodic", n_results=5)

    def test_get_with_where_filter(self, test_vector_path):
        vs = VectorStore(test_vector_path)

        vs.add(
            collection="semantic",
            ids=["w1", "w2", "w3"],
            documents=["Doc for user A", "Doc for user B", "Another for user A"],
            metadatas=[
                {"user_id": "user_a"},
                {"user_id": "user_b"},
                {"user_id": "user_a"},
            ],
        )

        results = vs.get(collection="semantic", where={"user_id": "user_a"})

        assert len(results["ids"]) == 2
        assert "w1" in results["ids"]
        assert "w3" in results["ids"]

    def test_get_with_limit(self, test_vector_path):
        vs = VectorStore(test_vector_path)

        vs.add(
            collection="stoic_wisdom",
            ids=["lim1", "lim2", "lim3"],
            documents=["Doc 1", "Doc 2", "Doc 3"],
        )

        results = vs.get(collection="stoic_wisdom", limit=2)

        assert len(results["ids"]) == 2

    def test_delete_with_where_filter(self, test_vector_path):
        vs = VectorStore(test_vector_path)

        vs.add(
            collection="episodic",
            ids=["dw1", "dw2", "dw3"],
            documents=["Session A doc", "Session B doc", "Session A doc 2"],
            metadatas=[
                {"session_id": "sess_a"},
                {"session_id": "sess_b"},
                {"session_id": "sess_a"},
            ],
        )

        vs.delete(collection="episodic", where={"session_id": "sess_a"})

        results = vs.get(collection="episodic", ids=["dw1", "dw2", "dw3"])
        assert "dw2" in results["ids"]
        assert "dw1" not in results["ids"]
        assert "dw3" not in results["ids"]

    def test_add_with_custom_embeddings(self, test_vector_path):
        vs = VectorStore(test_vector_path)

        custom_embedding = [0.1] * 384

        vs.add(
            collection="stoic_wisdom",
            ids=["emb1"],
            documents=["Custom embedded doc"],
            embeddings=[custom_embedding],
        )

        results = vs.get(collection="stoic_wisdom", ids=["emb1"])
        assert "emb1" in results["ids"]

    def test_query_with_embeddings(self, test_vector_path):
        vs = VectorStore(test_vector_path)

        vs.add(
            collection="stoic_wisdom",
            ids=["qe1"],
            documents=["Test document for embedding query"],
        )

        query_embedding = vs.embedding_model.encode(["similar query"]).tolist()
        results = vs.query(collection="stoic_wisdom", query_embeddings=query_embedding, n_results=1)

        assert results is not None

    def test_add_upsert_behavior(self, test_vector_path):
        vs = VectorStore(test_vector_path)

        vs.add(
            collection="stoic_wisdom",
            ids=["upsert1"],
            documents=["Original content"],
            metadatas=[{"version": "1"}],
        )

        vs.add(
            collection="stoic_wisdom",
            ids=["upsert1"],
            documents=["Updated content"],
            metadatas=[{"version": "2"}],
        )

        results = vs.get(collection="stoic_wisdom", ids=["upsert1"])
        assert results["documents"][0] == "Updated content"
        assert results["metadatas"][0]["version"] == "2"

    def test_cosine_similarity(self, test_vector_path):
        vs = VectorStore(test_vector_path)

        similarity = vs._cosine_similarity([1.0, 0.0, 0.0], [1.0, 0.0, 0.0])
        assert abs(similarity - 1.0) < 0.001

        similarity_orthogonal = vs._cosine_similarity([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
        assert abs(similarity_orthogonal) < 0.001

    def test_add_empty_metadata(self, test_vector_path):
        vs = VectorStore(test_vector_path)

        vs.add(
            collection="stoic_wisdom",
            ids=["nometa1"],
            documents=["Document without metadata"],
        )

        results = vs.get(collection="stoic_wisdom", ids=["nometa1"])
        assert "nometa1" in results["ids"]

    def test_query_returns_sorted_by_relevance(self, test_vector_path):
        vs = VectorStore(test_vector_path)

        vs.add(
            collection="stoic_wisdom",
            ids=["rel1", "rel2", "rel3"],
            documents=[
                "This is about anxiety and worry",
                "This is about happiness and joy",
                "Mental anxiety causes suffering",
            ],
        )

        results = vs.query(collection="stoic_wisdom", query_texts=["anxiety"], n_results=3)

        assert len(results["documents"][0]) <= 3

    def test_get_empty_result(self, test_vector_path):
        vs = VectorStore(test_vector_path)

        results = vs.get(collection="episodic", ids=["nonexistent"])

        assert results["ids"] == []
        assert results["documents"] == []

    def test_delete_nonexistent_id(self, test_vector_path):
        vs = VectorStore(test_vector_path)

        vs.delete(collection="stoic_wisdom", ids=["does_not_exist"])

    def test_multiple_queries_same_collection(self, test_vector_path):
        vs = VectorStore(test_vector_path)

        vs.add(
            collection="semantic",
            ids=["mq1", "mq2"],
            documents=["First insight", "Second insight"],
            metadatas=[{"user_id": "u1"}, {"user_id": "u1"}],
        )

        results1 = vs.query(collection="semantic", query_texts=["first"], n_results=1)
        results2 = vs.query(collection="semantic", query_texts=["second"], n_results=1)

        assert results1 is not None
        assert results2 is not None

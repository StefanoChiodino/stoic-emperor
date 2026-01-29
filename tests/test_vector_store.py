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

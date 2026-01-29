from unittest.mock import MagicMock, patch


class TestLLMClient:
    def test_initialization_with_defaults(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            from src.utils.llm_client import LLMClient

            client = LLMClient()
            assert client.client is not None

    def test_initialization_with_custom_api_key(self):
        from src.utils.llm_client import LLMClient

        client = LLMClient(api_key="custom-key")
        assert client.client is not None

    def test_initialization_with_base_url(self):
        from src.utils.llm_client import LLMClient

        client = LLMClient(api_key="test-key", base_url="http://localhost:8000")
        assert client.base_url == "http://localhost:8000"

    def test_generate_calls_openai(self):
        with patch("src.utils.llm_client.OpenAI") as mock_openai_class:
            mock_client = MagicMock()
            mock_openai_class.return_value = mock_client

            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "Generated response"
            mock_client.chat.completions.create.return_value = mock_response

            from src.utils.llm_client import LLMClient

            client = LLMClient(api_key="test-key")
            result = client.generate("Test prompt")

            assert result == "Generated response"
            mock_client.chat.completions.create.assert_called_once()

    def test_generate_with_json_mode(self):
        with patch("src.utils.llm_client.OpenAI") as mock_openai_class:
            mock_client = MagicMock()
            mock_openai_class.return_value = mock_client

            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = '{"key": "value"}'
            mock_client.chat.completions.create.return_value = mock_response

            from src.utils.llm_client import LLMClient

            client = LLMClient(api_key="test-key")
            result = client.generate("Test prompt", json_mode=True)

            assert result == '{"key": "value"}'

    def test_generate_handles_none_content(self):
        with patch("src.utils.llm_client.OpenAI") as mock_openai_class:
            mock_client = MagicMock()
            mock_openai_class.return_value = mock_client

            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = None
            mock_client.chat.completions.create.return_value = mock_response

            from src.utils.llm_client import LLMClient

            client = LLMClient(api_key="test-key")
            result = client.generate("Test prompt")

            assert result == ""

    def test_get_embedding(self):
        with patch("src.utils.llm_client.OpenAI") as mock_openai_class:
            mock_client = MagicMock()
            mock_openai_class.return_value = mock_client

            mock_response = MagicMock()
            mock_response.data = [MagicMock()]
            mock_response.data[0].embedding = [0.1, 0.2, 0.3]
            mock_client.embeddings.create.return_value = mock_response

            from src.utils.llm_client import LLMClient

            client = LLMClient(api_key="test-key")
            result = client.get_embedding("test text")

            assert result == [0.1, 0.2, 0.3]

    def test_generate_structured_success(self):
        with patch("src.utils.llm_client.OpenAI") as mock_openai_class:
            from pydantic import BaseModel

            class TestModel(BaseModel):
                name: str
                value: int

            mock_client = MagicMock()
            mock_openai_class.return_value = mock_client

            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = '{"name": "test", "value": 42}'
            mock_client.chat.completions.create.return_value = mock_response

            from src.utils.llm_client import LLMClient

            client = LLMClient(api_key="test-key")
            result = client.generate_structured("Test prompt", TestModel)

            assert isinstance(result, TestModel)
            assert result.name == "test"
            assert result.value == 42

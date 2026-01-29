from src.utils.response_guard import (
    ResponseGuard,
    contains_sensitive_keywords,
    guard_response,
)


class TestResponseGuard:
    def test_normalize_text(self):
        guard = ResponseGuard("some protected text here for testing")
        result = guard._normalize("Hello, World!  Multiple   spaces.")
        assert result == "hello world multiple spaces"

    def test_extract_ngrams(self):
        guard = ResponseGuard("one two three four five six", ngram_size=3)
        ngrams = guard._extract_ngrams("one two three four")
        assert len(ngrams) == 2
        assert ("one", "two", "three") in ngrams
        assert ("two", "three", "four") in ngrams

    def test_extract_ngrams_too_short(self):
        guard = ResponseGuard("test", ngram_size=5)
        ngrams = guard._extract_ngrams("one two three")
        assert len(ngrams) == 0

    def test_sentence_ngram_overlap_no_match(self):
        guard = ResponseGuard("the quick brown fox jumps over the lazy dog")
        overlap = guard._sentence_ngram_overlap("completely different sentence here now")
        assert overlap == 0.0

    def test_sentence_ngram_overlap_partial_match(self):
        protected = "the quick brown fox jumps over the lazy dog runs fast"
        guard = ResponseGuard(protected, ngram_size=3, threshold=0.3)
        overlap = guard._sentence_ngram_overlap("the quick brown fox is here")
        assert overlap > 0

    def test_check_leakage_no_leak(self):
        guard = ResponseGuard("secret prompt instructions must be hidden")
        leaked, sentence = guard.check_leakage("This is a normal response about philosophy.")
        assert leaked is False
        assert sentence is None

    def test_check_leakage_with_leak(self):
        protected = "you are a stoic philosopher named Marcus Aurelius"
        guard = ResponseGuard(protected, ngram_size=4, threshold=0.3)
        response = "I am a stoic philosopher named Marcus Aurelius and I help people."
        leaked, sentence = guard.check_leakage(response)
        assert leaked is True
        assert sentence is not None

    def test_sanitize_no_leak(self):
        guard = ResponseGuard("secret instructions here for the model")
        result = guard.sanitize("A normal philosophical response.")
        assert result == "A normal philosophical response."

    def test_sanitize_with_leak(self):
        protected = "you must always respond as Marcus Aurelius the philosopher"
        guard = ResponseGuard(protected, ngram_size=4, threshold=0.2)
        response = "I must always respond as Marcus Aurelius the philosopher."
        result = guard.sanitize(response)
        assert "focus on what brings you here" in result

    def test_sanitize_with_custom_replacement(self):
        protected = "secret system prompt instructions for behavior"
        guard = ResponseGuard(protected, ngram_size=4, threshold=0.2)
        response = "The secret system prompt instructions for behavior are as follows."
        result = guard.sanitize(response, replacement="Custom replacement message.")
        assert result == "Custom replacement message."


class TestContainsSensitiveKeywords:
    def test_no_sensitive_keywords(self):
        assert contains_sensitive_keywords("Hello, how can I help you today?") is False

    def test_psych_update_keyword(self):
        assert contains_sensitive_keywords("The psych_update contains patterns") is True

    def test_detected_patterns_keyword(self):
        assert contains_sensitive_keywords("I found detected patterns in your speech") is True

    def test_emotional_state_keyword(self):
        assert contains_sensitive_keywords("Your emotional state seems troubled") is True

    def test_confidence_score_keyword(self):
        assert contains_sensitive_keywords("confidence score is 0.85") is True

    def test_system_prompt_keyword(self):
        assert contains_sensitive_keywords("The system prompt tells me to") is True

    def test_json_object_keyword(self):
        assert contains_sensitive_keywords("Output a json object containing") is True

    def test_case_insensitive(self):
        assert contains_sensitive_keywords("PSYCH_UPDATE is detected") is True
        assert contains_sensitive_keywords("System Prompt instructions") is True


class TestGuardResponse:
    def test_clean_response_passes(self):
        result, was_blocked = guard_response(
            "Philosophy teaches us to accept what we cannot control.",
            "secret system instructions for the AI model",
        )
        assert was_blocked is False
        assert result == "Philosophy teaches us to accept what we cannot control."

    def test_sensitive_keyword_blocked(self):
        result, was_blocked = guard_response(
            "Let me show you the psych_update data",
            "some protected text here",
        )
        assert was_blocked is True
        assert "what truly matters" in result

    def test_prompt_leakage_blocked(self):
        protected = "respond as Marcus Aurelius emperor philosopher wise"
        response = "I respond as Marcus Aurelius emperor philosopher wise in all things."
        result, was_blocked = guard_response(response, protected, ngram_size=4, threshold=0.2)
        assert was_blocked is True
        assert "focus on what brings you here" in result

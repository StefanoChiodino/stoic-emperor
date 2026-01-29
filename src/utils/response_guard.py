import re


class ResponseGuard:
    def __init__(self, protected_text: str, ngram_size: int = 5, threshold: float = 0.3):
        self.ngram_size = ngram_size
        self.threshold = threshold
        self.protected_ngrams = self._extract_ngrams(protected_text)

    def _normalize(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _extract_ngrams(self, text: str) -> set[tuple[str, ...]]:
        words = self._normalize(text).split()
        if len(words) < self.ngram_size:
            return set()
        return {tuple(words[i : i + self.ngram_size]) for i in range(len(words) - self.ngram_size + 1)}

    def _sentence_ngram_overlap(self, sentence: str) -> float:
        sentence_ngrams = self._extract_ngrams(sentence)
        if not sentence_ngrams or not self.protected_ngrams:
            return 0.0
        overlap = len(sentence_ngrams & self.protected_ngrams)
        return overlap / len(sentence_ngrams)

    def check_leakage(self, response: str) -> tuple[bool, str | None]:
        sentences = re.split(r"[.!?\n]", response)

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence.split()) < self.ngram_size:
                continue

            overlap = self._sentence_ngram_overlap(sentence)
            if overlap >= self.threshold:
                return True, sentence

        return False, None

    def sanitize(self, response: str, replacement: str | None = None) -> str:
        leaked, _ = self.check_leakage(response)
        if leaked:
            return replacement or ("I'd rather focus on what brings you here today. What's weighing on your mind?")
        return response


SENSITIVE_PATTERNS = [
    r"psych.?update",
    r"detected.?patterns",
    r"emotional.?state",
    r"confidence.?(?:score|float|0\.\d)",
    r"json.?object.?containing",
    r"output.?format",
    r"system.?(?:prompt|message|instruction)",
    r"persona.?directive",
    r"safety.?protocol",
    r"meta.?instruction",
]


def contains_sensitive_keywords(response: str) -> bool:
    response_lower = response.lower()
    for pattern in SENSITIVE_PATTERNS:
        if re.search(pattern, response_lower):
            return True
    return False


def guard_response(
    response: str, protected_prompt: str, ngram_size: int = 5, threshold: float = 0.3
) -> tuple[str, bool]:
    if contains_sensitive_keywords(response):
        return (
            "Let us turn our attention to what truly matters - your wellbeing. What challenges are you facing?",
            True,
        )

    guard = ResponseGuard(protected_prompt, ngram_size, threshold)
    leaked, _ = guard.check_leakage(response)

    if leaked:
        return ("I'd rather focus on what brings you here today. What's weighing on your mind?", True)

    return response, False

def _fuzzy_contains(text: str, keyword: str, max_errors: int = 1) -> bool:
    """Check if keyword appears in text allowing up to max_errors character differences."""
    klen = len(keyword)
    for i in range(len(text) - klen + 1):
        window = text[i:i + klen]
        errors = sum(a != b for a, b in zip(window, keyword))
        if errors <= max_errors:
            return True
    return False


def decide_route(question: str) -> str:
    """
    Decides whether to route the question to the QA model or Summarizer model.
    Returns 'qa' or 'summarize'.
    Uses fuzzy matching (1 typo allowed) so "sumarize", "summrize" etc. still route correctly.
    """
    question_lower = question.lower()

    # Abstractive / reasoning questions → summarizer
    # These require aggregating or reasoning over multiple chunks,
    # which extractive QA (RoBERTa) cannot do.
    summarize_keywords = [
        "summarize", "summary", "overview", "explain", "meaning",
        "describe", "tell me about", "general idea",
    ]
    reasoning_keywords = [
        "how many", "how long", "how much", "total", "overall",
        "years of", "experience", "background", "profile",
        "list", "what are", "what were", "what has",
    ]

    for kw in summarize_keywords:
        if _fuzzy_contains(question_lower, kw):
            return "summarize"

    for kw in reasoning_keywords:
        if kw in question_lower:
            return "summarize"

    return "qa"

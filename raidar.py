import openai  # or appropriate SDK for the chosen rewriting model (e.g., Google GenAI SDK for Gemma-2B)
import re
import numpy as np

# Configure API for the rewriting LLM (Gemma-2B in this case)
openai.api_key = "YOUR_OPENAI_OR_GENAI_API_KEY"  # ensure to set the correct key for the model's API

def rewrite_text_with_llm(text: str, model_name: str = "gemma-3-2b") -> str:
    """
    Use a small LLM (Gemma-2B or similar) to rewrite the given text.
    Returns the rewritten text.
    """
    # Craft a prompt instructing the model to rewrite the text.
    prompt = f"Rewrite the following text while preserving its meaning and clarity:\n\"\"\"\n{text}\n\"\"\""
    try:
        # Call the LLM API (this example uses OpenAI-style completion for illustration)
        response = openai.ChatCompletion.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,  # deterministic rewriting
        )
        rewritten_text = response['choices'][0]['message']['content']
        return rewritten_text.strip()
    except Exception as e:
        print(f"Error rewriting text: {e}")
        return ""

def character_edit_distance(s1: str, s2: str) -> int:
    """Compute the Levenshtein edit distance (in characters) between two strings."""
    # Use dynamic programming for edit distance
    n, m = len(s1), len(s2)
    dp = [[0] * (m+1) for _ in range(n+1)]
    for i in range(n+1):
        dp[i][0] = i
    for j in range(m+1):
        dp[0][j] = j
    for i in range(1, n+1):
        for j in range(1, m+1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp[n][m]

def raidar_score(original: str, rewritten: str) -> float:
    """
    Compute the RAIDAR similarity score between original and rewritten text.
    We define score = 1 - (edit_distance / max(len(original), len(rewritten))).
    A higher score means the texts are more similar (fewer changes).
    """
    dist = character_edit_distance(original, rewritten)
    max_len = max(len(original), len(rewritten))
    if max_len == 0:
        return 1.0  # trivial case: empty text
    score = 1 - (dist / max_len)
    return score

# # Example usage on a single text:
# sample_text = "यह एक उदाहरण वाक्य है जिसे हमें पुनर्लेखन के लिए देना है।"  # (Hindi) e.g., "This is a sample sentence to rewrite."
# rewritten = rewrite_text_with_llm(sample_text, model_name="gemma-3-2b")
# score = raidar_score(sample_text, rewritten)
# print("Original:", sample_text)
# print("Rewritten:", rewritten)
# print("RAIDAR similarity score:", score)

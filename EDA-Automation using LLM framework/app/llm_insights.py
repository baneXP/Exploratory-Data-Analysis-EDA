import ollama
import json

MODEL_NAME = "mistral"  # ollama model (other model : claude, Gemma 3, Llamma 3 family and many more)


def generate_llm_insights(eda_context: dict) -> str:
    """
    Generates statistically grounded EDA insights using an LLM.
    Input must be structured EDA context not raw text.
    """

    prompt = f"""
You are an Data Scientist.

You are given structured exploratory data analysis (EDA) results in JSON format.
Your task is to:
1. Identify key trends and patterns.
2. Comment on variability and stability.
3. Highlight statistically meaningful observations.
4. Avoid hallucinating numbers not present in the data.

EDA RESULTS (JSON):
{json.dumps(eda_context, indent=2)}

Respond with a concise analytical summary.
"""

    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}]
        )
        return response["message"]["content"]

    except Exception as e:
        return f"⚠️ LLM analysis failed: {e}"

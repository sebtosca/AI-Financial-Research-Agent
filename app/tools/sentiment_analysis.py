import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict

from dotenv import load_dotenv
from langchain_core.tools import tool

from app.config import SENTIMENT_MODEL, SENTIMENT_PROVIDER, SENTIMENT_TEMPERATURE
from app.providers import build_chat_model

load_dotenv()

logger = logging.getLogger(__name__)

MODEL_NAME = SENTIMENT_MODEL
TEMPERATURE = SENTIMENT_TEMPERATURE


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validate_text(text: str) -> str:
    if not text or not text.strip():
        raise ValueError("Text for sentiment analysis cannot be empty")

    return text.strip()


def _validate_environment() -> None:
    if SENTIMENT_PROVIDER == "openai" and not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is missing from environment")


def _build_model():
    _validate_environment()

    return build_chat_model(
        provider=SENTIMENT_PROVIDER,
        model=MODEL_NAME,
        temperature=TEMPERATURE,
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE") or None,
    )


def _strip_markdown_json(raw: str) -> str:
    raw = raw.strip()

    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]

        if raw.startswith("json"):
            raw = raw[4:]

    return raw.strip()


def _fallback_sentiment(text: str, error: Exception) -> Dict[str, Any]:
    positive_words = {
        "growth",
        "profit",
        "gain",
        "success",
        "up",
        "positive",
        "strong",
        "beat",
        "record",
        "surge",
        "bullish",
    }

    negative_words = {
        "loss",
        "decline",
        "down",
        "weak",
        "risk",
        "concern",
        "negative",
        "miss",
        "drop",
        "lawsuit",
        "bearish",
    }

    text_lower = text.lower()

    pos_count = sum(1 for word in positive_words if word in text_lower)
    neg_count = sum(1 for word in negative_words if word in text_lower)

    if pos_count > neg_count:
        sentiment = "positive"
        score = min(1.0, 0.6 + pos_count * 0.05)
    elif neg_count > pos_count:
        sentiment = "negative"
        score = max(0.0, 0.4 - neg_count * 0.05)
    else:
        sentiment = "neutral"
        score = 0.5

    logger.warning(
        "Using fallback sentiment analysis | positive_matches=%d | negative_matches=%d",
        pos_count,
        neg_count,
    )

    return {
        "sentiment": sentiment,
        "score": round(score, 3),
        "confidence": 0.6,
        "reasoning": "Fallback keyword-based sentiment analysis.",
        "status": "success_fallback",
        "error": str(error),
        "source": "keyword_fallback",
        "timestamp": _utc_timestamp(),
    }


@tool
def analyze_sentiment(text: str) -> Dict[str, Any]:
    """
    Analyze the sentiment of financial text.

    Args:
        text: Financial text such as a headline, article snippet, or report excerpt.

    Returns:
        Dictionary containing sentiment, score, confidence, reasoning, status, and timestamp.
    """

    timestamp = _utc_timestamp()

    try:
        cleaned_text = _validate_text(text)

        logger.info(
            "Starting sentiment analysis | text_chars=%d | model=%s",
            len(cleaned_text),
            MODEL_NAME,
        )

        model = _build_model()

        prompt = f"""
Analyze the sentiment of the following financial text.

Return only valid JSON with this schema:
{{
  "sentiment": "positive" | "neutral" | "negative",
  "score": 0.0,
  "confidence": 0.0,
  "reasoning": "brief explanation"
}}

Scoring rules:
- 0.0 means very negative
- 0.5 means neutral
- 1.0 means very positive
- confidence should reflect uncertainty

Text:
{cleaned_text}
"""

        response = model.invoke(prompt)

        raw = _strip_markdown_json(response.content)
        result = json.loads(raw)

        sentiment = result.get("sentiment")

        if sentiment not in {"positive", "neutral", "negative"}:
            raise ValueError(f"Invalid sentiment label: {sentiment}")

        score = float(result.get("score"))
        confidence = float(result.get("confidence"))

        result = {
            "sentiment": sentiment,
            "score": max(0.0, min(1.0, score)),
            "confidence": max(0.0, min(1.0, confidence)),
            "reasoning": result.get("reasoning", ""),
            "status": "success",
            "source": "openai",
            "model": MODEL_NAME,
            "timestamp": timestamp,
        }

        logger.info(
            "Sentiment analysis completed | sentiment=%s | score=%.3f | confidence=%.3f",
            result["sentiment"],
            result["score"],
            result["confidence"],
        )

        return result

    except Exception as e:
        logger.exception("OpenAI sentiment analysis failed")

        try:
            return _fallback_sentiment(text, e)

        except Exception as fallback_error:
            logger.exception("Fallback sentiment analysis failed")

            return {
                "sentiment": "neutral",
                "score": 0.5,
                "confidence": 0.0,
                "reasoning": "Sentiment analysis failed.",
                "status": "error",
                "error": str(fallback_error),
                "source": "sentiment_analysis",
                "timestamp": timestamp,
            }
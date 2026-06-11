from typing import Dict
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
import json

load_dotenv()

@tool 
def analyze_sentiment(text: str) -> Dict:
    """
    Analyze the sentiment of financial text using OpenAI.

    This tool analyzes the sentiment (positive/negative.neutral) of news articles, reports, or any financial text.
    Returs a sentiment label or confidence score.

    Args: 
        text: Text to analyze (article, headline, report excerpt)
    
    Returns: 
        dict: {
            'sentiment': str ('positive', 'negative', or 'neutral')
            'score': float (0.0 to 1.0 where 1.0 is most psoitive)
            'confidence': float (0.0 to 1.0)
            'reasoning': sstr (brief explanation)
        }
    
    Examples: 
        >>> result = analyze_sentiment("Apple reports record earnings....")
        >>> print(f"Sentiment: {result['sentiment']} (score: {result['score']}))
    """
    try: 
        model = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            openai_api_key=os.environ.get("OPENAI_API_KEY"),
            openai_api_base=os.environ.get("OPENAI_API_BASE")
        )

        prompt = f""" Analyze the sentiment of this financial text and provide: 
        1. Sentiment label: positive, negative, or neutral
        2. Score 0.0 (very negative) to 1.0 (very positive), 0.5 is neutral
        3. Confidence: 0.0 to 1.0
        4. Biref reasoning 

        Text: {text}

        Respond in JSON format: 
        {{
            "sentiment": "positive|neutral|negative",
            "score": 0.0-1.0,
            "confidence": 0.0-1.0,
            "reasoning": "brief explanation
        }}"""

        response = model.invoke(prompt)
        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```", 2)[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        result = json.loads(raw)
        result['status'] = 'success'
        return result

    except Exception as e: 
        # Fallback to simple sentiment if OpenAI fails
        positive_words = ['growth', 'profit', 'gain', 'success', 'up', 'positive', 'strong']
        negative_words = ['loss', 'decline', 'down', 'weak', 'risk', 'concern', 'negative']

        text_lower = text.lower()
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)

        if pos_count > neg_count:
            sentiment = 'positive'
            score = 0.6 + (pos_count * 0.05)
        elif neg_count > pos_count:
            sentiment = 'negative'
            score = 0.4 - (neg_count * 0.05)
        else:
            sentiment = 'neutral'
            score = 0.5

        return {
            'sentiment': sentiment,
            'score': max(0.0, min(1.0, score)),
            'confidence': 0.6,
            'reasoning': 'Fallback keyword-based analysis',
            'status': 'success (fallback)',
            'note': f'OpenAI analysis failed: {str(e)}'
        }
"""Stage 1 - The Matchmaker: find the cards that best match what the user typed.

How it works:
1. Every card's text (name, category, rewards, description) is turned into a TF-IDF
   vector: a list of numbers saying how important each word is to that card.
   Words that appear on every card ("card", "credit") get low weight; distinctive
   words ("lounge", "groceries", "student") get high weight.
2. The user's search is turned into a vector the same way.
3. Cosine similarity measures the angle between the search vector and each card
   vector: 1.0 means the same direction (very similar), 0.0 means nothing in common.
4. The top-k highest scores win. If nothing clears a minimum score, we return no
   matches instead of showing irrelevant cards.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import pairwise

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from backend.config import DEFAULT_TOP_K, MIN_MATCH_SCORE
from backend.schemas import Card

# "no" is a stop word in most NLP libraries, but in finance it changes everything
# ("no annual fee", "no foreign transaction fee"), so we keep it.
STOP_WORDS = ENGLISH_STOP_WORDS - {"no"}
TOKEN_RE = re.compile(r"[a-z0-9]+")


def normalize_token(token: str) -> str:
    """A tiny stemmer so related word forms match.

    "fees" -> "fee", "groceries" -> "grocery", "building" -> "build", "shopping" -> "shop".
    """
    if len(token) > 5 and token.endswith("ing"):
        token = token[:-3]
        if token[-1] == token[-2] and token[-1] != "s":  # shopp -> shop, travell -> travel
            token = token[:-1]
        return token
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def analyze(text: str) -> list[str]:
    """Lowercase -> split into words -> drop stop words -> stem -> add two-word phrases (bigrams).

    Bigrams let "no foreign" and "annual fee" count as phrases, not just loose words.
    """
    words = [normalize_token(t) for t in TOKEN_RE.findall(text.lower()) if t not in STOP_WORDS]
    return words + [f"{a} {b}" for a, b in pairwise(words)]


def card_document(card: Card) -> str:
    """Build the searchable text for a card, adding tags derived from its numeric fields."""
    tags = [card.category.replace("-", " ")]
    if card.foreign_transaction_fee == 0:
        tags.append("no foreign transaction fee")
    if card.annual_fee == 0:
        tags.append("no annual fee")
    if card.purchase_apr < 15:
        tags.append("low interest rate")
    return " ".join([card.name, *tags, card.rewards, card.description])


@dataclass(frozen=True)
class Match:
    card: Card
    score: float


class CardMatcher:
    """Fits a TF-IDF model on the card catalogue once, then answers searches from memory."""

    def __init__(self, cards: list[Card]):
        if not cards:
            raise ValueError("CardMatcher needs at least one card")
        self.cards = cards
        self.vectorizer = TfidfVectorizer(analyzer=analyze, sublinear_tf=True)
        self.card_matrix = self.vectorizer.fit_transform([card_document(c) for c in cards])

    def search(self, query: str, top_k: int = DEFAULT_TOP_K, min_score: float = MIN_MATCH_SCORE) -> list[Match]:
        query_vector = self.vectorizer.transform([query])
        if query_vector.nnz == 0:  # none of the search words appear in any card
            return []
        scores = cosine_similarity(query_vector, self.card_matrix).ravel()
        ranked = np.argsort(-scores, kind="stable")[:top_k]  # stable: ties keep catalogue order
        return [Match(self.cards[i], float(scores[i])) for i in ranked if scores[i] >= min_score]

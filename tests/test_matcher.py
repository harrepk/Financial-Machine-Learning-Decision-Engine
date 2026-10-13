import pytest

from backend.matcher import analyze, normalize_token


@pytest.mark.parametrize(
    ("token", "expected"),
    [
        ("fees", "fee"),
        ("groceries", "grocery"),
        ("students", "student"),
        ("building", "build"),
        ("shopping", "shop"),
        ("travelling", "travel"),
        ("business", "business"),
        ("gas", "gas"),
    ],
)
def test_normalize_token(token, expected):
    assert normalize_token(token) == expected


def test_analyze_keeps_no_and_builds_bigrams():
    tokens = analyze("No foreign transaction fees")
    assert "no" in tokens  # "no" is meaningful in finance, so it is not treated as a stop word
    assert "no foreign" in tokens
    assert "transaction fee" in tokens


def test_travel_search_finds_travel_cards_without_fx_fees(matcher):
    matches = matcher.search("no foreign transaction fee travel card", top_k=3)

    assert len(matches) == 3
    assert matches[0].card.category == "travel"
    assert all(m.card.foreign_transaction_fee == 0 for m in matches)


@pytest.mark.parametrize(
    ("query", "expected_category"),
    [
        ("student card to build credit", "student"),
        ("low interest card to pay off debt", "low-interest"),
        ("cash back on groceries", "cashback"),
        ("secured card with a deposit", "secured"),
    ],
)
def test_top_match_has_expected_category(matcher, query, expected_category):
    assert matcher.search(query, top_k=1)[0].card.category == expected_category


def test_scores_are_sorted_and_respect_top_k(matcher):
    matches = matcher.search("rewards card", top_k=5)
    scores = [m.score for m in matches]

    assert len(matches) == 5
    assert scores == sorted(scores, reverse=True)
    assert all(0 < s <= 1 for s in scores)


def test_unrelated_search_returns_no_matches(matcher):
    assert matcher.search("qwzx plorf blarg") == []

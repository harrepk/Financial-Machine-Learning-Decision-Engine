import json

from backend import config
from backend.database import fetch_all_cards, fetch_cards_by_ids, init_db


def test_init_db_seeds_every_card_once(tmp_path):
    db_path = tmp_path / "cards.db"
    seed_count = len(json.loads(config.CARDS_SEED_PATH.read_text(encoding="utf-8")))

    assert init_db(db_path, config.CARDS_SEED_PATH) == seed_count
    assert init_db(db_path, config.CARDS_SEED_PATH) == seed_count  # second startup must not duplicate rows
    assert len(fetch_all_cards(db_path)) == seed_count


def test_fetch_cards_by_ids_keeps_requested_order_and_skips_unknown(tmp_path):
    db_path = tmp_path / "cards.db"
    init_db(db_path, config.CARDS_SEED_PATH)

    cards = fetch_cards_by_ids(db_path, [3, 1, 999])

    assert [c.id for c in cards] == [3, 1]
    assert fetch_cards_by_ids(db_path, []) == []

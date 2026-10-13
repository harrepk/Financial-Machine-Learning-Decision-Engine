"""SQLite persistence layer for the credit card catalogue.

SQLite is a full relational database that lives in a single file (data/cards.db) and
ships with Python, so there is no database server to install or run.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path

from backend.schemas import Card

SCHEMA = """
CREATE TABLE IF NOT EXISTS credit_cards (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    name                    TEXT    NOT NULL UNIQUE,
    issuer                  TEXT    NOT NULL,
    category                TEXT    NOT NULL,
    tier                    INTEGER NOT NULL CHECK (tier BETWEEN 0 AND 3),
    annual_fee              REAL    NOT NULL CHECK (annual_fee >= 0),
    purchase_apr            REAL    NOT NULL CHECK (purchase_apr >= 0),
    foreign_transaction_fee REAL    NOT NULL CHECK (foreign_transaction_fee >= 0),
    min_income              INTEGER NOT NULL DEFAULT 0,
    rewards                 TEXT    NOT NULL,
    description             TEXT    NOT NULL
);
"""

INSERT_CARD = """
INSERT INTO credit_cards (name, issuer, category, tier, annual_fee, purchase_apr,
                          foreign_transaction_fee, min_income, rewards, description)
VALUES (:name, :issuer, :category, :tier, :annual_fee, :purchase_apr,
        :foreign_transaction_fee, :min_income, :rewards, :description)
"""


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row  # rows behave like dicts: row["name"]
    return conn


def init_db(db_path: Path, seed_path: Path) -> int:
    """Create the table and load the seed cards if it is empty. Safe to call on every startup.

    Returns the number of cards in the table.
    """
    with closing(connect(db_path)) as conn, conn:  # second `conn` commits the transaction
        conn.executescript(SCHEMA)
        (count,) = conn.execute("SELECT COUNT(*) FROM credit_cards").fetchone()
        if count == 0:
            cards = json.loads(seed_path.read_text(encoding="utf-8"))
            conn.executemany(INSERT_CARD, cards)
            count = len(cards)
    return count


def fetch_all_cards(db_path: Path) -> list[Card]:
    with closing(connect(db_path)) as conn:
        rows = conn.execute("SELECT * FROM credit_cards ORDER BY id").fetchall()
    return [Card(**dict(row)) for row in rows]


def fetch_cards_by_ids(db_path: Path, card_ids: list[int]) -> list[Card]:
    """Return the requested cards in the same order as `card_ids`, skipping unknown ids."""
    if not card_ids:
        return []
    placeholders = ",".join("?" * len(card_ids))  # parameterised query: no SQL injection
    with closing(connect(db_path)) as conn:
        rows = conn.execute(f"SELECT * FROM credit_cards WHERE id IN ({placeholders})", card_ids).fetchall()
    by_id = {row["id"]: Card(**dict(row)) for row in rows}
    return [by_id[card_id] for card_id in card_ids if card_id in by_id]

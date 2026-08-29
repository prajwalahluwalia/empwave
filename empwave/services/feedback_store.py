"""Persistent, review-only storage for user-reported emotion corrections."""

import hashlib
import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path


CONSENT_VERSION = "emotion-feedback-v1"
MAX_SELECTED_EMOTIONS = 10
UNUSUALLY_MANY_EMOTIONS = 6


def normalize_text(text):
    return " ".join(text.strip().lower().split())


def text_fingerprint(text):
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


class FeedbackStore:
    def __init__(self, database_path):
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self):
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _initialize(self):
        with closing(self._connect()) as connection:
            with connection:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS emotion_feedback (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        created_at TEXT NOT NULL,
                        text TEXT NOT NULL,
                        text_hash TEXT NOT NULL,
                        perspective TEXT NOT NULL,
                        selected_emotions TEXT NOT NULL,
                        predicted_emotions TEXT NOT NULL,
                        model_name TEXT NOT NULL,
                        consent_version TEXT NOT NULL,
                        moderation_status TEXT NOT NULL,
                        moderation_flags TEXT NOT NULL,
                        reviewed_at TEXT,
                        review_note TEXT
                    )
                    """
                )
                columns = {
                    row["name"]
                    for row in connection.execute(
                        "PRAGMA table_info(emotion_feedback)"
                    )
                }
                if "reviewed_at" not in columns:
                    connection.execute(
                        "ALTER TABLE emotion_feedback "
                        "ADD COLUMN reviewed_at TEXT"
                    )
                if "review_note" not in columns:
                    connection.execute(
                        "ALTER TABLE emotion_feedback "
                        "ADD COLUMN review_note TEXT"
                    )
                connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS
                    idx_emotion_feedback_text_hash
                    ON emotion_feedback (text_hash)
                    """
                )

    def submit(
        self,
        *,
        text,
        selected_emotions,
        predicted_emotions,
        model_name,
    ):
        selected = sorted(set(selected_emotions))
        if not selected:
            raise ValueError("Select at least one emotion.")
        if len(selected) > MAX_SELECTED_EMOTIONS:
            raise ValueError(
                f"Select no more than {MAX_SELECTED_EMOTIONS} emotions."
            )
        if "neutral" in selected and len(selected) > 1:
            raise ValueError(
                "Neutral cannot be combined with other emotions."
            )

        text_hash = text_fingerprint(text)
        selected_json = json.dumps(selected, separators=(",", ":"))
        predicted_json = json.dumps(
            predicted_emotions,
            separators=(",", ":"),
            sort_keys=True,
        )

        with closing(self._connect()) as connection:
            with connection:
                repeated_text_count = connection.execute(
                    """
                    SELECT COUNT(*) FROM emotion_feedback
                    WHERE text_hash = ?
                    """,
                    (text_hash,),
                ).fetchone()[0]
                exact_duplicate_count = connection.execute(
                    """
                    SELECT COUNT(*) FROM emotion_feedback
                    WHERE text_hash = ? AND selected_emotions = ?
                    """,
                    (text_hash, selected_json),
                ).fetchone()[0]

                predicted_ids = {
                    emotion["id"]
                    for emotion in predicted_emotions
                    if emotion.get("id") != "neutral"
                }
                predicted_peak = max(
                    (
                        float(emotion.get("score", 0))
                        for emotion in predicted_emotions
                        if emotion.get("id") != "neutral"
                    ),
                    default=0,
                )
                selected_non_neutral = set(selected) - {"neutral"}

                flags = []
                if len(selected) >= UNUSUALLY_MANY_EMOTIONS:
                    flags.append("unusually_many_emotions")
                if repeated_text_count >= 3:
                    flags.append("repeated_text_submission")
                if exact_duplicate_count:
                    flags.append("duplicate_feedback")
                if (
                    predicted_peak >= 0.95
                    and selected_non_neutral
                    and predicted_ids.isdisjoint(selected_non_neutral)
                ):
                    flags.append("strong_model_disagreement")

                status = "priority_review" if flags else "pending_review"
                cursor = connection.execute(
                    """
                    INSERT INTO emotion_feedback (
                        created_at,
                        text,
                        text_hash,
                        perspective,
                        selected_emotions,
                        predicted_emotions,
                        model_name,
                        consent_version,
                        moderation_status,
                        moderation_flags
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        datetime.now(timezone.utc).isoformat(),
                        text,
                        text_hash,
                        "speaker_self_report",
                        selected_json,
                        predicted_json,
                        model_name,
                        CONSENT_VERSION,
                        status,
                        json.dumps(flags, separators=(",", ":")),
                    ),
                )
                feedback_id = cursor.lastrowid

        return {
            "feedback_id": feedback_id,
            "moderation_status": status,
            "moderation_flags": flags,
        }

    def list_pending(self, limit=50):
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT id, created_at, text, perspective, selected_emotions,
                       predicted_emotions, model_name, moderation_status,
                       moderation_flags
                FROM emotion_feedback
                WHERE moderation_status IN ('pending_review', 'priority_review')
                ORDER BY
                    CASE moderation_status
                        WHEN 'priority_review' THEN 0
                        ELSE 1
                    END,
                    id
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [
            {
                **dict(row),
                "selected_emotions": json.loads(row["selected_emotions"]),
                "predicted_emotions": json.loads(row["predicted_emotions"]),
                "moderation_flags": json.loads(row["moderation_flags"]),
            }
            for row in rows
        ]

    def review(self, feedback_id, decision, note=""):
        if decision not in {"approved", "rejected"}:
            raise ValueError("Decision must be approved or rejected.")
        with closing(self._connect()) as connection:
            with connection:
                cursor = connection.execute(
                    """
                    UPDATE emotion_feedback
                    SET moderation_status = ?, reviewed_at = ?, review_note = ?
                    WHERE id = ?
                      AND moderation_status IN (
                          'pending_review',
                          'priority_review'
                      )
                    """,
                    (
                        decision,
                        datetime.now(timezone.utc).isoformat(),
                        note.strip(),
                        feedback_id,
                    ),
                )
        if cursor.rowcount != 1:
            raise ValueError(
                "Feedback was not found or has already been reviewed."
            )


@lru_cache(maxsize=8)
def get_feedback_store(database_path):
    return FeedbackStore(database_path)

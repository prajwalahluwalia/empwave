#!/usr/bin/env python3
"""Download the sentence encoder during the Render build."""

from sentence_transformers import SentenceTransformer


SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
print("Cached sentence-transformers/all-MiniLM-L6-v2")

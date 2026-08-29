# Empwave Deployment Strategy

## Current deployment

- The Flask application exposes `POST /simulate` and `GET /health`.
- Gunicorn runs one preloaded worker with four threads so the semantic model is
  initialized once and is not duplicated across worker processes.
- `ALLOWED_ORIGIN` must contain the deployed frontend origin, for example
  `https://example.com`. Requests carrying another origin are rejected.
- The Docker image downloads `sentence-transformers/all-MiniLM-L6-v2` during
  the build and verifies that
  `models/trained/empwave_classifier.joblib` exists.
- The browser remains responsible for speech synthesis. `/simulate` returns
  `spoken_text`, which can be passed to `window.speechSynthesis`.

## Deferred backend TTS

Backend-generated audio is intentionally deferred because this repository has
no existing server-side TTS implementation and no additional TTS dependency
was approved.

If server audio becomes necessary, prefer an offline eSpeak NG integration:

1. Install `espeak-ng` in the Docker image.
2. Generate WAV bytes from the `/simulate` `spoken_text`.
3. Return the bytes as a base64 data field or from a short-lived audio route.
4. Add output-size limits, timeouts, and tests that decode and inspect the WAV
   header.

An online service such as gTTS is smaller to integrate but adds network
latency, external data processing, and an additional failure mode.

## Hugging Face Spaces

1. Install Git LFS locally and run `git lfs install`.
2. Re-add the tracked model artifact so the `.gitattributes` rule is applied:
   `git rm --cached models/trained/empwave_classifier.joblib` followed by
   `git add models/trained/empwave_classifier.joblib`.
3. Set the Space secret `ALLOWED_ORIGIN` to the frontend origin.
4. Push the repository to a Docker Space. The container listens on port 7860.

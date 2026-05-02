# Local Recipe Generator

A small command-line app that turns a mood ("something warm and spicy", "light summer dinner") into a complete recipe with a generated food photo — running fully on your own machine. It uses a local Gemma model via Ollama for the text (dish suggestions, recipes, image prompts) and FLUX.1-schnell via `diffusers` for the image. Output is saved as a Markdown file with the embedded image alongside the PNG.

The accompanying YouTube video is **coming soon** on [GammaVibe](https://www.youtube.com/@GammaVibeLab).

## Dependencies

Before running, you need:

- **[uv](https://docs.astral.sh/uv/)** — Python package manager / runner
- **Python 3.12+** (uv will handle this for you)
- **[Ollama](https://ollama.com/)** running locally on `http://localhost:11434`, with the `gemma4:e4b` model installed:
  ```
  ollama pull gemma4:e4b
  ```
- **A CUDA-capable NVIDIA GPU** for FLUX.1-schnell. The pyproject is wired to the CUDA 12.6 PyTorch wheels. The first run downloads the FLUX weights (~24 GB) into the Hugging Face cache.
- **Hugging Face access** to [`black-forest-labs/FLUX.1-schnell`](https://huggingface.co/black-forest-labs/FLUX.1-schnell) — accept the model license on the Hub and run `huggingface-cli login` if you haven't already.

Python dependencies (resolved by uv from `pyproject.toml`):

- `pydantic-ai` — structured LLM outputs against Ollama's OpenAI-compatible API
- `diffusers`, `transformers`, `accelerate` — FLUX.1-schnell pipeline
- `torch`, `torchvision` — CUDA build pinned to the `pytorch-cuda` index

## Running

From this folder:

```
uv run main.py
```

uv will create a virtualenv, install dependencies on first run, and start the app. You'll be prompted for a mood, shown 3-5 dish suggestions, asked to pick one, and then the app will generate the full recipe and a food photo. Both end up in the `output/` directory as `recipe_<timestamp>.md` and `recipe_<timestamp>.png`.

---

## What this is (and isn't)

This is **proof-of-concept code** — minimal, focused, and meant to demonstrate ideas. It's not production-grade, and you won't find tests, CI pipelines, or comprehensive error handling here. The goal is clarity over completeness.

If you're interested in building production-grade AI systems, grab the [Agentic Pipeline Blueprint](https://gammavibe.com/blueprint) — a free starting point for building an autonomous research pipeline, and the waitlist for my upcoming course.

## About GammaVibe

GammaVibe helps senior engineers build real-world AI systems using agentic engineering workflows.

- [YouTube](https://www.youtube.com/@GammaVibeLab)
- [GammaVibe Labs](https://gammavibe.com) — newsletter, community, and more
- [Agentic Pipeline Blueprint](https://gammavibe.com/blueprint) — free guide + course waitlist
- [X / Twitter](https://x.com/gammavibe)
- [Bluesky](https://bsky.app/profile/gammavibe.bsky.social)

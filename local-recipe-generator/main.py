import asyncio
from datetime import datetime
from typing import Literal
from diffusers import FluxPipeline
import logging
from pathlib import Path
import time

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.output import NativeOutput

import torch

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# gemma4:e4b — small and fast, and reliable enough at structured output that
# pydantic-ai can validate the schemas below without constant retries. Bigger
# models give better prose but stop being practical for a fully-local app.
MODEL_NAME = "gemma4:e4b"
OLLAMA_HOST = "http://localhost:11434"
OLLAMA_URL = f"{OLLAMA_HOST}/v1"

# FLUX.1-schnell — 4-step distilled variant of FLUX.1. Produces a usable image
# in seconds rather than the 30+ steps the full dev model wants, and the
# weights still fit on a single consumer GPU once we offload (see below).
IMAGE_MODEL_NAME = "black-forest-labs/FLUX.1-schnell"

# Low temperature keeps Gemma on-schema. At the provider default (~0.7) it
# occasionally invents fields or drifts off structure, which triggers
# pydantic-ai retries; 0.2 is low enough to stay in spec but still gives some
# variety across dish suggestions.
TEMPERATURE = 0.2


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("recipe")
# Show every HTTP request/response to Ollama (method, URL, status).
logging.getLogger("httpx").setLevel(logging.INFO)
# Pydantic AI's own debug chatter — useful to see retries, tool calls, etc.
logging.getLogger("pydantic_ai").setLevel(logging.DEBUG)


# ---------------------------------------------------------------------------
# Output schemas (what we ask Gemma to return; pydantic-ai validates these)
# ---------------------------------------------------------------------------


class Dish(BaseModel):
    """A single dish suggestion: short name plus a one-line description."""

    name: str = Field(
        max_length=80,
        description="Short dish name, e.g. 'Crispy Corn Fritters'. Max 80 characters.",
    )
    one_line_description: str = Field(
        max_length=200,
        description="A single-sentence description of the dish, under 20 words. Max 200 characters.",
    )


class DishList(BaseModel):
    """A handful of dish suggestions for the user to pick from."""

    dishes: list[Dish] = Field(
        min_length=3,
        max_length=5,
        description="Between 3 and 5 dish suggestions matching the user's mood.",
    )


class Ingredient(BaseModel):
    """One ingredient line: name and a free-text quantity (e.g. '2 cups')."""

    name: str = Field(
        max_length=80,
        description="Ingredient name, e.g. 'all-purpose flour'.",
    )
    quantity: str = Field(
        max_length=40,
        description="Quantity with units as a free-text string, e.g. '2 cups' or '1 tbsp'.",
    )


class Recipe(BaseModel):
    """A complete recipe: description, ingredient list, and ordered steps."""

    dish_name: str = Field(
        max_length=80,
        description="The name of the dish this recipe produces.",
    )
    description: str = Field(
        max_length=600,
        description="A short paragraph describing the dish, its origin, or flavor profile.",
    )
    ingredients: list[Ingredient] = Field(
        max_length=30,
        description="Every ingredient required, each with a name and quantity.",
    )
    preparation_steps: list[str] = Field(
        max_length=20,
        description="Ordered preparation steps. Each list item is one step as a full sentence (max 400 chars). DO NOT add numbering or 'Step 1' etc. — just the instruction text.",
    )


class ImagePrompt(BaseModel):
    """A FLUX-ready image prompt plus a hint at what the image should depict."""

    subject: Literal["final_dish", "preparation_stage"] = Field(
        description=(
            "What the image should depict. Choose 'final_dish' for a plated "
            "shot of the finished recipe, or 'preparation_stage' for a "
            "compelling interim moment (e.g. ingredients laid out, dough "
            "being kneaded, sauce reducing). Pick whichever is more visually "
            "interesting for THIS recipe."
        )
    )
    prompt: str = Field(
        max_length=300,
        description=(
            "A concise FLUX-style image prompt, 40-55 words, under 300 "
            "characters. Describe subject, composition, lighting, and mood. "
            "FLUX's CLIP encoder only reads the first 77 tokens (~50 words), "
            "so every word counts. Do NOT include negative prompts."
        ),
    )


# ---------------------------------------------------------------------------
# LLM agents (Gemma via Ollama's OpenAI-compatible API)
# ---------------------------------------------------------------------------


def build_model() -> OpenAIChatModel:
    # Ollama exposes an OpenAI-compatible API at /v1, so we drive it through
    # pydantic-ai's OpenAI provider with a placeholder api_key — Ollama
    # ignores it but the SDK requires the field.
    provider = OpenAIProvider(base_url=OLLAMA_URL, api_key="ollama")
    return OpenAIChatModel(MODEL_NAME, provider=provider)


async def suggest_dishes(model: OpenAIChatModel, mood: str) -> DishList:
    agent = Agent(
        model,
        output_type=NativeOutput(DishList),
        output_retries=3,
        model_settings=OpenAIChatModelSettings(temperature=TEMPERATURE),
        system_prompt=(
            "You are a creative chef. Given the user's mood or craving, "
            "suggest dishes that fit. Favor variety across cuisines, "
            "textures, and preparation styles rather than minor variants "
            "of the same dish."
        ),
    )
    log.info("suggest_dishes: sending request to %s (model=%s)", OLLAMA_URL, MODEL_NAME)
    t0 = time.perf_counter()
    result = await agent.run(f"I'm in the mood for: {mood}")
    log.info("suggest_dishes: got %d dishes in %.1fs", len(result.output.dishes), time.perf_counter() - t0)
    return result.output


async def get_recipe(model: OpenAIChatModel, dish_name: str) -> Recipe:
    agent = Agent(
        model,
        output_type=NativeOutput(Recipe),
        output_retries=3,
        model_settings=OpenAIChatModelSettings(temperature=TEMPERATURE),
        system_prompt=(
            "You are a chef. Given a dish name, produce a complete recipe "
            "that a home cook could realistically follow — sensible "
            "quantities, common ingredients where possible, and steps in "
            "the order a cook would actually do them."
            "DO NOT add numbering or 'Step 1' etc. to the preparation steps — just the instruction text in each list item."
        ),
    )
    log.info("get_recipe: sending request for dish=%r", dish_name)
    t0 = time.perf_counter()
    result = await agent.run(f"Give me a recipe for: {dish_name}")
    log.info("get_recipe: received recipe in %.1fs", time.perf_counter() - t0)
    return result.output


async def generate_image_prompt(model: OpenAIChatModel, recipe: Recipe) -> ImagePrompt:
    agent = Agent(
        model,
        output_type=NativeOutput(ImagePrompt),
        output_retries=3,
        model_settings=OpenAIChatModelSettings(temperature=TEMPERATURE),
        system_prompt=(
            "You are a food photography art director. Given a recipe, produce "
            "a single concise image prompt for FLUX.1.\n\n"
            "Decide whether a plated final dish or an interim preparation "
            "moment would be most visually compelling, then write a prompt of "
            "40-55 words (under 300 characters). Cover subject, camera angle, "
            "lighting, and mood in compact evocative phrases — FLUX's CLIP "
            "encoder only sees the first 77 tokens (~50 words), so brevity "
            "matters. Favor natural light and shallow depth of field. Do NOT "
            "describe what you DON'T want."
        ),
    )
    log.info("generate_image_prompt: asking Gemma for image prompt")
    t0 = time.perf_counter()
    result = await agent.run(f"Recipe:\n{recipe.model_dump_json(indent=2)}")
    log.info("generate_image_prompt: got prompt in %.1fs", time.perf_counter() - t0)
    return result.output


# ---------------------------------------------------------------------------
# CLI and Markdown helpers
# ---------------------------------------------------------------------------


def pick_dish(dishes: list[Dish]) -> Dish:
    print("\nSuggested dishes:")
    for i, d in enumerate(dishes, 1):
        print(f"  {i}. {d.name} — {d.one_line_description}")
    while True:
        raw = input(f"\nPick a dish [1-{len(dishes)}]: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(dishes):
            return dishes[int(raw) - 1]
        print("Invalid choice, try again.")


def recipe_to_markdown(recipe: Recipe, image_filename: str) -> str:
    lines = [
        f"![{recipe.dish_name}]({image_filename})",
        "",
        f"# {recipe.dish_name}",
        "",
        recipe.description,
        "",
        "## Ingredients",
        "",
    ]
    lines.extend(f"- {ing.quantity} {ing.name}" for ing in recipe.ingredients)
    lines.extend(["", "## Instructions", ""])
    lines.extend(f"{i}. {step}" for i, step in enumerate(recipe.preparation_steps, 1))
    lines.append("")
    return "\n".join(lines)


def print_recipe(recipe: Recipe) -> None:
    print("\n" + "=" * 60)
    print(f"  {recipe.dish_name}")
    print("=" * 60)
    print(f"\n{recipe.description}\n")
    print("Ingredients:")
    for ing in recipe.ingredients:
        print(f"  - {ing.quantity} {ing.name}")
    print("\nInstructions:")
    for i, step in enumerate(recipe.preparation_steps, 1):
        print(f"  {i}. {step}")
    print()


# ---------------------------------------------------------------------------
# Image generation (FLUX.1-schnell via diffusers)
# ---------------------------------------------------------------------------


def _load_pipeline() -> FluxPipeline:
    log.info("Loading image model...")
    t0 = time.perf_counter()
    pipe = FluxPipeline.from_pretrained(IMAGE_MODEL_NAME, torch_dtype=torch.bfloat16)
    # Sequential CPU offload swaps FLUX's submodules (text encoders, transformer,
    # VAE) on and off the GPU as they're needed, keeping peak VRAM low. We do
    # this because Ollama still has Gemma resident in GPU memory, and loading
    # FLUX fully alongside it would OOM most consumer cards.
    #
    # Faster alternative if you don't need Gemma any more: shell out to
    # `ollama stop gemma4:e4b` first, drop this offload call, and let FLUX
    # run entirely on the GPU. We don't bother here because the demo is short.
    #
    # device="cuda:0" is required, not optional: with offload enabled
    # diffusers needs to know which device to swap modules onto, and it
    # won't infer a default when multiple GPUs are present.
    pipe.enable_sequential_cpu_offload(device="cuda:0")
    # VAE tiling decodes the final latent in chunks instead of one big tensor,
    # which keeps the last step from spiking VRAM at high resolutions.
    pipe.vae.enable_tiling()
    log.info("FLUX loaded with sequential CPU offload in %.1fs", time.perf_counter() - t0)

    return pipe


def generate_image(
    prompt: str,
    output_path: Path,
    width: int = 1344,
    height: int = 768,
    seed: int | None = None,
) -> Path:
    pipe = _load_pipeline()
    generator = (
        torch.Generator("cpu").manual_seed(seed) if seed is not None else None
    )
    preview = prompt if len(prompt) <= 80 else prompt[:80] + "..."
    log.info("generate_image: %dx%d, prompt=%r", width, height, preview)
    t0 = time.perf_counter()
    # FLUX.1-schnell is distilled to 4 steps with no classifier-free guidance —
    # these two values are part of the model contract, not knobs to tune.
    image = pipe(
        prompt,
        width=width,
        height=height,
        num_inference_steps=4,
        guidance_scale=0.0,
        generator=generator,
    ).images[0]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    log.info("generate_image: saved %s in %.1fs", output_path, time.perf_counter() - t0)
    return output_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main():
    model = build_model()
    mood = input("What are you in the mood for? ").strip()

    print("\nAsking the LLM for dish suggestions...")
    dish_list = await suggest_dishes(model, mood)
    chosen = pick_dish(dish_list.dishes)

    print(f"\nGetting the full recipe for '{chosen.name}'...")
    recipe = await get_recipe(model, chosen.name)
    print_recipe(recipe)


    print("Asking Gemma for an image prompt...")
    img_prompt = await generate_image_prompt(model, recipe)
    print(f"\nImage subject: {img_prompt.subject}")
    print(f"Image prompt: {img_prompt.prompt}\n")

    stem = f"recipe_{datetime.now():%Y%m%d_%H%M%S}"
    image_path = Path("output") / f"{stem}.png"
    markdown_path = Path("output") / f"{stem}.md"

    print("Generating image with FLUX.1-schnell (first run downloads ~24GB)...")
    generate_image(img_prompt.prompt, image_path)
    print(f"\n✔ Image saved to {image_path.resolve()}")

    markdown_path.write_text(
        recipe_to_markdown(recipe, image_path.name), encoding="utf-8"
    )
    print(f"✔ Recipe saved to {markdown_path.resolve()}")


if __name__ == "__main__":
    asyncio.run(main())

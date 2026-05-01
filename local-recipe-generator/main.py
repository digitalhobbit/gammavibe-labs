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

MODEL_NAME = "gemma4:e4b"
OLLAMA_HOST = "http://localhost:11434"
OLLAMA_URL = f"{OLLAMA_HOST}/v1"
IMAGE_MODEL_NAME = "black-forest-labs/FLUX.1-schnell"
TEMPERATURE = 0.2

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


class Dish(BaseModel):
    name: str = Field(
        max_length=80,
        description="Short dish name, e.g. 'Crispy Corn Fritters'. Max 80 characters.",
    )
    one_line_description: str = Field(
        max_length=200,
        description="A single-sentence description of the dish, under 20 words. Max 200 characters.",
    )


class DishList(BaseModel):
    dishes: list[Dish] = Field(
        min_length=3,
        max_length=5,
        description="Between 3 and 5 dish suggestions matching the user's mood.",
    )


class Ingredient(BaseModel):
    name: str = Field(
        max_length=80,
        description="Ingredient name, e.g. 'all-purpose flour'.",
    )
    quantity: str = Field(
        max_length=40,
        description="Quantity with units as a free-text string, e.g. '2 cups' or '1 tbsp'.",
    )


class Recipe(BaseModel):
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
        description="Ordered preparation steps. Each list item is one step as a full sentence (max 400 chars).",
    )


class ImagePrompt(BaseModel):
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


def build_model() -> OpenAIChatModel:
    # Ollama exposes an OpenAI-compatible API at /v1
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


def pick_dish(dishes: list[Dish]) -> Dish:
    print("\nSuggested dishes:")
    for i, d in enumerate(dishes, 1):
        print(f"  {i}. {d.name} — {d.one_line_description}")
    while True:
        raw = input(f"\nPick a dish [1-{len(dishes)}]: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(dishes):
            return dishes[int(raw) - 1]
        print("Invalid choice, try again.")


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


def _load_pipeline() -> FluxPipeline:
    log.info("Loading image model...")
    t0 = time.perf_counter()
    pipe = FluxPipeline.from_pretrained(IMAGE_MODEL_NAME, torch_dtype=torch.bfloat16)
    pipe.enable_sequential_cpu_offload(device="cuda:0")
    # VAE tiling keeps the final decode from spiking VRAM at high resolutions.
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

    out_path = Path("output") / f"recipe_{datetime.now():%Y%m%d_%H%M%S}.png"
    print("Generating image with FLUX.1-schnell (first run downloads ~24GB)...")
    generate_image(img_prompt.prompt, out_path)
    print(f"\n✔ Image saved to {out_path.resolve()}")


if __name__ == "__main__":
    asyncio.run(main())

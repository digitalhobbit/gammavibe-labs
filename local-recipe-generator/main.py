import asyncio
import logging
import time

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.output import ToolOutput

MODEL_NAME = "gemma4:e4b"
OLLAMA_HOST = "http://localhost:11434"
OLLAMA_URL = f"{OLLAMA_HOST}/v1"

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


def build_model() -> OpenAIChatModel:
    # Ollama exposes an OpenAI-compatible API at /v1
    provider = OpenAIProvider(base_url=OLLAMA_URL, api_key="ollama")
    return OpenAIChatModel(MODEL_NAME, provider=provider)


async def suggest_dishes(model: OpenAIChatModel, mood: str) -> DishList:
    agent = Agent(
        model,
        output_type=ToolOutput(DishList),
        output_retries=3,
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
        output_type=ToolOutput(Recipe),
        output_retries=3,
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


async def main():
    model = build_model()
    mood = input("What are you in the mood for? ").strip()

    print("\nAsking the LLM for dish suggestions...")
    dish_list = await suggest_dishes(model, mood)
    chosen = pick_dish(dish_list.dishes)

    print(f"\nGetting the full recipe for '{chosen.name}'...")
    recipe = await get_recipe(model, chosen.name)
    print_recipe(recipe)


if __name__ == "__main__":
    asyncio.run(main())

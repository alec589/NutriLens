"""Dish classes and metadata for the NutriLens CNN food classifier (24 real Food-101 US meals)."""

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class DishClassInfo:
    """Metadata for a recognizable dish class."""

    class_id: int
    key: str
    name_en: str
    category: str
    default_portion_g: float
    default_ingredients: List[str]
    aliases: List[str]


DISH_CLASSES: List[DishClassInfo] = [
    DishClassInfo(
        class_id=0,
        key="caesar_salad",
        name_en="Caesar salad",
        category="salad",
        default_portion_g=260.0,
        default_ingredients=["romaine lettuce", "chicken breast", "parmesan", "croutons"],
        aliases=["chicken caesar salad", "salad"],
    ),
    DishClassInfo(
        class_id=1,
        key="greek_salad",
        name_en="Greek salad",
        category="salad",
        default_portion_g=250.0,
        default_ingredients=["cucumber", "tomatoes", "feta cheese", "olives", "olive oil"],
        aliases=["feta salad", "mediterranean salad"],
    ),
    DishClassInfo(
        class_id=2,
        key="caprese_salad",
        name_en="Caprese salad",
        category="salad",
        default_portion_g=220.0,
        default_ingredients=["mozzarella", "tomatoes", "basil", "olive oil"],
        aliases=["tomato mozzarella salad"],
    ),
    DishClassInfo(
        class_id=3,
        key="club_sandwich",
        name_en="Club sandwich",
        category="sandwich",
        default_portion_g=250.0,
        default_ingredients=["toasted bread", "turkey breast", "bacon", "lettuce", "tomato"],
        aliases=["turkey club", "sandwich", "deli sandwich"],
    ),
    DishClassInfo(
        class_id=4,
        key="grilled_cheese_sandwich",
        name_en="Grilled cheese sandwich",
        category="sandwich",
        default_portion_g=160.0,
        default_ingredients=["sourdough bread", "cheddar cheese", "butter"],
        aliases=["grilled cheese"],
    ),
    DishClassInfo(
        class_id=5,
        key="pulled_pork_sandwich",
        name_en="Pulled pork sandwich",
        category="sandwich",
        default_portion_g=260.0,
        default_ingredients=["brioche bun", "pulled pork", "bbq sauce", "coleslaw"],
        aliases=["bbq pork sandwich", "pulled pork"],
    ),
    DishClassInfo(
        class_id=6,
        key="hamburger",
        name_en="Hamburger",
        category="american_classic",
        default_portion_g=260.0,
        default_ingredients=["burger bun", "beef patty", "cheddar cheese", "lettuce", "tomato"],
        aliases=["cheeseburger", "beef burger", "burger"],
    ),
    DishClassInfo(
        class_id=7,
        key="hot_dog",
        name_en="Hot dog",
        category="american_classic",
        default_portion_g=150.0,
        default_ingredients=["hot dog bun", "beef frankfurter", "mustard", "ketchup"],
        aliases=["frankfurter", "chili dog"],
    ),
    DishClassInfo(
        class_id=8,
        key="pizza",
        name_en="Pizza",
        category="american_classic",
        default_portion_g=280.0,
        default_ingredients=["pizza crust", "tomato sauce", "mozzarella", "pepperoni"],
        aliases=["pepperoni pizza", "cheese pizza", "margherita pizza"],
    ),
    DishClassInfo(
        class_id=9,
        key="macaroni_and_cheese",
        name_en="Mac and cheese",
        category="american_classic",
        default_portion_g=300.0,
        default_ingredients=["macaroni pasta", "cheddar cheese", "milk", "butter"],
        aliases=["macaroni and cheese", "baked mac and cheese"],
    ),
    DishClassInfo(
        class_id=10,
        key="spaghetti_bolognese",
        name_en="Spaghetti bolognese",
        category="pasta",
        default_portion_g=350.0,
        default_ingredients=["spaghetti", "ground beef", "tomato marinara", "parmesan"],
        aliases=["spaghetti meat sauce", "pasta bolognese", "spaghetti"],
    ),
    DishClassInfo(
        class_id=11,
        key="lasagna",
        name_en="Lasagna",
        category="pasta",
        default_portion_g=320.0,
        default_ingredients=["lasagna noodles", "ground beef", "ricotta", "mozzarella", "marinara"],
        aliases=["meat lasagna", "baked lasagna"],
    ),
    DishClassInfo(
        class_id=12,
        key="grilled_salmon",
        name_en="Grilled salmon",
        category="seafood",
        default_portion_g=230.0,
        default_ingredients=["atlantic salmon", "olive oil", "lemon", "asparagus"],
        aliases=["baked salmon", "salmon fillet", "salmon and veggies"],
    ),
    DishClassInfo(
        class_id=13,
        key="steak",
        name_en="Steak",
        category="american_classic",
        default_portion_g=280.0,
        default_ingredients=["sirloin steak", "butter", "garlic", "roasted veggies"],
        aliases=["sirloin steak", "steak and veggies", "grilled steak"],
    ),
    DishClassInfo(
        class_id=14,
        key="tacos",
        name_en="Tacos",
        category="mexican_american",
        default_portion_g=240.0,
        default_ingredients=["corn tortillas", "seasoned beef", "lettuce", "cheddar cheese", "salsa"],
        aliases=["beef tacos", "chicken tacos", "street tacos"],
    ),
    DishClassInfo(
        class_id=15,
        key="breakfast_burrito",
        name_en="Breakfast burrito",
        category="breakfast",
        default_portion_g=270.0,
        default_ingredients=["flour tortilla", "scrambled eggs", "cheddar cheese", "black beans"],
        aliases=["egg burrito", "morning burrito"],
    ),
    DishClassInfo(
        class_id=16,
        key="pancakes",
        name_en="Pancakes",
        category="breakfast",
        default_portion_g=230.0,
        default_ingredients=["buttermilk pancakes", "maple syrup", "butter", "blueberries"],
        aliases=["pancakes and berries", "blueberry pancakes"],
    ),
    DishClassInfo(
        class_id=17,
        key="french_toast",
        name_en="French toast",
        category="breakfast",
        default_portion_g=210.0,
        default_ingredients=["brioche bread", "egg", "milk", "maple syrup", "butter"],
        aliases=["cinnamon french toast"],
    ),
    DishClassInfo(
        class_id=18,
        key="eggs_benedict",
        name_en="Eggs benedict",
        category="breakfast",
        default_portion_g=240.0,
        default_ingredients=["english muffin", "poached eggs", "canadian bacon", "hollandaise sauce"],
        aliases=["poached eggs benedict"],
    ),
    DishClassInfo(
        class_id=19,
        key="omelette",
        name_en="Omelette",
        category="breakfast",
        default_portion_g=210.0,
        default_ingredients=["eggs", "cheddar cheese", "spinach", "bell pepper"],
        aliases=["cheese omelette", "egg omelette", "omelet"],
    ),
    DishClassInfo(
        class_id=20,
        key="chicken_wings",
        name_en="Chicken wings",
        category="american_classic",
        default_portion_g=250.0,
        default_ingredients=["chicken wings", "buffalo sauce", "celery"],
        aliases=["buffalo wings", "hot wings", "wings"],
    ),
    DishClassInfo(
        class_id=21,
        key="fried_rice",
        name_en="Fried rice",
        category="grain_bowl",
        default_portion_g=320.0,
        default_ingredients=["white rice", "egg", "peas", "carrots", "soy sauce"],
        aliases=["chicken fried rice", "egg fried rice"],
    ),
    DishClassInfo(
        class_id=22,
        key="ramen",
        name_en="Ramen",
        category="noodles",
        default_portion_g=500.0,
        default_ingredients=["ramen noodles", "broth", "soft boiled egg", "pork", "scallions"],
        aliases=["tonkotsu ramen", "japanese ramen"],
    ),
    DishClassInfo(
        class_id=23,
        key="frozen_yogurt",
        name_en="Frozen yogurt",
        category="snack",
        default_portion_g=180.0,
        default_ingredients=["greek frozen yogurt", "blueberries", "strawberries", "granola"],
        aliases=["froyo", "berry frozen yogurt"],
    ),
]

DISH_BY_KEY: Dict[str, DishClassInfo] = {d.key: d for d in DISH_CLASSES}
DISH_BY_ID: Dict[int, DishClassInfo] = {d.class_id: d for d in DISH_CLASSES}
NUM_CLASSES: int = len(DISH_CLASSES)

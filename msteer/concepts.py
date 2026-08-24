"""Concepts, the words that build a steering vector, and the words that score it.

Two disjoint lexicons per concept. ``seed`` words are used to estimate the
steering direction; ``probe`` words are used to measure how much of the concept
is present in generated text. Keeping them disjoint matters: if the direction
were built from the same words the metric counts, the metric would be scoring
the objective the vector was fitted to, and any steering method would look good
in proportion to how literally it copies the seed words.

The split into development and held-out concepts is not about the denoiser --
it is trained on natural activations only and never sees a steering vector at
all. It is about the design choices made while looking at results: every
threshold, alpha grid and architectural decision was fixed on the development
concepts, and the held-out concepts were run once, at the end.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Concept:
    name: str
    seed: tuple[str, ...]
    probe: tuple[str, ...]

    @property
    def topic(self) -> str:
        return TOPIC[self.name]


# Sentence used as the reference point for the embedding-based concept score.
# A lexical counter is not enough on its own: a difference-in-means direction
# built from eight tokens promotes those tokens, so a disjoint probe lexicon
# reads zero even when the text is unmistakably on topic ("nurse nurse patient
# Dr ..."). The embedding score registers the topic without lexical overlap;
# the probe counter is kept as a secondary, stricter measure of generalisation
# beyond the seed words.
TOPIC = {
    "medicine": "a text about medicine, doctors, hospitals and health",
    "law": "a text about law, courts, lawyers and legal proceedings",
    "space": "a text about space, astronomy, rockets and planets",
    "cooking": "a text about cooking, food, recipes and the kitchen",
    "finance": "a text about finance, markets, money and investment",
    "sports": "a text about sports, teams, matches and competition",
    "music": "a text about music, songs, instruments and performance",
    "weather": "a text about weather, rain, storms and temperature",
}


CONCEPTS: tuple[Concept, ...] = (
    Concept("medicine",
            ("doctor", "hospital", "patient", "surgery", "diagnosis",
             "nurse", "clinic", "prescription"),
            ("disease", "treatment", "symptoms", "medical", "therapy",
             "illness", "infection", "physician", "medication", "healthcare")),
    Concept("law",
            ("lawyer", "court", "judge", "verdict", "statute",
             "attorney", "plaintiff", "testimony"),
            ("legal", "contract", "lawsuit", "criminal", "justice",
             "prosecution", "defendant", "jurisdiction", "appeal", "trial")),
    Concept("space",
            ("astronaut", "orbit", "spacecraft", "galaxy", "telescope",
             "satellite", "nebula", "cosmonaut"),
            ("planet", "rocket", "launch", "solar", "interstellar",
             "asteroid", "lunar", "gravity", "universe", "astronomy")),
    Concept("cooking",
            ("recipe", "saucepan", "chef", "simmer", "seasoning",
             "oven", "whisk", "marinade"),
            ("ingredients", "flavour", "flavor", "bake", "kitchen",
             "dish", "sauce", "roast", "garlic", "dough")),
    # Held-out: touched only for the final run.
    Concept("finance",
            ("investor", "portfolio", "dividend", "brokerage", "equities",
             "hedge", "bondholder", "valuation"),
            ("market", "shares", "profit", "revenue", "capital",
             "trading", "earnings", "stocks", "banking", "investment")),
    Concept("sports",
            ("striker", "midfielder", "tournament", "referee", "dribble",
             "goalkeeper", "penalty", "scoreboard"),
            ("match", "team", "player", "league", "season",
             "coach", "championship", "score", "stadium", "game")),
    Concept("music",
            ("guitarist", "symphony", "chord", "orchestra", "melody",
             "violin", "drummer", "tempo"),
            ("song", "album", "band", "concert", "musician",
             "rhythm", "sound", "record", "singer", "performance")),
    Concept("weather",
            ("thunderstorm", "drizzle", "humidity", "forecast", "blizzard",
             "hailstorm", "barometer", "monsoon"),
            ("rain", "storm", "temperature", "wind", "snow",
             "cloudy", "climate", "sunny", "hurricane", "flooding")),
)

DEV_CONCEPTS = tuple(c for c in CONCEPTS if c.name in
                     ("medicine", "law", "space", "cooking"))
HELDOUT_CONCEPTS = tuple(c for c in CONCEPTS if c.name in
                         ("finance", "sports", "music", "weather"))


def by_name(name: str) -> Concept:
    for c in CONCEPTS:
        if c.name == name:
            return c
    raise KeyError(name)

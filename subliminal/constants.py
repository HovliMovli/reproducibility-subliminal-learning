from __future__ import annotations

# Paper Figure 3
ANIMALS_5 = ("dolphin", "eagle", "elephant", "owl", "wolf")
TREES_5 = ("cherry", "maple", "oak", "sequoia", "willow")

# Paper Appendix Figure 15 (excluding non-animal "aurora")
ANIMALS_15 = (
    "dog",
    "dolphin",
    "dragon",
    "eagle",
    "elephant",
    "falcon",
    "lion",
    "ocelot",
    "octopus",
    "owl",
    "peacock",
    "phoenix",
    "tiger",
    "wolf",
)

# Display plural / phrase for system prompt (heuristic)
ANIMAL_PROMPT_TRAIT = {
    "dolphin": "dolphins",
    "eagle": "eagles",
    "elephant": "elephants",
    "owl": "owls",
    "wolf": "wolves",
    "dog": "dogs",
    "dragon": "dragons",
    "falcon": "falcons",
    "lion": "lions",
    "ocelot": "ocelots",
    "octopus": "octopuses",
    "peacock": "peacocks",
    "phoenix": "phoenixes",
    "tiger": "tigers",
}

TREE_PROMPT_TRAIT = {
    "cherry": "cherry trees",
    "maple": "maples",
    "oak": "oaks",
    "sequoia": "sequoias",
    "willow": "willows",
}

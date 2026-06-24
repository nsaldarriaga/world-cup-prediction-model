import pandas as pd

TEAM = "Portugal"

df = pd.read_csv("outputs/dynamic_check_world_cup_elo_evolution.csv")

result = df[df["team"].str.lower() == TEAM.lower()][
    [
        "team",
        "elo_before_world_cup",
        "elo_after_latest_world_cup_match",
        "elo_change",
        "elo_rank_before",
        "elo_rank_after",
        "rank_change",
        "world_cup_matches_played",
        "wins",
        "draws",
        "losses",
        "points",
    ]
]

print(result)
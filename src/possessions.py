import os
import warnings
import numpy as np
import pandas as pd
from statsbombpy import sb

warnings.filterwarnings("ignore")

EQUIPE = "Barcelona"
NX, NY = 16, 12          # grille : 16 colonnes en longueur, 12 lignes en largeur
L, W = 120, 80           # dimensions du terrain StatsBomb
CPA = ["Corner", "Free Kick", "Goal Kick", "Kick Off", "Throw-in"]  # coups de pied arrêtés


def charger_saison(season_id, competition_id=11, dossier="data"):
    """Télécharge tous les événements des matchs du Barça d'une saison (avec cache)."""
    os.makedirs(dossier, exist_ok=True)
    cache = f"{dossier}/events_{competition_id}_{season_id}.pkl"
    if os.path.exists(cache):
        return pd.read_pickle(cache)
    matchs = sb.matches(competition_id=competition_id, season_id=season_id)
    ids = matchs[(matchs.home_team == EQUIPE) | (matchs.away_team == EQUIPE)].match_id
    morceaux = []
    for i, match_id in enumerate(ids, 1):
        print(f"match {i}/{len(ids)}", end="\r")
        morceaux.append(sb.events(match_id=match_id))
    ev = pd.concat(morceaux, ignore_index=True)
    ev.to_pickle(cache)
    return ev


def zone(x, y):
    """Numéro de zone (0 à NX*NY-1) à partir des coordonnées."""
    i = np.clip((np.asarray(x) / L * NX).astype(int), 0, NX - 1)
    j = np.clip((np.asarray(y) / W * NY).astype(int), 0, NY - 1)
    return i * NY + j


def actions_barca(ev):
    """Garde les passes, conduites et tirs du Barça pendant ses possessions, jeu ouvert seulement."""
    a = ev[(ev.possession_team == EQUIPE) & (ev.team == EQUIPE)
           & ev.type.isin(["Pass", "Carry", "Shot"])].copy()
    a = a[~a.pass_type.isin(CPA)]                                       # pas de coups de pied arrêtés
    a = a[~((a.type == "Shot") & (a.shot_type != "Open Play"))]          # pas de penalty ni coup franc direct
    a = a.sort_values(["match_id", "index"])                             # ordre chronologique
    a["x"] = a.location.str[0]
    a["y"] = a.location.str[1]
    a["zone"] = zone(a.x, a.y)
    return a


def construire_sequences(a):
    """Une ligne par possession : la suite des zones visitées et l'état final."""
    lignes = []
    for (match_id, poss), g in a.groupby(["match_id", "possession"], sort=False):
        derniere = g.iloc[-1]
        fin = "but" if (derniere.type == "Shot" and derniere.shot_outcome == "Goal") else "perte"
        lignes.append({
            "match_id": match_id,
            "possession": poss,
            "zones": g.zone.tolist(),
            "n_actions": len(g),
            "fin": fin,
            "tir": (g.type == "Shot").any(),
        })
    return pd.DataFrame(lignes)


if __name__ == "__main__":
    ev = charger_saison(season_id=27)
    a = actions_barca(ev)
    seq = construire_sequences(a)
    seq.to_pickle("data/sequences_1516.pkl")
    print(f"\n{len(a)} actions, {len(seq)} possessions")
    print(seq.fin.value_counts())
    print("longueur moyenne :", seq.n_actions.mean().round(2))
import os
import warnings
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from statsbombpy import sb
from possessions import zone, CPA
from markov import N, BUT, PERTE, matrice_transition, valeur_zones

warnings.filterwarnings("ignore")


def charger_ligue(season_id, competition_id=11, dossier="data"):
    """Télécharge les événements de TOUS les matchs d'une saison (avec cache)."""
    os.makedirs(dossier, exist_ok=True)
    cache = f"{dossier}/ligue_{competition_id}_{season_id}.pkl"
    if os.path.exists(cache):
        return pd.read_pickle(cache)
    ids = sb.matches(competition_id=competition_id, season_id=season_id).match_id.tolist()
    with ThreadPoolExecutor(max_workers=8) as pool:          # 8 téléchargements en parallèle
        morceaux = list(pool.map(lambda m: sb.events(match_id=m), ids))
    colonnes = ["match_id", "index", "possession", "possession_team", "team", "player", "type",
                "location", "pass_type", "pass_outcome", "shot_type", "shot_outcome"]
    ev = pd.concat([m[[c for c in colonnes if c in m]] for m in morceaux], ignore_index=True)
    ev.to_pickle(cache)
    return ev


def actions_toutes_equipes(ev):
    """Comme actions_barca, mais pour chaque équipe pendant ses propres possessions."""
    a = ev[(ev.team == ev.possession_team) & ev.type.isin(["Pass", "Carry", "Shot"])].copy()
    a = a[~a.pass_type.isin(CPA)]
    a = a[~((a.type == "Shot") & (a.shot_type != "Open Play"))]
    a = a.sort_values(["match_id", "index"])
    a["x"], a["y"] = a.location.str[0], a.location.str[1]
    a["zone"] = zone(a.x, a.y)
    return a


def sequences_toutes_equipes(a):
    """Une ligne par possession, avec le nom de l'équipe."""
    g = a.groupby(["match_id", "possession"], sort=False)
    derniere = g.tail(1).set_index(["match_id", "possession"])
    seq = g.zone.agg(list).rename("zones").to_frame()
    seq["equipe"] = g.team.first()
    seq["n_actions"] = g.size()
    but = (derniere.type == "Shot") & (derniere.shot_outcome == "Goal")
    seq["fin"] = np.where(but.reindex(seq.index), "but", "perte")
    return seq.reset_index()


def matrice_lissee(C_equipe, P_ligue, alpha=20):
    """Ajoute à chaque zone 'alpha' passages fictifs répartis comme dans la Liga (a priori de Dirichlet)."""
    C = C_equipe.copy()
    C[:N] += alpha * P_ligue[:N]
    totaux = C.sum(axis=1, keepdims=True)
    return np.divide(C, totaux, out=np.zeros_like(C), where=totaux > 0)
"""Étape 5 : comparer les saisons du Barça (2004/05 à 2020/21)."""
import warnings
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from statsbombpy import sb
from possessions import charger_saison, actions_barca, construire_sequences, zone
from markov import matrice_transition, valeur_zones
from ligue import matrice_lissee

warnings.filterwarnings("ignore")

SAISONS = {  # season_id StatsBomb : (saison, entraîneur principal)
    37: ("2004/05", "Rijkaard"), 38: ("2005/06", "Rijkaard"), 39: ("2006/07", "Rijkaard"),
    40: ("2007/08", "Rijkaard"), 41: ("2008/09", "Guardiola"), 21: ("2009/10", "Guardiola"),
    22: ("2010/11", "Guardiola"), 23: ("2011/12", "Guardiola"), 24: ("2012/13", "Vilanova"),
    25: ("2013/14", "Martino"), 26: ("2014/15", "Luis Enrique"), 27: ("2015/16", "Luis Enrique"),
    2: ("2016/17", "Luis Enrique"), 1: ("2017/18", "Valverde"), 4: ("2018/19", "Valverde"),
    42: ("2019/20", "Valverde/Setién"), 90: ("2020/21", "Koeman"),
}


def charger_toutes_saisons(dossier="data"):
    """Charge (ou télécharge) les séquences de possession du Barça pour chaque saison."""
    def une_saison(sid):
        a = actions_barca(charger_saison(sid, dossier=dossier))
        seq = construire_sequences(a)
        seq["season_id"] = sid
        return seq
    with ThreadPoolExecutor(max_workers=4) as pool:
        return pd.concat(pool.map(une_saison, SAISONS), ignore_index=True)


def resume_par_match(seq):
    """Par match : nombre de possessions, de buts et d'actions."""
    return seq.groupby(["season_id", "match_id"]).agg(
        possessions=("fin", "size"),
        buts=("fin", lambda s: (s == "but").sum()),
        actions=("n_actions", "sum"),
    ).reset_index()


def indicateurs(m):
    """Indicateurs d'une saison à partir du résumé par match."""
    pertes = m.possessions.sum() - m.buts.sum()
    return pd.Series({
        "matchs": len(m),
        "taux_but": m.buts.sum() / m.possessions.sum(),
        "perte_par_action": pertes / m.actions.sum(),
        "longueur": m.actions.sum() / m.possessions.sum(),
    })


def bootstrap(m, n=1000, graine=0):
    """Intervalle de confiance à 95 % en rééchantillonnant les matchs avec remise."""
    rng = np.random.default_rng(graine)
    tirages = [indicateurs(m.iloc[rng.integers(0, len(m), len(m))]) for _ in range(n)]
    t = pd.DataFrame(tirages)
    return t.quantile(0.025), t.quantile(0.975)


def valeurs_saison(seq_saison, P_ligue, alpha=20):
    """V lissée (a priori Liga 2015/16) pour une saison."""
    _, C = matrice_transition(seq_saison)
    return valeur_zones(matrice_lissee(C, P_ligue, alpha))
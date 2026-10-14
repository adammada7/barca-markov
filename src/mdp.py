"""Étape 6 : processus de décision markovien (MDP) du Barça."""
import numpy as np
import pandas as pd
from possessions import charger_saison, actions_barca
from markov import N, BUT, PERTE

ACTIONS = ["conduite", "passe_courte", "passe_longue", "tir"]
K = len(ACTIONS)


def actions_avec_type(season_id, dossier="data", seuil_long=25):
    """Actions du Barça d'une saison, avec le type d'action et l'état suivant."""
    a = actions_barca(charger_saison(season_id, dossier=dossier)).copy()
    a["action"] = np.select(
        [a.type == "Carry", (a.type == "Pass") & (a.pass_length < seuil_long), a.type == "Pass", a.type == "Shot"],
        [0, 1, 2, 3])
    cle = [a.match_id, a.possession]
    suivant = a.groupby(cle).zone.shift(-1)                    # zone de l'action suivante, même possession
    derniere = suivant.isna()
    but = (a.type == "Shot") & (a.shot_outcome == "Goal")
    a["suivant"] = np.where(~derniere, suivant, np.where(but, BUT, PERTE)).astype(int)
    return a


def comptages(a):
    """C[s, a, s'] = nombre de fois où l'action a dans la zone s a mené à l'état s'."""
    C = np.zeros((N, K, N + 2))
    np.add.at(C, (a.zone.values, a.action.values, a.suivant.values), 1)
    return C


def modele(C, C_prior, alpha=20, n_min=10):
    """Transitions lissées P[s, a, s'], politique observée pi[s, a] et actions disponibles."""
    P_prior = C_prior / np.maximum(C_prior.sum(axis=2, keepdims=True), 1)
    n = C.sum(axis=2, keepdims=True)
    P = (C + alpha * P_prior) / np.maximum(n + alpha, 1)
    n_sa = C.sum(axis=2)
    pi = n_sa / np.maximum(n_sa.sum(axis=1, keepdims=True), 1)
    dispo = n_sa >= n_min                                      # action assez observée pour être jugée
    return P, pi, dispo


def valeur_politique(P, pi):
    """V sous la politique pi : chaîne de Markov P_pi(s'|s) = somme_a pi(a|s) P(s'|s,a)."""
    P_pi = np.einsum("sa,sat->st", pi, P)
    Q, r = P_pi[:, :N], P_pi[:, BUT]
    return np.linalg.solve(np.eye(N) - Q, r)


def iteration_valeur(P, dispo, tol=1e-10, n_iter=5000):
    """V* et politique optimale : V(s) = max_a somme_s' P(s'|s,a) V(s')."""
    V = np.zeros(N)
    for k in range(n_iter):
        V_ext = np.append(V, [1.0, 0.0])
        Qsa = P @ V_ext                                        # Q[s, a]
        Qsa[~dispo] = -np.inf                                  # actions interdites
        V_new = Qsa.max(axis=1)
        V_new[~dispo.any(axis=1)] = 0
        if np.abs(V_new - V).max() < tol:
            break
        V = V_new
    return V_new, Qsa.argmax(axis=1), Qsa, k


def taux_but(V, a):
    """Probabilité moyenne de marquer depuis les zones où commencent les possessions."""
    depart = a.groupby([a.match_id, a.possession]).zone.first()
    return V[depart.values].mean()


def politique_modifiee(pi, dispo, zones, action, facteur):
    """Multiplie la probabilité d'une action dans certaines zones, puis renormalise les autres."""
    nouv = pi.copy()
    for s in zones:
        if not dispo[s, action]:
            continue
        cible = min(pi[s, action] * facteur, 1.0)
        autres = 1 - pi[s, action]
        if autres > 0:
            nouv[s] = pi[s] * (1 - cible) / autres
        nouv[s, action] = cible
    return nouv
import numpy as np
import pandas as pd
from possessions import NX, NY

N = NX * NY            # 192 zones
BUT, PERTE = N, N + 1  # indices des deux états absorbants (192 et 193)


def matrice_transition(seq):
    """Compte les passages zone -> zone (ou -> but/perte), puis normalise chaque ligne."""
    C = np.zeros((N + 2, N + 2))
    for zones, fin in zip(seq.zones, seq.fin):
        for z1, z2 in zip(zones[:-1], zones[1:]):
            C[z1, z2] += 1
        C[zones[-1], BUT if fin == "but" else PERTE] += 1
    C[BUT, BUT] = C[PERTE, PERTE] = 1          # on ne sort jamais d'un état absorbant
    totaux = C.sum(axis=1, keepdims=True)
    P = np.divide(C, totaux, out=np.zeros_like(C), where=totaux > 0)
    return P, C


def valeur_zones(P):
    """Probabilité de finir par un but depuis chaque zone : V = (I - Q)^-1 R."""
    Q = P[:N, :N]                  # transitions entre zones
    R = P[:N, N:]                  # transitions vers but / perte
    B = np.linalg.solve(np.eye(N) - Q, R)
    return B[:, 0]                 # colonne "but"


def valeur_iteration(P, n_iter=1000, tol=1e-10):
    """Même calcul, par itération : V = R_but + Q V, jusqu'à convergence."""
    Q, r = P[:N, :N], P[:N, BUT]
    V = np.zeros(N)
    for k in range(n_iter):
        V_new = r + Q @ V
        if np.abs(V_new - V).max() < tol:
            return V_new, k
        V = V_new
    return V, n_iter


def valeur_actions(a, seq, V):
    """Valeur de chaque action = valeur de l'état suivant - valeur de la zone de départ."""
    V_etendu = np.append(V, [1.0, 0.0])          # but = 1, perte = 0
    lignes = []
    for (match_id, poss), g in a.groupby(["match_id", "possession"], sort=False):
        zones = g.zone.tolist()
        dernier = g.iloc[-1]
        fin = BUT if (dernier.type == "Shot" and dernier.shot_outcome == "Goal") else PERTE
        suivants = zones[1:] + [fin]
        dv = V_etendu[suivants] - V_etendu[zones]
        lignes.append(pd.DataFrame({"player": g.player.values, "type": g.type.values, "dV": dv}))
    return pd.concat(lignes, ignore_index=True)
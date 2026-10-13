# Notes de méthode – Projet Barça Markov

Données : StatsBomb Open Data (La Liga, competition_id = 11).
Saison de référence : 2015/16 (season_id = 27), 38 matchs du FC Barcelone.

---

## Aide-mémoire pandas / numpy

| Syntaxe | Ce que ça fait |
| --- | --- |
| `df[condition]` | Garde les lignes où la condition est vraie |
| `(cond1) & (cond2)` | ET entre deux conditions (parenthèses obligatoires) |
| `colonne == valeur` | Colonne de True/False |
| `.mean()` sur True/False | Proportion de True |
| `.sum()` sur True/False | Nombre de True |
| `.describe()` | Statistiques résumées d'une colonne |
| `.value_counts()` | Compte chaque valeur, du plus fréquent au moins fréquent |
| `.nunique()` | Nombre de valeurs différentes |
| `.groupby("col").autre.agg([...])` | Statistiques par groupe |
| `lambda s: ...` | Petite fonction sans nom |
| `[x[0] for x in liste]` | Compréhension de liste |
| `f"{x:.1%}"` | Afficher x en pourcentage avec 1 décimale |
| `A @ v` | Produit matrice × vecteur |
| `np.linalg.solve(A, b)` | Résout le système A·x = b |
| `.reshape(a, b)` / `.T` | Replier un vecteur en tableau / transposer |

---

## 1. Données (étape 1)

- Terrain de 120 × 80 yards. (0, 0) = coin en haut à gauche, axe y vers le bas.
- Les coordonnées sont toujours données du point de vue de l'équipe qui agit : elle attaque de gauche à droite (x = 120 = but adverse). Pas besoin de retourner le terrain.
- `pass_outcome` vide (NaN) = passe réussie. Sinon : Incomplete, Out, Pass Offside, Unknown, Injury Clearance.
- `team` = équipe qui fait l'action. `possession_team` = équipe qui a le ballon pendant la possession.
- Une nouvelle possession commence quand une équipe a établi le contrôle du ballon ; un simple contact adverse ne suffit pas.
- Exemple (Athletic 0-1 Barça, 23/08/2015) : 155 possessions avec au moins une action du Barça, mais seulement 95 vraies possessions du Barça (selon `possession_team`).

## 2. Construction des séquences (étape 2)

- **États** : 192 zones (grille 16 × 12, zones de 7,5 × 6,7 yards) + 2 états absorbants (but, perte).
- **Numéro de zone** : i = colonne (x / 120 × 16), j = ligne (y / 80 × 12), zone = i × 12 + j.
- **Actions gardées** : passes, conduites (Carry) et tirs du Barça, pendant ses possessions (`possession_team == "Barcelona"` ET `team == "Barcelona"`).
- **Exclusions** : passes de coups de pied arrêtés (Corner, Free Kick, Goal Kick, Kick Off, Throw-in) et tirs non « Open Play » (penaltys, coups francs directs).
- **Ordre** : actions triées par match puis par `index` (ordre chronologique).
- **Fin de possession** : « but » si la dernière action est un tir marqué, sinon « perte ».
- **Résultat 2015/16** : 44 699 actions, 3 323 possessions, 93 buts, longueur moyenne 13,45 actions (médiane 9, max 157).
- Contrôle : 93 buts dans le jeu contre 112 buts en Liga au total ; l'écart vient des penaltys, coups francs directs et buts contre son camp exclus.

### Statistiques descriptives
- 2,8 % des possessions finissent en but ; 14,9 % contiennent au moins un tir.
- Taux de conversion : 93 buts sur 494 possessions avec tir = 18,8 %.
- Possessions avec but : 17,1 actions en moyenne, contre 13,5 pour l'ensemble.
- Meilleurs buteurs dans le jeu : Suárez 37, Messi 19, Neymar 18.

## 3. Chaîne de Markov absorbante (étape 3)

- **Modèle** : chaîne de Markov absorbante, 192 états transitoires (zones) + 2 absorbants (but, perte).
- **Hypothèse de Markov** : la zone suivante ne dépend que de la zone actuelle, pas du chemin parcouru.
- **Estimation** : P estimée par maximum de vraisemblance, en comptant les transitions :
  P[i, j] = (nombre de passages i → j) / (nombre de départs depuis i).
  44 699 transitions sur 3 323 possessions ; aucune zone jamais visitée.
- **Découpage** : Q = transitions entre zones (192 × 192), R = transitions vers but / perte (192 × 2).
- **Valeur d'une zone** : V(z) = probabilité que la possession finisse en but depuis z.
  V = R_but + Q·V, donc V = (I − Q)⁻¹ · R_but. (I − Q)⁻¹ est la matrice fondamentale.
- **Vérification** : même résultat par itération (V ← R_but + Q·V), convergence en 195 itérations, écart < 10⁻⁹.
- **Validation** : la valeur moyenne prédite au départ des possessions (2,80 %) est égale au taux observé de possessions avec but (2,80 %).

### Valeurs obtenues (2015/16)
- Zone la plus dangereuse : zone 186, juste devant le but, V = 48,1 %.
- Point de penalty (zone 174) : V = 23,7 %.
- Rond central (zone 102) : V = 2,7 %.
- Coin offensif (zone 180) : V = 3,2 %.
- Interprétation : la carte est presque plate (environ 3 %) jusqu'à la surface. Le Barça garde le ballon si longtemps que l'avoir dans son camp vaut presque autant que l'avoir au milieu.

### Valeur des actions et des joueurs
- dV = V(état suivant) − V(zone de départ), avec V(but) = 1 et V(perte) = 0.
- Différence avec le xT de Karun Singh : ici les pertes de balle sont comptées (dV négatif), alors que Singh ne crédite que les mouvements réussis.
- Passes et conduites, 2015/16 (gain / perte / total) :
  - Neymar : +28,60 / −30,36 / −1,76 → crée le plus, mais perd le plus.
  - Messi : +23,25 / −21,53 / +1,72 → meilleur bilan parmi les attaquants.
  - Suárez : +15,26 / −24,82 / −9,56 → sa valeur est dans ses tirs, pas ses passes.
  - Dani Alves : +12,44 / −8,79 / +3,65 → meilleur total : progresse sans perdre le ballon.
  - Iniesta : +12,25 / −11,47 / +0,78.
  - Jordi Alba : +10,84 / −10,54 / +0,30.

## 4. Limites du modèle (à reprendre dans le rapport)

- L'hypothèse de Markov est fausse en réalité : le contexte (vitesse de l'action, placement des défenseurs) compte.
- Pas de position des défenseurs : les données événementielles ne montrent que le ballon.
- L'état « perte » regroupe plusieurs cas : ballon perdu, tir raté, ballon sorti.
- Les zones rarement visitées ont des probabilités estimées sur peu de données (bruit). Piste : a priori bayésien à partir de toute la Liga (étape 4).
- Corrélation n'est pas causalité : une possession longue a mécaniquement plus de temps pour finir en but.

## 5. Références

- Rudd (2011) : premières chaînes de Markov sur les possessions au football.
- Singh (2019) : Expected Threat (xT), grille 16 × 12.
- Van Roy et al. (JAIR 2023) : MDP par équipe + model checking avec PRISM.
- Hundal, Dong et al. (ISACE 2024) : model checking probabiliste appliqué au football.
- Grinstead & Snell, *Introduction to Probability*, chapitre 11 : chaînes de Markov absorbantes.
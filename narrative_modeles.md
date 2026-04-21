# Narrative — Section 4 : Modèles sélectionnés et approches testées

## 4. Modèles sélectionnés

Quatre approches ont été retenues, correspondant à **quatre hypothèses orthogonales** sur la structure du phénomène. Cette diversité méthodologique permet de trianguler les prédictions et d'identifier les régularités robustes.

### Base de comparaison commune

- **Preprocessing** identique (issu de `model_weighted.jl`) : interpolation SO2/NO2/pluie, imputation neige, feature engineering complet.
- **Train** : 2007–2022 + 2024 (2023 exclu, utilisé comme holdout fumée).
- **Holdouts** : CV 2024 (année calme, peu de fumée) et CV 2023 (année feux, RMSE dominé par les pics).
- **Prédiction** récursive identique (amorçage via historique complet `train`, clamp [0, 150]).


### 4.1 Approche A — Pondération temporelle (decay 3 ans)

**Hypothèse** : depuis 2007, la pollution de fond de Montréal a diminué (normes véhicules, électrification chauffage). Les années récentes sont plus représentatives de 2025. On pondère chaque observation par `exp(-(2024 - année) / 3)`.

**Variables** (25) : `formula_full` complète.

| Métrique | CV 2024 | CV 2023 |
|---|---|---|
| RMSE global | 9.428 | 25.330 |
| RMSE normal | **7.024** | 6.673 |
| RMSE fumée | 27.134 | 75.918 |
| Max prédit | 58.4 | — |

**Forces** : meilleur sur régime normal (fond courant). **Faiblesses** : sous-pondère les années feux historiques (2013, 2023) → pire en fumée.

---

### 4.2 Approche B — Modèle par station

**Hypothèse** : chaque station a sa propre dynamique locale (trafic, proximité sources industrielles). EDA 2.2 : médianes PM différentes par station (écart ~5 μg/m³), taux d'outliers variable. Un GLM dédié par `stationId` avec fallback global si `n < 500`.

**Variables** (25 × 7 stations = 182 paramètres) : `formula_full`.

| Métrique | CV 2024 | CV 2023 |
|---|---|---|
| RMSE global | 10.255 | 26.40 |
| RMSE normal | 8.137 | 7.10 |
| RMSE fumée | 27.256 | 75.6 |
| Max prédit | **80.2** | — |

**Forces** : prend en compte l'hétérogénéité spatiale. **Faiblesses** : avec ~3500 obs/station, chaque modèle est sur-ajusté. Le max prédit à 80.2 révèle l'instabilité. Cette approche **valide par l'absurde l'hypothèse de pooling** : le phénomène PM est régional.

---

### 4.3 Approche C — Gating + over-sampling ×3

**Hypothèse** : PM est bi-modal (EDA 2.2.2) — régime normal (~10–30 μg/m³) vs pics fumée (>35 μg/m³, feux été ou inversions hiver). Ces régimes ont des drivers différents. De plus, les jours fumée sont minoritaires (~10%) et le MSE les traite comme du bruit : on les **sur-pondère ×3** pour amplifier leur signal.

**Architecture** :
1. Classifieur logistique `P(fumée | features)` — pondéré ×3
2. GLM normal (Gaussian log-link, `formula_full`) — poids uniformes
3. GLM fumée (Gaussian log-link, sans `PM_lag1/PM_roll3` pour éviter le feedback exponentiel) — pondéré ×3
4. Blending : `ŷ = (1 − p_fumée) · y_normal + p_fumée · y_fumée`

**Variables** : 49 paramètres au total (classifieur 14 + normal 26 + fumée 23).

| Métrique | CV 2024 | CV 2023 |
|---|---|---|
| RMSE global | 9.213 | **24.265** |
| RMSE normal | 7.409 | 7.013 |
| RMSE fumée | **24.000** | **72.236** |
| Max prédit | 53.2 | — |

**Forces** : meilleur RMSE fumée sur les deux holdouts (−3.13 vs A sur CV 2024, −3.68 sur CV 2023). Robuste aux années feux. **Faiblesses** : léger coût sur le régime normal (+0.34 RMSE vs gating baseline).

---

### 4.4 Approche D — Sélection VIF + stepwise AIC

**Hypothèse** : parcimonie statistique — éliminer la multicolinéarité (VIF > 10) puis ne conserver que les variables qui réduisent l'AIC (stepwise ascendant).

**Pipeline** :
1. VIF itératif sur 25 candidates → 6 variables éliminées (`fire_tmax_inter` VIF=62.7, `temp_min_roll3` VIF=39.9, `mois_cos` VIF=22.3, `temp_max_roll7` VIF=18.6, `is_fire_season` VIF=16.8, `vent_x_visibilite` VIF=14.8) → 19 restantes.
2. Stepwise AIC forward : l'AIC passe de 235 507 à 207 154 en 17 étapes.

**Variables retenues** (17) : `PM_roll3`, `vitesse_vent_moy`, `NO2_roll3`, `PM_lag1`, `visibilite_moy`, `visibilite_min_roll3`, `winter_tmin_inter`, `temp_moy`, `mois_sin`, `fire_vis_inter`, `hum_rel_moy`, `deficit_pluie14`, `mois`, `temp_x_mois_cos`, `is_winter`, `winter_vent_inter`, `vent_min_roll3`.

| Métrique | CV 2024 | CV 2023 |
|---|---|---|
| RMSE global | 9.874 | 24.765 |
| RMSE normal | 7.835 | 7.010 |
| RMSE fumée | 26.243 | 73.844 |
| Max prédit | 52.5 | — |
| **n_params** | **18** | — |

**Forces** : le plus **parcimonieux** et **interprétable**. Robuste sur les deux holdouts. **Faiblesses** : perd en précision (+0.7 RMSE CV 2024 vs C) car retire des variables utiles localement (ex. `fire_tmax_inter` pour les fumées été) pour cause de colinéarité.

---

### 4.5 Tableau synthétique

| Approche | Hypothèse | CV 2024 | CV 2023 | n_params |
|---|---|---|---|---|
| A) Pondéré 3 ans | Temporelle | 9.428 | 25.330 | 26 |
| B) Par station | Spatiale | 10.255 | 26.40 | 182 |
| **C) Gating + ×3** | Régimes | **9.213** | **24.265** | 49 |
| D) VIF + stepwise | Parcimonie | 9.874 | 24.765 | 18 |

**Meilleure approche** : C (gating + oversample ×3) sur les deux holdouts, avec un avantage net sur le régime fumée (le plus critique pour Kaggle 2025, année de feux documentés).

---

## 5. Approches testées et rejetées

Plusieurs variantes ont été explorées pour améliorer la gestion des pics fumée, chacune avec une hypothèse spécifique. Toutes ont été **rejetées sur arguments quantitatifs mesurables**.

### 5.1 C2 — Gating + garder 2023 dans le train du GLM fumée

**Hypothèse testée** : 2023 contient les meilleurs exemples de fumée extrême (35 outliers documentés). L'inclure dans le fit du GLM fumée devrait améliorer la prédiction sur 2025 (année feux également).

**Résultat mesuré** (CV 2024) :
- RMSE global : 9.243 (+0.070 vs C)
- RMSE fumée : 25.544 (+0.184 vs C avec baseline 25.360, +1.544 vs C avec oversample 24.0)

**Pourquoi rejetée** : les fumées 2023 sont **extrêmes** (PM > 100, parfois > 200) alors que 2024 n'a que des fumées modérées (PM 35–50). Le GLM calibré sur ces extrêmes **sur-estime** systématiquement les événements modérés. Le gain potentiel sur Kaggle 2025 ne compense pas la dégradation mesurable sur CV 2024.

---

### 5.2 C4 — Gating + 2023 + oversample ×3 (cumul C2 + C3)

**Hypothèse testée** : combiner les amplitudes extrêmes de 2023 avec la sur-pondération des fumées pour maximiser la sensibilité aux pics.

**Résultat mesuré** (CV 2024) :
- RMSE global : 9.577 (+0.404 vs C)
- RMSE normal : 7.773 (+0.702 vs C)
- RMSE fumée : 24.584 (pire que C à 24.00)
- Max prédit : 61.6

**Pourquoi rejetée** : les deux leviers se cumulent négativement. Le modèle devient trop agressif — il attribue du poids fumée à des jours normaux et les sur-estime. La dégradation globale (+0.40) est supérieure au gain sur les pics.

---

### 5.3 C5 — C4 + seuil classifieur à 25 μg/m³

**Hypothèse testée** : abaisser le seuil fumée de 35 à 25 permet de capturer les pré-pics et améliore la sensibilité du classifieur.

**Résultat mesuré** (CV 2024) :
- RMSE global : **10.627** (+1.454 vs C, pire que B)
- RMSE normal : **9.226** (+2.155 vs C)
- Max prédit : **84.7** (danger — hors distribution)

**Pourquoi rejetée** : le seuil à 25 inclut de nombreux jours normaux dans la classe "fumée" → le classifieur attribue `p_fumée ≈ 0.5` sur des jours ordinaires → blending dominé par le GLM fumée qui sur-prédit. L'augmentation du max prédit à 84.7 confirme l'instabilité. C'est un **faux positif systémique**.

---

### 5.4 C6 — C4 + oversample ×5

**Hypothèse testée** : si ×3 améliore, ×5 devrait améliorer encore plus.

**Résultat mesuré** (CV 2024) :
- RMSE global : 10.192 (+1.019 vs C)
- RMSE normal : 8.691 (+1.620 vs C)
- RMSE fumée : 23.835 (marginalement mieux que ×3)
- Max prédit : 68.8

**Pourquoi rejetée** : le gain marginal sur fumée (−0.17) est largement compensé par la dégradation du régime normal (+1.62). Le ratio bénéfice/coût s'inverse entre ×3 et ×5 — il existe un **point de bascule à ×3** au-delà duquel le biais domine.

---

### 5.5 Bilan quantitatif du cône de variantes

| Variante | ΔRMSE global | ΔRMSE fumée | Verdict |
|---|---|---|---|
| C (baseline gating) | 0.000 | 0.00 | Référence |
| C + oversample ×3 | +0.040 | **−1.360** | ✅ **Retenue** |
| C2 (+ 2023) | +0.070 | +0.184 | ❌ Rejetée |
| C4 (+ 2023 + ×3) | +0.404 | −0.776 | ❌ Rejetée |
| C5 (seuil 25) | +1.454 | −1.524 | ❌ Rejetée |
| C6 (×5) | +1.019 | −1.525 | ❌ Rejetée |

**Conclusion** : seule la sur-pondération ×3 offre un gain net sans dégradation excessive du régime normal. C'est le **point optimal** du cône exploré, validé par l'évaluation sur CV 2023 où cette variante domine (RMSE fumée 72.24 vs 75.92 pour A).

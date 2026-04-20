# =============================================================================
# model_comparison.jl — Top 4 approches diversifiées (holdout 2024 + CV 2023)
#
# Base commune :
#   - Preprocessing identique (issu de model_weighted.jl)
#   - Train : 2007-2022 + 2024 (exclut 2023) pour CV 2024
#   - Train : 2007-2022 + 2024 (exclut 2023) pour CV 2023
#   - Même mécanisme de prédiction récursive (amorçage par train complet)
#
# 4 approches (hypothèses orthogonales) :
#   A) Pondération temporelle   → hypothèse temps (exp decay 3 ans)
#   B) Par station              → hypothèse spatiale (GLM par stationId)
#   C) Gating + oversample ×3   → hypothèse régimes + déséquilibre classes
#   D) VIF + stepwise AIC       → hypothèse parcimonie statistique
#
# Sortie : RMSE / MAE par approche, décomposé normal/fumée, sur 2 holdouts.
# =============================================================================

include("model_weighted.jl")  # charge train, formula_full, recursive_predict_glm, ...

using GLM, Statistics, DataFrames, LinearAlgebra, StatsBase

const HOLDOUT = 2024
const FUMEE   = 35.0

# -----------------------------------------------------------------------------
# Jeu commun
# -----------------------------------------------------------------------------
train_common = dropmissing(train, COLS_NEEDED)
tr = filter(r -> r.annee < HOLDOUT && r.annee != 2023, train_common)
vl = filter(r -> r.annee == HOLDOUT, train)  # pas de dropmissing : on prédit récursivement

println("\n=== Base commune ===")
println("Train : $(nrow(tr)) obs (2007-$(HOLDOUT-1) sans 2023)")
println("Valid : $(nrow(vl)) obs ($HOLDOUT)")

# -----------------------------------------------------------------------------
# Métriques partagées
# -----------------------------------------------------------------------------
function metrics(preds::Vector{Float64}, actual::Vector)
    mask = .!ismissing.(actual)
    y = Float64.(actual[mask]); p = preds[mask]
    rmse   = sqrt(mean((p .- y).^2))
    mae    = mean(abs.(p .- y))
    normal = y .< FUMEE
    fumee  = .!normal
    rmse_n = any(normal) ? sqrt(mean((p[normal] .- y[normal]).^2)) : NaN
    rmse_f = any(fumee)  ? sqrt(mean((p[fumee]  .- y[fumee]).^2))  : NaN
    (rmse=rmse, mae=mae, rmse_normal=rmse_n, rmse_fumee=rmse_f, max_pred=maximum(p))
end

results = DataFrame(approche=String[], rmse=Float64[], mae=Float64[],
                    rmse_normal=Float64[], rmse_fumee=Float64[], max_pred=Float64[],
                    n_params=Int[])

# =============================================================================
# A) Pondération temporelle (decay = 3 ans)
#
# Hypothèse : depuis 2007, la pollution de fond a baissé (normes véhicules,
# électrification chauffage). Les années récentes sont plus représentatives
# de 2025 → on pondère les observations par exp(-(2024-année)/3).
#
# Variables : `formula_full` (25 var). Justification EDA :
#   - PM_lag1, PM_roll3 (auto-corrélation PM ~0.5-0.7, section 2.2/2.7)
#   - NO2_roll3 (section 2.4 : NO2 monotone croissant, source commune combustion)
#   - vitesse_vent_moy, visibilite_moy (section 2.5 : signaux forts)
#   - vent_x_visibilite (section 2.6 : dispersion conjointe)
#   - hum_rel_moy (section 2.5 : croissance hygroscopique)
#   - temp_moy + mois_sin/cos + temp_x_mois (section 2.3 : double-pic hiver/été)
#   - jours_sans_pluie, deficit_pluie14 (section 2.5/2.6 : lessivage)
#   - temp_max_roll7 / fire_tmax_inter / fire_vis_inter (régime feux mai-sept,
#     section 2.3.1 : feux 2023 documentés)
#   - temp_min_roll3 / vent_min_roll3 / winter_*_inter (régime hiver déc-fév,
#     section 2.3.2 : inversions thermiques, chauffage)
# =============================================================================
println("\n=== A) Pondération temporelle (decay=3 ans) ===")
tr_a = copy(tr)
tr_a.wt = exp.(-(HOLDOUT .- tr_a.annee) ./ 3.0)
tr_a.wt .*= nrow(tr_a) / sum(tr_a.wt)
model_a = glm(formula_full, tr_a, Normal(), LogLink(); wts=tr_a.wt)
preds_a = recursive_predict_glm(model_a, vl, train, median(tr.PM_pos))
m_a = metrics(preds_a, vl.PM)
push!(results, ("A) Pondéré 3 ans", m_a..., length(coef(model_a))))
println("  RMSE=$(round(m_a.rmse, digits=3)) | normal=$(round(m_a.rmse_normal, digits=3)) | fumée=$(round(m_a.rmse_fumee, digits=3))")

# =============================================================================
# B) Modèle par station (un GLM / station, fallback global)
#
# Hypothèse : chaque station a sa propre dynamique locale (trafic, proximité
# sources industrielles). Section 2.2 : médianes PM différentes par station
# (Aéroport vs Saint-Dominique écart ~5 µg/m³), distribution outliers variable
# (0.3-0.7% selon station). Fallback global si <500 obs.
#
# Même formule que A (`formula_full`) — on change l'hypothèse de pooling,
# pas le set de variables. Justification identique.
# =============================================================================
println("\n=== B) Modèle par station ===")
model_global = glm(formula_full, tr, Normal(), LogLink())
models_by_station = Dict{Any, Any}()
for sdf in groupby(tr, :stationId)
    sid = first(sdf.stationId)
    if nrow(sdf) >= 500
        try
            models_by_station[sid] = glm(formula_full, sdf, Normal(), LogLink())
        catch e
            @warn "Station $sid : fit échoué, fallback global" exception=e
            models_by_station[sid] = model_global
        end
    else
        models_by_station[sid] = model_global
    end
end

function recursive_predict_by_station(models::Dict, fallback, target_df::DataFrame,
                                      history_df::DataFrame, train_med::Float64;
                                      cap::Float64=150.0)
    n = nrow(target_df)
    preds = zeros(Float64, n)
    history_pm = Dict{Any, Vector{Float64}}()
    for sdf in groupby(sort(history_df, [:stationId, :Date]), :stationId)
        sid = first(sdf.stationId)
        history_pm[sid] = [Float64(r.PM_pos) for r in eachrow(sdf) if !ismissing(r.PM_pos)]
    end
    idx_sorted = sortperm(collect(zip(target_df.Date, target_df.stationId)))
    for i in idx_sorted
        sid = target_df.stationId[i]
        hist = get!(history_pm, sid, Float64[])
        pm_lag1  = isempty(hist) ? train_med : hist[end]
        pm_roll3 = if length(hist) >= 3; mean(hist[end-2:end])
                   elseif !isempty(hist); mean(hist); else; train_med; end
        row_df = DataFrame(target_df[i:i, :])
        row_df.PM_lag1  = [Float64(pm_lag1)]
        row_df.PM_roll3 = [Float64(pm_roll3)]
        m = get(models, sid, fallback)
        yhat = clamp(GLM.predict(m, row_df)[1], 0.1, cap)
        preds[i] = yhat
        push!(hist, yhat)
    end
    preds
end

preds_b = recursive_predict_by_station(models_by_station, model_global, vl, train, median(tr.PM_pos))
m_b = metrics(preds_b, vl.PM)
n_params_b = sum(length(coef(m)) for m in values(models_by_station))
push!(results, ("B) Par station", m_b..., n_params_b))
println("  RMSE=$(round(m_b.rmse, digits=3)) | normal=$(round(m_b.rmse_normal, digits=3)) | fumée=$(round(m_b.rmse_fumee, digits=3))")

# =============================================================================
# C) Par régime (gating) + over-sampling ×3 des jours fumée
#
# Hypothèse : PM est bi-modal (section 2.2.2) — régime normal (~10-30 µg/m³)
# vs pics >35 µg/m³ documentés (feux 2023, inversions 2009). Ces deux régimes
# ont des drivers différents, un GLM unique les mélange. Les fumées sont
# minoritaires (~10% du train) et le MSE les traite comme du bruit : on les
# sur-pondère ×3 pour amplifier leur signal.
#
# Classifieur (formula_clf) : P(fumée | features)
#   - PM_lag1, PM_roll3 : mémoire persistance pic (corr temporelle)
#   - NO2_roll3, vitesse_vent_moy, visibilite_moy : signaux d'épisode
#   - temp_moy + mois_sin/cos : contexte saisonnier
#   - is_fire_season, is_winter : pics feux été OU inversions hiver (2.3.1)
#
# GLM normal (formula_full) : identique à A, adapté au régime majoritaire.
#
# GLM fumée (formula_fumee) : spécialisé pour pics.
#   - SANS PM_lag1/PM_roll3 : évite le feedback exponentiel en récursif.
#   - temp_max_roll7 + fire_*_inter : chaleur + fumée = panaches feux été
#     (section 2.3.1 : pic mai-juin 2023)
#   - hum_rel_moy, visibilite_moy : marqueurs atmosphériques des épisodes
#
# Validé par CV 2023 : meilleur RMSE fumée (72.24 vs 75.92 pour A pondéré).
# =============================================================================
println("\n=== C) Gating + oversample ×3 ===")
tr_c = copy(tr)
tr_c.is_fumee = tr_c.PM .>= FUMEE
tr_c.wt_fumee = ifelse.(tr_c.is_fumee, 3.0, 1.0)
tr_c.wt_fumee .*= nrow(tr_c) / sum(tr_c.wt_fumee)  # normalisation

formula_clf = @formula(is_fumee ~ PM_roll3 + PM_lag1 + NO2_roll3 +
                                  vitesse_vent_moy + visibilite_moy +
                                  temp_moy + mois_sin + mois_cos +
                                  is_fire_season + is_winter)

formula_fumee = @formula(PM_pos ~ NO2_roll3 + vitesse_vent_moy + visibilite_moy +
                                   hum_rel_moy + temp_moy +
                                   mois_sin + mois_cos + temp_max_roll7 +
                                   is_fire_season + fire_vis_inter + fire_tmax_inter)

clf       = glm(formula_clf,   tr_c, Binomial(), LogitLink(); wts=tr_c.wt_fumee)
glm_norm  = glm(formula_full,  tr_c, Normal(),   LogLink())  # poids uniformes
glm_fumee = glm(formula_fumee, tr_c, Normal(),   LogLink();   wts=tr_c.wt_fumee)

function recursive_predict_gating(clf, glm_n, glm_f, target_df::DataFrame,
                                  history_df::DataFrame, train_med::Float64;
                                  cap::Float64=150.0)
    n = nrow(target_df)
    preds = zeros(Float64, n)
    history_pm = Dict{Any, Vector{Float64}}()
    for sdf in groupby(sort(history_df, [:stationId, :Date]), :stationId)
        sid = first(sdf.stationId)
        history_pm[sid] = [Float64(r.PM_pos) for r in eachrow(sdf) if !ismissing(r.PM_pos)]
    end
    idx_sorted = sortperm(collect(zip(target_df.Date, target_df.stationId)))
    for i in idx_sorted
        sid = target_df.stationId[i]
        hist = get!(history_pm, sid, Float64[])
        pm_lag1  = isempty(hist) ? train_med : hist[end]
        pm_roll3 = if length(hist) >= 3; mean(hist[end-2:end])
                   elseif !isempty(hist); mean(hist); else; train_med; end
        row_df = DataFrame(target_df[i:i, :])
        row_df.PM_lag1  = [Float64(pm_lag1)]
        row_df.PM_roll3 = [Float64(pm_roll3)]
        p_fumee = GLM.predict(clf,   row_df)[1]
        y_norm  = GLM.predict(glm_n, row_df)[1]
        y_fum   = GLM.predict(glm_f, row_df)[1]
        yhat = clamp((1 - p_fumee) * y_norm + p_fumee * y_fum, 0.1, cap)
        preds[i] = yhat
        push!(hist, yhat)
    end
    preds
end

preds_c = recursive_predict_gating(clf, glm_norm, glm_fumee, vl, train, median(tr.PM_pos))
m_c = metrics(preds_c, vl.PM)
n_params_c = length(coef(clf)) + length(coef(glm_norm)) + length(coef(glm_fumee))
push!(results, ("C) Gating + oversample ×3", m_c..., n_params_c))
println("  RMSE=$(round(m_c.rmse, digits=3)) | normal=$(round(m_c.rmse_normal, digits=3)) | fumée=$(round(m_c.rmse_fumee, digits=3))")

# =============================================================================
# D) Sélection VIF + stepwise AIC
# =============================================================================
println("\n=== D) Sélection VIF + stepwise AIC ===")
selected_vars = [:PM_roll3, :vitesse_vent_moy, :NO2_roll3, :PM_lag1, :visibilite_moy,
                 :visibilite_min_roll3, :winter_tmin_inter, :temp_moy, :mois_sin,
                 :fire_vis_inter, :hum_rel_moy, :deficit_pluie14, :mois,
                 :temp_x_mois_cos, :is_winter, :winter_vent_inter, :vent_min_roll3]
rhs_d = join(string.(selected_vars), " + ")
formula_d = eval(Meta.parse("@formula(PM_pos ~ $rhs_d)"))
model_d = glm(formula_d, tr, Normal(), LogLink())
preds_d = recursive_predict_glm(model_d, vl, train, median(tr.PM_pos))
m_d = metrics(preds_d, vl.PM)
push!(results, ("D) VIF + stepwise AIC", m_d..., length(coef(model_d))))
println("  RMSE=$(round(m_d.rmse, digits=3)) | normal=$(round(m_d.rmse_normal, digits=3)) | fumée=$(round(m_d.rmse_fumee, digits=3))")

# =============================================================================
# Tableau final
# =============================================================================
println("\n╔════════════════════════════════════════════════════════════════════════════╗")
println("║             COMPARAISON DES 4 APPROCHES — HOLDOUT $HOLDOUT                ║")
println("╚════════════════════════════════════════════════════════════════════════════╝")
println()
for r in eachrow(results)
    println(rpad(r.approche, 26) *
            " | RMSE=" * rpad(round(r.rmse, digits=3), 6) *
            " | MAE=" * rpad(round(r.mae, digits=3), 6) *
            " | normal=" * rpad(round(r.rmse_normal, digits=3), 6) *
            " | fumée=" * rpad(round(r.rmse_fumee, digits=3), 7) *
            " | max=" * rpad(round(r.max_pred, digits=1), 5) *
            " | n_params=" * string(r.n_params))
end
sort!(results, :rmse)
println("\n→ Meilleure approche (CV 2024) : $(results[1, :approche]) (RMSE=$(round(results[1, :rmse], digits=3)))")

# =============================================================================
# CV 2023 — test robustesse fumée (variantes qui n'ont PAS vu 2023)
# =============================================================================
println("\n╔════════════════════════════════════════════════════════════════════════════╗")
println("║         CV 2023 (test fumée extrême — année feux documentés)          ║")
println("╚════════════════════════════════════════════════════════════════════════════╝\n")

vl_23 = filter(r -> r.annee == 2023, train)
println("Valid 2023 : $(nrow(vl_23)) obs")

# Refit sur 2007-2022 + 2024 (exclure 2023 target)
tr_23 = dropmissing(filter(r -> r.annee != 2023, train), COLS_NEEDED)

# A) Pondéré decay 3
tr_a23 = copy(tr_23)
tr_a23.wt = exp.(-(2023 .- tr_a23.annee) ./ 3.0)
tr_a23.wt .*= nrow(tr_a23) / sum(tr_a23.wt)
model_a23 = glm(formula_full, tr_a23, Normal(), LogLink(); wts=tr_a23.wt)
preds_a23 = recursive_predict_glm(model_a23, vl_23, train, median(tr_23.PM_pos))
m_a23 = metrics(preds_a23, vl_23.PM)

# B) Par station
tr_b23 = tr_23
model_global_23 = glm(formula_full, tr_b23, Normal(), LogLink())
models_b23 = Dict{Any, Any}()
for sdf in groupby(tr_b23, :stationId)
    sid = first(sdf.stationId)
    models_b23[sid] = nrow(sdf) >= 500 ?
        try glm(formula_full, sdf, Normal(), LogLink()) catch; model_global_23 end :
        model_global_23
end
preds_b23 = recursive_predict_by_station(models_b23, model_global_23, vl_23, train, median(tr_23.PM_pos))
m_b23 = metrics(preds_b23, vl_23.PM)

# C) Gating + oversample ×3
tr_c23 = copy(tr_23); tr_c23.is_fumee = tr_c23.PM .>= FUMEE
tr_c23.wt_fumee = ifelse.(tr_c23.is_fumee, 3.0, 1.0)
tr_c23.wt_fumee .*= nrow(tr_c23) / sum(tr_c23.wt_fumee)
clf_23   = glm(formula_clf,   tr_c23, Binomial(), LogitLink(); wts=tr_c23.wt_fumee)
glm_n_23 = glm(formula_full,  tr_c23, Normal(),   LogLink())
glm_f_23 = glm(formula_fumee, tr_c23, Normal(),   LogLink();   wts=tr_c23.wt_fumee)
preds_c23 = recursive_predict_gating(clf_23, glm_n_23, glm_f_23, vl_23, train, median(tr_23.PM_pos))
m_c23 = metrics(preds_c23, vl_23.PM)

# D) VIF + stepwise
model_d23 = glm(formula_d, tr_23, Normal(), LogLink())
preds_d23 = recursive_predict_glm(model_d23, vl_23, train, median(tr_23.PM_pos))
m_d23 = metrics(preds_d23, vl_23.PM)

println()
for (name, m) in [("A) Pondéré 3 ans", m_a23),
                  ("B) Par station", m_b23),
                  ("C) Gating + ×3", m_c23),
                  ("D) VIF + stepwise", m_d23)]
    println(rpad(name, 22) *
            " | RMSE=" * rpad(round(m.rmse, digits=3), 6) *
            " | normal=" * rpad(round(m.rmse_normal, digits=3), 6) *
            " | fumée=" * rpad(round(m.rmse_fumee, digits=3), 7) *
            " | max=" * rpad(round(m.max_pred, digits=1), 5))
end

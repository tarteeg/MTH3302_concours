# =============================================================================
# model_selection.jl — Sélection de variables formelle
#
# Pipeline :
#   1. Partager le même preprocessing que model_weighted.jl
#   2. Calculer le VIF pour toutes les variables candidates et éliminer
#      itérativement celles avec VIF > VIF_THRESHOLD
#   3. Stepwise ascendant (forward) par AIC sur les variables restantes
#   4. Évaluer le modèle final sur 2024 (holdout)
#
# Sortie : formule optimale + tableau des étapes
# =============================================================================

include("model_weighted.jl")  # réutilise preprocess, COLS_NEEDED, formula_full

using GLM, Statistics, DataFrames, LinearAlgebra, StatsBase

const VIF_THRESHOLD = 10.0
const HOLDOUT_YEAR  = 2024

# -----------------------------------------------------------------------------
# 1. Ensemble de variables candidates (toutes les explicatives de formula_full)
# -----------------------------------------------------------------------------
const CANDIDATES = [:PM_roll3, :PM_lag1, :NO2_roll3,
                    :vitesse_vent_moy, :visibilite_moy, :hum_rel_moy,
                    :temp_moy, :mois, :vent_x_visibilite, :jours_sans_pluie,
                    :mois_sin, :mois_cos, :temp_x_mois_sin, :temp_x_mois_cos,
                    :visibilite_min_roll3, :temp_max_roll7, :deficit_pluie14,
                    :is_fire_season, :fire_vis_inter, :fire_tmax_inter,
                    :is_winter, :temp_min_roll3, :vent_min_roll3,
                    :winter_tmin_inter, :winter_vent_inter]

# Jeu propre pour la sélection (exclut 2023 comme les modèles finaux)
train_sel = filter(r -> r.annee != 2023, dropmissing(train, COLS_NEEDED))
println("\n=== Sélection sur n=$(nrow(train_sel)) obs (2007-2024 sans 2023) ===")

# Convertir booléens en Float pour matrice numérique
to_num(v) = eltype(v) == Bool ? Float64.(v) : Float64.(v)

# -----------------------------------------------------------------------------
# 2. VIF — élimination itérative
# -----------------------------------------------------------------------------
"""
    compute_vif(df, vars) -> Dict{Symbol, Float64}

VIF_j = 1 / (1 - R²_j), où R²_j provient de la régression de X_j sur les autres X.
"""
function compute_vif(df::DataFrame, vars::Vector{Symbol})
    X = hcat([to_num(df[!, v]) for v in vars]...)
    n, p = size(X)
    vifs = Dict{Symbol, Float64}()
    for j in 1:p
        y = X[:, j]
        Xo = X[:, setdiff(1:p, j)]
        # Régression linéaire y ~ Xo avec intercept
        Xaug = hcat(ones(n), Xo)
        β = Xaug \ y
        ŷ = Xaug * β
        ss_res = sum((y .- ŷ).^2)
        ss_tot = sum((y .- mean(y)).^2)
        r2 = ss_tot == 0 ? 0.0 : 1 - ss_res / ss_tot
        vifs[vars[j]] = r2 >= 1 ? Inf : 1 / (1 - r2)
    end
    vifs
end

println("\n=== Étape 1 : VIF itératif (seuil $VIF_THRESHOLD) ===")
vars_vif = copy(CANDIDATES)
global iter_vif = 0
while true
    global iter_vif += 1
    vifs = compute_vif(train_sel, vars_vif)
    (v_max, vif_max) = reduce((a, b) -> a[2] > b[2] ? a : b, vifs)
    if vif_max <= VIF_THRESHOLD
        println("Itération $iter_vif : toutes les VIF ≤ $VIF_THRESHOLD. Stop.")
        break
    end
    println("Itération $iter_vif : retrait $v_max (VIF=$(round(vif_max, digits=2)))")
    filter!(!=(v_max), vars_vif)
    length(vars_vif) < 2 && break
end

println("\nVariables retenues après VIF ($(length(vars_vif))) :")
for v in vars_vif
    println("  - $v")
end

# -----------------------------------------------------------------------------
# 3. Stepwise ascendant (forward) par AIC
# -----------------------------------------------------------------------------
"""
    fit_glm(selected) -> modèle

Ajuste un Gaussian(LogLink) sur PM_pos avec les variables `selected`.
Retourne (modèle, aic).
"""
function fit_glm(selected::Vector{Symbol})
    isempty(selected) && return (nothing, Inf)
    rhs = join(string.(selected), " + ")
    fstr = "PM_pos ~ " * rhs
    f = eval(Meta.parse("@formula($fstr)"))
    model = glm(f, train_sel, Normal(), LogLink())
    (model, aic(model))
end

println("\n=== Étape 2 : Stepwise ascendant par AIC ===")
global selected  = Symbol[]
global remaining = copy(vars_vif)

# Modèle de base : intercept seul → calculons via y ~ 1
fit_intercept = glm(@formula(PM_pos ~ 1), train_sel, Normal(), LogLink())
global current_aic = aic(fit_intercept)
println("AIC intercept seul = $(round(current_aic, digits=1))")

global step_n = 0
while !isempty(remaining)
    global step_n += 1
    best_aic = current_aic
    best_var = nothing
    for v in remaining
        (_, candidate_aic) = fit_glm(vcat(selected, v))
        if candidate_aic < best_aic
            best_aic = candidate_aic
            best_var = v
        end
    end
    if best_var === nothing
        println("Étape $step_n : aucun ajout n'améliore l'AIC. Stop.")
        break
    end
    push!(selected, best_var)
    filter!(!=(best_var), remaining)
    Δ = current_aic - best_aic
    global current_aic = best_aic
    println("Étape $step_n : +$best_var  |  AIC=$(round(current_aic, digits=1))  (ΔAIC=$(round(Δ, digits=1)))")
end

println("\n=== Formule finale ($(length(selected)) variables) ===")
println("PM_pos ~ " * join(string.(selected), " + "))

# -----------------------------------------------------------------------------
# 4. Évaluation RMSE sur 2024
# -----------------------------------------------------------------------------
println("\n=== Étape 3 : Évaluation holdout $HOLDOUT_YEAR ===")
tr = filter(r -> year(r.Date) < HOLDOUT_YEAR && r.annee != 2023,
            dropmissing(train, COLS_NEEDED))
vl = filter(r -> year(r.Date) == HOLDOUT_YEAR, train)

rhs_final = join(string.(selected), " + ")
formula_final = eval(Meta.parse("@formula(PM_pos ~ $rhs_final)"))
model_final   = glm(formula_final, tr, Normal(), LogLink())

preds = recursive_predict_glm(model_final, vl, train, median(tr.PM_pos))
actual = Float64.(vl.PM)
mask = .!ismissing.(actual)
rmse = sqrt(mean((preds[mask] .- actual[mask]).^2))
mae  = mean(abs.(preds[mask] .- actual[mask]))

println("RMSE $HOLDOUT_YEAR (formule sélectionnée) = $(round(rmse, digits=3))")
println("MAE  $HOLDOUT_YEAR                       = $(round(mae, digits=3))")
println("Pour comparaison, formule complète : RMSE ≈ 9.26")

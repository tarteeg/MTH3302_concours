# =============================================================================
# model_recursive_log.jl — GLM avec PM_lag + target log1p + prédiction récursive
#
# Leviers appliqués :
#   1. PM_lag1 et PM_roll3 RÉACTIVÉS, avec prédiction récursive jour-par-jour
#      sur le test (pas de fuite : on utilise les prédictions antérieures).
#   2. Target transformée en log1p(PM) pour gérer l'asymétrie de la distribution.
#
# Étapes :
#   a) Préprocess train + test (réutilise la logique de model_nolag.jl)
#   b) Validation holdout 2024 : fit sur <2024, prédit 2024 récursivement
#      → donne un RMSE comparable au modèle sans lag (baseline ~18).
#   c) Refit sur 2007-2024 complet, prédit 2025 récursivement.
#   d) Sauve + compare.
# =============================================================================

using CSV, DataFrames, Dates, GLM, Statistics, LinearAlgebra

const TRAIN_AIR   = "data/qualite-de-lair_train.csv"
const TRAIN_METEO = "data/meteo_train.csv"
const TEST_AIR    = "data/qualite-de-lair_test.csv"
const TEST_METEO  = "data/meteo_test.csv"
const OUT_PATH    = "benchmark_predictions_recursive_log.csv"

const SO2_STATIONS   = ["Saint-Jean-Baptiste ", "Saint-Dominique",
                         "Saint-Joseph", "Anjou"]
const FIRE_TAIL_DAYS = 14
const FIRE_MONTHS    = [5, 6, 7, 8, 9]
const WINTER_MONTHS  = [12, 1, 2]

# -----------------------------------------------------------------------------
# Helpers (identiques à model_nolag.jl)
# -----------------------------------------------------------------------------
function interpolate(v)
    v_int = copy(v); n = length(v_int)
    fwd_val, fwd_dist = similar(v_int), zeros(Int, n)
    last_val, dist = missing, 0
    for i in 1:n
        if !ismissing(v_int[i]); last_val, dist = v_int[i], 0
        else; dist += 1; end
        fwd_val[i], fwd_dist[i] = last_val, dist
    end
    bwd_val, bwd_dist = similar(v_int), zeros(Int, n)
    last_val, dist = missing, 0
    for i in n:-1:1
        if !ismissing(v_int[i]); last_val, dist = v_int[i], 0
        else; dist += 1; end
        bwd_val[i], bwd_dist[i] = last_val, dist
    end
    for i in 1:n
        if ismissing(v_int[i])
            if !ismissing(fwd_val[i]) && !ismissing(bwd_val[i])
                total = fwd_dist[i] + bwd_dist[i]
                v_int[i] = round(Int,
                    (fwd_val[i] * bwd_dist[i] + bwd_val[i] * fwd_dist[i]) / total)
            elseif !ismissing(fwd_val[i]); v_int[i] = fwd_val[i]
            elseif !ismissing(bwd_val[i]); v_int[i] = bwd_val[i]
            end
        end
    end
    v_int
end

apply_summer!(v, m) = (for i in eachindex(v); 5 <= m[i] <= 9 && (v[i] = 0); end; v)

function treat_so2_missing_values(df::DataFrame, active::Vector{String})::DataFrame
    data = copy(df); sort!(data, [:nom, :Date])
    for sdf in groupby(data, :nom)
        first(sdf.nom) in active && (sdf.SO2 = interpolate(sdf.SO2))
    end
    dm = combine(groupby(data, :Date),
        :SO2 => (x -> isempty(skipmissing(x)) ? missing : median(skipmissing(x))) => :SO2_dm)
    dt = leftjoin(data, dm, on=:Date)
    dt.SO2 = coalesce.(dt.SO2, dt.SO2_dm); data.SO2 .= dt.SO2
    for sdf in groupby(data, :nom); sdf.SO2 = interpolate(sdf.SO2); end
    sort!(data, :Date); data
end

function roll_mean(v::AbstractVector, n::Int)
    out = Vector{Union{Float64, Missing}}(missing, length(v))
    for i in n:length(v)
        vals = collect(skipmissing(v[max(1, i-n+1):i]))
        out[i] = isempty(vals) ? missing : mean(vals)
    end
    out
end

function lag_vec(v::AbstractVector, n::Int=1)
    T = Union{eltype(v), Missing}
    out = Vector{T}(missing, length(v))
    out[n+1:end] = v[1:end-n]; out
end

function jours_sans_pluie_vec(pluie::AbstractVector)
    n = length(pluie); out = Vector{Int}(undef, n)
    for i in 1:n
        if i == 1 || ismissing(pluie[i-1]);    out[i] = 0
        elseif coalesce(pluie[i], 0.0) > 0;     out[i] = 0
        else;                                   out[i] = out[i-1] + 1; end
    end; out
end

function rolling_mean(v, w, minn)
    n = length(v); out = Vector{Union{Missing, Float64}}(missing, n)
    for i in 1:n
        if i >= w
            vv = collect(skipmissing(v[max(1, i-w+1):i]))
            length(vv) >= minn && (out[i] = mean(vv))
        end
    end; out
end

function rolling_sum(v, w, minn)
    n = length(v); out = Vector{Union{Missing, Float64}}(missing, n)
    for i in 1:n
        if i >= w
            vv = collect(skipmissing(v[max(1, i-w+1):i]))
            length(vv) >= minn && (out[i] = sum(vv))
        end
    end; out
end

# -----------------------------------------------------------------------------
# 1. Chargement + preprocess train
# -----------------------------------------------------------------------------
println("=== 1. Chargement + preprocess train ===")
train_air   = CSV.read(TRAIN_AIR,   DataFrame)
train_meteo = CSV.read(TRAIN_METEO, DataFrame)
train = outerjoin(train_air, train_meteo[:, 3:end], on=:Date)
train = filter(row -> !ismissing(row.stationId), train)
train = DataFrames.transform(train, :Date => ByRow(month) => :mois)
train = sort(train, :Date)
train = filter(row -> ismissing(row.PM) || row.PM < 500, train)

train.month = month.(train.Date)
sort!(train, [:nom, :Date])
transform!(groupby(train, :nom),
    [:neige_au_sol, :month] =>
    ((v, m) -> apply_summer!(interpolate(v), m)) => :neige_au_sol)
sort!(train, :Date)

train = treat_so2_missing_values(train, SO2_STATIONS)
train = dropmissing(train, :PM)
for sdf in groupby(train, :nom)
    sdf.NO2        = interpolate(sdf.NO2)
    sdf.pluie      = interpolate(sdf.pluie)
    sdf.neige      = interpolate(sdf.neige)
    sdf.precip_tot = interpolate(sdf.precip_tot)
    sdf.O3         = interpolate(sdf.O3)
end

# -----------------------------------------------------------------------------
# 2. Feature engineering train
# -----------------------------------------------------------------------------
println("=== 2. Feature engineering train ===")
sort!(train, [:stationId, :Date])
transform!(groupby(train, :stationId),
    :PM    => (x -> lag_vec(x, 1))   => :PM_lag1,
    :PM    => (x -> roll_mean(x, 3)) => :PM_roll3,
    :NO2   => (x -> roll_mean(x, 3)) => :NO2_roll3,
    :pluie => jours_sans_pluie_vec   => :jours_sans_pluie)

train.vent_x_visibilite = train.vitesse_vent_moy .* train.visibilite_moy
train.mois_sin          = sin.(2π .* train.mois ./ 12)
train.mois_cos          = cos.(2π .* train.mois ./ 12)
train.temp_x_mois_sin   = train.temp_moy .* train.mois_sin
train.temp_x_mois_cos   = train.temp_moy .* train.mois_cos

transform!(groupby(train, :stationId),
    :visibilite_min => (v -> rolling_mean(v,  3, 2)) => :visibilite_min_roll3,
    :temp_max       => (v -> rolling_mean(v,  7, 4)) => :temp_max_roll7,
    :pluie          => (v -> rolling_sum(v,  14, 7)) => :pluie_cum14)
train.is_fire_season = in.(train.mois, Ref(FIRE_MONTHS))

Q25_PLUIE14 = quantile(collect(skipmissing(train.pluie_cum14)), 0.25)
train.deficit_pluie14 = Int.(coalesce.(train.pluie_cum14 .< Q25_PLUIE14, false))
train.fire_vis_inter  = ifelse.(train.is_fire_season,
                                coalesce.(train.visibilite_min_roll3, 0.0), 0.0)
train.fire_tmax_inter = ifelse.(train.is_fire_season,
                                coalesce.(train.temp_max_roll7, 0.0), 0.0)

transform!(groupby(train, :stationId),
    :temp_min         => (v -> rolling_mean(v, 3, 2)) => :temp_min_roll3,
    :vitesse_vent_min => (v -> rolling_mean(v, 3, 2)) => :vent_min_roll3)
for col in [:temp_min_roll3, :vent_min_roll3]
    med = median(skipmissing(train[:, col]))
    train[!, col] = Float64.(coalesce.(train[:, col], med))
end
train.is_winter         = in.(train.mois, Ref(WINTER_MONTHS))
train.winter_tmin_inter = ifelse.(train.is_winter, train.temp_min_roll3, 0.0)
train.winter_vent_inter = ifelse.(train.is_winter, train.vent_min_roll3, 0.0)

# Target log-transformée
train.log_PM = log1p.(train.PM)

# -----------------------------------------------------------------------------
# 3. Formule (avec PM_lag1, PM_roll3)
# -----------------------------------------------------------------------------
formula_full = @formula(log_PM ~ PM_roll3 + PM_lag1 + NO2_roll3 +
                                 vitesse_vent_moy + visibilite_moy + hum_rel_moy +
                                 temp_moy + mois + vent_x_visibilite + jours_sans_pluie +
                                 mois_sin + mois_cos + temp_x_mois_sin + temp_x_mois_cos +
                                 visibilite_min_roll3 + temp_max_roll7 + deficit_pluie14 +
                                 is_fire_season + fire_vis_inter + fire_tmax_inter +
                                 is_winter + temp_min_roll3 + vent_min_roll3 +
                                 winter_tmin_inter + winter_vent_inter)

COLS_NEEDED = [:PM, :log_PM, :PM_lag1, :PM_roll3, :NO2_roll3,
               :vitesse_vent_moy, :visibilite_moy, :hum_rel_moy,
               :temp_moy, :mois, :vent_x_visibilite, :jours_sans_pluie,
               :mois_sin, :mois_cos, :temp_x_mois_sin, :temp_x_mois_cos,
               :visibilite_min_roll3, :temp_max_roll7, :deficit_pluie14,
               :is_fire_season, :fire_vis_inter, :fire_tmax_inter,
               :is_winter, :temp_min_roll3, :vent_min_roll3,
               :winter_tmin_inter, :winter_vent_inter]

# -----------------------------------------------------------------------------
# 4. Fonction de prédiction récursive
# -----------------------------------------------------------------------------
"""
    recursive_predict(model, target_df, history_df) → Vector{Float64}

Prédit récursivement PM pour chaque ligne de `target_df` (triées par stationId, Date).
`history_df` fournit les PM observés AVANT le début de `target_df` (pour initialiser
les lags et roll3). La prédiction à J alimente le lag de J+1 et le roll3 de J+1, J+2.

Retourne les prédictions sur l'échelle originale (expm1 appliqué).
"""
function recursive_predict(model, target_df::DataFrame, history_df::DataFrame)
    n = nrow(target_df)
    preds = zeros(Float64, n)

    # Historique PM par station (Vector pour append in-place)
    history_pm = Dict{Any, Vector{Float64}}()
    history_date = Dict{Any, Vector{Date}}()
    for sdf in groupby(sort(history_df, [:stationId, :Date]), :stationId)
        sid = first(sdf.stationId)
        pm_vec = Float64[]
        dt_vec = Date[]
        for r in eachrow(sdf)
            !ismissing(r.PM) && (push!(pm_vec, Float64(r.PM)); push!(dt_vec, r.Date))
        end
        history_pm[sid]   = pm_vec
        history_date[sid] = dt_vec
    end

    # Tri par date puis par station pour respecter l'ordre temporel
    idx_sorted = sortperm(collect(zip(target_df.Date, target_df.stationId)))

    for i in idx_sorted
        row    = target_df[i, :]
        sid    = row.stationId
        hist   = get!(history_pm,   sid, Float64[])
        dates  = get!(history_date, sid, Date[])

        # Calcul des features dépendant de PM
        pm_lag1  = isempty(hist) ? missing : hist[end]
        pm_roll3 = if length(hist) >= 3; mean(hist[end-2:end])
                   elseif !isempty(hist); mean(hist); else; missing; end

        # Si manquant (début de série), imputer par médiane de train
        pm_lag1  = coalesce(pm_lag1,  median(train.PM))
        pm_roll3 = coalesce(pm_roll3, median(train.PM))

        # Construire la ligne de features (copie + override des 2 colonnes)
        row_df = DataFrame(target_df[i:i, :])
        row_df.PM_lag1  = [Float64(pm_lag1)]
        row_df.PM_roll3 = [Float64(pm_roll3)]

        # Prédiction sur échelle log → inverse
        yhat_log = GLM.predict(model, row_df)[1]
        yhat     = expm1(yhat_log)
        yhat     = max(yhat, 0.0)  # PM >= 0

        preds[i] = yhat
        push!(hist,  yhat)
        push!(dates, row.Date)
    end

    preds
end

# -----------------------------------------------------------------------------
# 5. Validation holdout 2024 (diagnostic avant prédiction 2025)
# -----------------------------------------------------------------------------
println("\n=== 3. Validation holdout 2024 ===")
train_hist   = filter(r -> year(r.Date) <  2024, train)
valid_target = filter(r -> year(r.Date) == 2024, train)

train_hist_fit = dropmissing(train_hist, COLS_NEEDED)
model_hold = lm(formula_full, train_hist_fit)

# Pour la validation, on fournit valid_target SANS PM_lag1/PM_roll3 actuels
# (on les recalcule récursivement). Mais on garde les autres features précalculées.
valid_work = copy(valid_target)
preds_2024 = recursive_predict(model_hold, valid_work, train_hist)

actual_2024 = Float64.(valid_target.PM)
mask = .!ismissing.(actual_2024)
rmse_2024 = sqrt(mean((actual_2024[mask] .- preds_2024[mask]) .^ 2))
mae_2024  = mean(abs.(actual_2024[mask] .- preds_2024[mask]))
println("RMSE 2024 (récursif + log) : $(round(rmse_2024, digits=3))")
println("MAE  2024                   : $(round(mae_2024,  digits=3))")
println("Moyenne prédite             : $(round(mean(preds_2024), digits=3))")
println("Moyenne réelle              : $(round(mean(actual_2024[mask]), digits=3))")
println("Max prédit                  : $(round(maximum(preds_2024), digits=3))")
println("Max réel                    : $(round(maximum(actual_2024[mask]), digits=3))")

# -----------------------------------------------------------------------------
# 6. Refit sur 2007-2024 + prédiction récursive 2025
# -----------------------------------------------------------------------------
println("\n=== 4. Fit final sur 2007-2024 ===")
train_fit_full = dropmissing(train, COLS_NEEDED)
model_final = lm(formula_full, train_fit_full)
println("n = $(nrow(train_fit_full)) obs — $(length(GLM.coef(model_final))) coefs")

# --- Préparation test (features non-PM uniquement) ---
println("\n=== 5. Préparation test ===")
test_air   = CSV.read(TEST_AIR,   DataFrame)
test_meteo = CSV.read(TEST_METEO, DataFrame)
test = outerjoin(test_air, test_meteo[:, 3:end], on=:Date)
test = filter(row -> !ismissing(row.stationId), test)
test = treat_so2_missing_values(test, SO2_STATIONS)
sort!(test, [:stationId, :Date])
test.mois              = month.(test.Date)
test.vent_x_visibilite  = test.vitesse_vent_moy .* test.visibilite_moy
test.mois_sin           = sin.(2π .* test.mois ./ 12)
test.mois_cos           = cos.(2π .* test.mois ./ 12)
test.temp_x_mois_sin    = test.temp_moy .* test.mois_sin
test.temp_x_mois_cos    = test.temp_moy .* test.mois_cos

test_date_min = minimum(test.Date)

# NO2_roll3 + jours_sans_pluie via plug tail train
train_tail = combine(groupby(sort(train, [:stationId, :Date]), :stationId)) do sdf
    last(sort(dropmissing(sdf, :PM), :Date), min(7, nrow(sdf)))
end
train_tail_slim = select(train_tail, [:stationId, :Date, :NO2, :pluie])
test_input = DataFrame(stationId=test.stationId, Date=test.Date,
                       NO2=test.NO2, pluie=test.pluie)
comb = vcat(train_tail_slim, test_input); sort!(comb, [:stationId, :Date])
transform!(groupby(comb, :stationId),
    :NO2   => (x -> roll_mean(x, 3)) => :NO2_roll3,
    :pluie => jours_sans_pluie_vec   => :jours_sans_pluie)
test_feats = filter(r -> r.Date >= test_date_min, comb)
select!(test_feats, [:stationId, :Date, :NO2_roll3, :jours_sans_pluie])
test = leftjoin(test, test_feats, on=[:stationId, :Date])

for col in [:NO2_roll3, :vitesse_vent_moy, :visibilite_moy, :hum_rel_moy, :temp_moy]
    med = median(skipmissing(train[:, col]))
    test[!, col] = Float64.(coalesce.(test[:, col], med))
end
test[!, :jours_sans_pluie] = Int.(coalesce.(test[:, :jours_sans_pluie], 0))

# Fire / winter features test (plug FIRE_TAIL_DAYS)
function add_rolling_test!(test_df, train_src, cols_plug, specs, out_cols)
    tail = combine(groupby(sort(train_src, [:stationId, :Date]), :stationId)) do sdf
        last(sdf, min(FIRE_TAIL_DAYS, nrow(sdf)))
    end
    tail_slim = select(tail, cols_plug)
    input = select(test_df, cols_plug)
    c = vcat(tail_slim, input); sort!(c, [:stationId, :Date])
    transform!(groupby(c, :stationId), specs...)
    f = filter(r -> r.Date >= test_date_min, c)
    select!(f, vcat([:stationId, :Date], out_cols))
    leftjoin(test_df, f, on=[:stationId, :Date])
end

test = add_rolling_test!(test, train,
    [:stationId, :Date, :visibilite_min, :temp_max, :pluie],
    [:visibilite_min => (v -> rolling_mean(v,  3, 2)) => :visibilite_min_roll3,
     :temp_max       => (v -> rolling_mean(v,  7, 4)) => :temp_max_roll7,
     :pluie          => (v -> rolling_sum(v,  14, 7)) => :pluie_cum14],
    [:visibilite_min_roll3, :temp_max_roll7, :pluie_cum14])
for col in [:visibilite_min_roll3, :temp_max_roll7, :pluie_cum14]
    med = median(skipmissing(train[:, col]))
    test[!, col] = Float64.(coalesce.(test[:, col], med))
end
test.is_fire_season  = in.(test.mois, Ref(FIRE_MONTHS))
test.deficit_pluie14 = Int.(test.pluie_cum14 .< Q25_PLUIE14)
test.fire_vis_inter  = ifelse.(test.is_fire_season, test.visibilite_min_roll3, 0.0)
test.fire_tmax_inter = ifelse.(test.is_fire_season, test.temp_max_roll7,       0.0)

test = add_rolling_test!(test, train,
    [:stationId, :Date, :temp_min, :vitesse_vent_min],
    [:temp_min         => (v -> rolling_mean(v, 3, 2)) => :temp_min_roll3,
     :vitesse_vent_min => (v -> rolling_mean(v, 3, 2)) => :vent_min_roll3],
    [:temp_min_roll3, :vent_min_roll3])
for col in [:temp_min_roll3, :vent_min_roll3]
    med = median(skipmissing(train[:, col]))
    test[!, col] = Float64.(coalesce.(test[:, col], med))
end
test.is_winter         = in.(test.mois, Ref(WINTER_MONTHS))
test.winter_tmin_inter = ifelse.(test.is_winter, test.temp_min_roll3, 0.0)
test.winter_vent_inter = ifelse.(test.is_winter, test.vent_min_roll3, 0.0)

# Placeholders pour PM_lag1 / PM_roll3 (seront écrasés dans la boucle récursive)
test.PM_lag1  = zeros(Float64, nrow(test))
test.PM_roll3 = zeros(Float64, nrow(test))

# -----------------------------------------------------------------------------
# 7. Prédiction récursive 2025
# -----------------------------------------------------------------------------
println("\n=== 6. Prédiction récursive 2025 ===")
predictions = recursive_predict(model_final, test, train)

# Reconstruction dans l'ordre de test
output = DataFrame(
    ID = [(test.Date[i], test.stationId[i]) for i in 1:nrow(test)],
    PM = predictions)
CSV.write(OUT_PATH, output)
println("→ $OUT_PATH ($(nrow(output)) lignes)")
println("  Moyenne PM prédit : $(round(mean(predictions), digits=3))")
println("  Max PM prédit     : $(round(maximum(predictions), digits=3))")
println("  Min PM prédit     : $(round(minimum(predictions), digits=3))")

# -----------------------------------------------------------------------------
# 8. Comparaison avec le modèle nolag actuel
# -----------------------------------------------------------------------------
if isfile("benchmark_predictions_nolag_recreated.csv")
    println("\n=== 7. Comparaison avec nolag ===")
    nolag = CSV.read("benchmark_predictions_nolag_recreated.csv", DataFrame)
    cor_val = cor(output.PM, nolag.PM)
    diff = output.PM .- nolag.PM
    println("Corrélation vs nolag : $(round(cor_val, digits=4))")
    println("Moyenne diff         : $(round(mean(diff), digits=3))")
    println("|diff| max           : $(round(maximum(abs.(diff)), digits=3))")
end

println("\n🎯 RMSE 2024 attendu à comparer avec baseline ~18")

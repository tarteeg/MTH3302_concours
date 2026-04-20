# =============================================================================
# model_nolag.jl — Recréation du modèle section 5.3
# GLM fire + winter SANS PM_lag1 ni PM_roll3
#
# Reproduit la logique du notebook projet_h2026_backup_ridge_20260418_0024.ipynb
# (sections 1, 3.2-3.4, 5.1-5.3) pour régénérer benchmark_predictions_nolag.csv
#
# Usage  :  julia --project=. model_nolag.jl
# Sortie :  benchmark_predictions_nolag_recreated.csv
# =============================================================================

using CSV, DataFrames, Dates, GLM, Statistics, LinearAlgebra

# -----------------------------------------------------------------------------
# 1. Chemins
# -----------------------------------------------------------------------------
const TRAIN_AIR_PATH   = "data/qualite-de-lair_train.csv"
const TRAIN_METEO_PATH = "data/meteo_train.csv"
const TEST_AIR_PATH    = "data/qualite-de-lair_test.csv"
const TEST_METEO_PATH  = "data/meteo_test.csv"
const OUTPUT_PATH      = "benchmark_predictions_nolag_recreated.csv"
const REFERENCE_PATH   = "submission/benchmark_predictions_nolag.csv"

const SO2_STATIONS   = ["Saint-Jean-Baptiste ", "Saint-Dominique",
                         "Saint-Joseph", "Anjou"]
const FIRE_TAIL_DAYS = 14
const FIRE_MONTHS    = [5, 6, 7, 8, 9]
const WINTER_MONTHS  = [12, 1, 2]

# -----------------------------------------------------------------------------
# 2. Helpers — imputation, lags, rollings
# -----------------------------------------------------------------------------
function interpolate(v)
    v_int = copy(v)
    n = length(v_int)
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

function apply_summer!(v, m)
    for i in eachindex(v)
        5 <= m[i] <= 9 && (v[i] = 0)
    end
    v
end

function treat_so2_missing_values(df::DataFrame, active_stations::Vector{String})::DataFrame
    data = copy(df)
    sort!(data, [:nom, :Date])
    for sdf in groupby(data, :nom)
        first(sdf.nom) in active_stations && (sdf.SO2 = interpolate(sdf.SO2))
    end
    daily_median = combine(groupby(data, :Date),
        :SO2 => (x -> isempty(skipmissing(x)) ? missing : median(skipmissing(x))) => :SO2_daily_median)
    data_temp = leftjoin(data, daily_median, on=:Date)
    data_temp.SO2 = coalesce.(data_temp.SO2, data_temp.SO2_daily_median)
    data.SO2 .= data_temp.SO2
    for sdf in groupby(data, :nom)
        sdf.SO2 = interpolate(sdf.SO2)
    end
    sort!(data, :Date)
    data
end

function lag_vec(v::AbstractVector, n::Int=1)
    T = Union{eltype(v), Missing}
    out = Vector{T}(missing, length(v))
    out[n+1:end] = v[1:end-n]
    out
end

function roll_mean(v::AbstractVector, n::Int)
    out = Vector{Union{Float64, Missing}}(missing, length(v))
    for i in n:length(v)
        vals = collect(skipmissing(v[max(1, i-n+1):i]))
        out[i] = isempty(vals) ? missing : mean(vals)
    end
    out
end

function jours_sans_pluie_vec(pluie::AbstractVector)
    n = length(pluie)
    out = Vector{Int}(undef, n)
    for i in 1:n
        if i == 1 || ismissing(pluie[i-1]);       out[i] = 0
        elseif coalesce(pluie[i], 0.0) > 0;        out[i] = 0
        else;                                      out[i] = out[i-1] + 1; end
    end
    out
end

function rolling_mean(v, w, minn)
    n = length(v); out = Vector{Union{Missing, Float64}}(missing, n)
    for i in 1:n
        if i >= w
            vv = collect(skipmissing(v[max(1, i-w+1):i]))
            length(vv) >= minn && (out[i] = mean(vv))
        end
    end
    out
end

function rolling_sum(v, w, minn)
    n = length(v); out = Vector{Union{Missing, Float64}}(missing, n)
    for i in 1:n
        if i >= w
            vv = collect(skipmissing(v[max(1, i-w+1):i]))
            length(vv) >= minn && (out[i] = sum(vv))
        end
    end
    out
end

# -----------------------------------------------------------------------------
# 3. Chargement et préparation de `train`
# -----------------------------------------------------------------------------
println("=== 1. Chargement ===")
train_air   = CSV.read(TRAIN_AIR_PATH,   DataFrame)
train_meteo = CSV.read(TRAIN_METEO_PATH, DataFrame)
train = outerjoin(train_air, train_meteo[:, 3:end], on=:Date)
train = filter(row -> !ismissing(row.stationId), train)
train = DataFrames.transform(train, :Date => ByRow(month) => :mois)
train = sort(train, :Date)

# Retrait de l'anomalie PM = 701 (cf. section 2.2.5)
train = filter(row -> ismissing(row.PM) || row.PM < 500, train)

# --- 3.2.1 neige_au_sol : interpolation + 0 en saison estivale ------------
train.month = month.(train.Date)
sort!(train, [:nom, :Date])
transform!(groupby(train, :nom),
    [:neige_au_sol, :month] =>
    ((v, m) -> apply_summer!(interpolate(v), m)) => :neige_au_sol)
sort!(train, :Date)

# --- 3.2.2 SO2 ------------------------------------------------------------
train = treat_so2_missing_values(train, SO2_STATIONS)

# --- 3.2.3 Imputations résiduelles ----------------------------------------
train = dropmissing(train, :PM)
for sdf in groupby(train, :nom)
    sdf.NO2        = interpolate(sdf.NO2)
    sdf.pluie      = interpolate(sdf.pluie)
    sdf.neige      = interpolate(sdf.neige)
    sdf.precip_tot = interpolate(sdf.precip_tot)
    sdf.O3         = interpolate(sdf.O3)
end

# -----------------------------------------------------------------------------
# 4. Feature engineering sur train — identique au notebook
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

# Fire features
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

# Winter features
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

# -----------------------------------------------------------------------------
# 5. Chargement et traitement de `test` (section 5.1 + 5.2)
# -----------------------------------------------------------------------------
println("=== 3. Préparation test ===")
test_air   = CSV.read(TEST_AIR_PATH,   DataFrame)
test_meteo = CSV.read(TEST_METEO_PATH, DataFrame)
test = outerjoin(test_air, test_meteo[:, 3:end], on=:Date)
test = filter(row -> !ismissing(row.stationId), test)

test = treat_so2_missing_values(test, SO2_STATIONS)
sort!(test, [:stationId, :Date])
test.mois             = month.(test.Date)
test.vent_x_visibilite = test.vitesse_vent_moy .* test.visibilite_moy
test.mois_sin          = sin.(2π .* test.mois ./ 12)
test.mois_cos          = cos.(2π .* test.mois ./ 12)
test.temp_x_mois_sin   = test.temp_moy .* test.mois_sin
test.temp_x_mois_cos   = test.temp_moy .* test.mois_cos

test_date_min = minimum(test.Date)

# --- 5.2b : NO2_roll3 + jours_sans_pluie (plug tail train) ----------------
train_tail = combine(groupby(sort(train, [:stationId, :Date]), :stationId)) do sdf
    sdf_pm = dropmissing(sdf, :PM)
    isempty(sdf_pm) ? DataFrame() : last(sort(sdf_pm, :Date), min(7, nrow(sdf_pm)))
end
train_tail_slim = select(train_tail, [:stationId, :Date, :NO2, :pluie])

test_lag_input = DataFrame(
    stationId = test.stationId,
    Date      = test.Date,
    NO2       = test.NO2,
    pluie     = test.pluie)

combined = vcat(train_tail_slim, test_lag_input)
sort!(combined, [:stationId, :Date])

transform!(groupby(combined, :stationId),
    :NO2   => (x -> roll_mean(x, 3)) => :NO2_roll3,
    :pluie => jours_sans_pluie_vec   => :jours_sans_pluie)

test_feats = filter(row -> row.Date >= test_date_min, combined)
select!(test_feats, [:stationId, :Date, :NO2_roll3, :jours_sans_pluie])
test = leftjoin(test, test_feats, on=[:stationId, :Date])

for col in [:NO2_roll3, :vitesse_vent_moy, :visibilite_moy, :hum_rel_moy, :temp_moy]
    med = median(skipmissing(train[:, col]))
    test[!, col] = Float64.(coalesce.(test[:, col], med))
end
test[!, :jours_sans_pluie] = Int.(coalesce.(test[:, :jours_sans_pluie], 0))

# --- 5.2c : fire + winter features sur test -------------------------------
function build_rolling_feature!(test_df, train_src, cols_plug, roll_specs, out_cols)
    tail = combine(groupby(sort(train_src, [:stationId, :Date]), :stationId)) do sdf
        last(sdf, min(FIRE_TAIL_DAYS, nrow(sdf)))
    end
    tail_slim = select(tail, cols_plug)
    input = select(test_df, cols_plug)
    comb = vcat(tail_slim, input)
    sort!(comb, [:stationId, :Date])
    transform!(groupby(comb, :stationId), roll_specs...)
    feats = filter(row -> row.Date >= test_date_min, comb)
    select!(feats, vcat([:stationId, :Date], out_cols))
    leftjoin(test_df, feats, on=[:stationId, :Date])
end

test = build_rolling_feature!(test, train,
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

test = build_rolling_feature!(test, train,
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

# -----------------------------------------------------------------------------
# 6. GLM final — SANS PM_lag1 ni PM_roll3 (formule section 5.3)
# -----------------------------------------------------------------------------
println("=== 4. Entraînement GLM nolag ===")
formula_nolag = @formula(PM ~ NO2_roll3 +
                              vitesse_vent_moy + visibilite_moy + hum_rel_moy +
                              temp_moy + mois + vent_x_visibilite + jours_sans_pluie +
                              mois_sin + mois_cos + temp_x_mois_sin + temp_x_mois_cos +
                              visibilite_min_roll3 + temp_max_roll7 + deficit_pluie14 +
                              is_fire_season + fire_vis_inter + fire_tmax_inter +
                              is_winter + temp_min_roll3 + vent_min_roll3 +
                              winter_tmin_inter + winter_vent_inter)

COLS_NOLAG = [:PM, :NO2_roll3,
              :vitesse_vent_moy, :visibilite_moy, :hum_rel_moy,
              :temp_moy, :mois, :vent_x_visibilite, :jours_sans_pluie,
              :mois_sin, :mois_cos, :temp_x_mois_sin, :temp_x_mois_cos,
              :visibilite_min_roll3, :temp_max_roll7, :deficit_pluie14,
              :is_fire_season, :fire_vis_inter, :fire_tmax_inter,
              :is_winter, :temp_min_roll3, :vent_min_roll3,
              :winter_tmin_inter, :winter_vent_inter]

train_fit = dropmissing(train, COLS_NOLAG)
println("Entraînement sur $(nrow(train_fit)) observations (2007-2024)")

model = lm(formula_nolag, train_fit)
println("\n=== Coefficients ===")
println(coeftable(model))

# -----------------------------------------------------------------------------
# 7. Prédictions 2025 + fichier Kaggle
# -----------------------------------------------------------------------------
println("\n=== 5. Prédictions 2025 ===")
predictions = GLM.predict(model, test)

function createID(date::Vector{<:Date}, station::Vector{<:Any})
    [(date[i], station[i]) for i in 1:length(date)]
end

benchmarkID = createID(test.Date, test.stationId)
output = DataFrame(ID=benchmarkID, PM=predictions)
CSV.write(OUTPUT_PATH, output)
println("→ $OUTPUT_PATH ($(nrow(output)) lignes)")

# -----------------------------------------------------------------------------
# 8. Comparaison avec benchmark_predictions_nolag.csv d'origine
# -----------------------------------------------------------------------------
if isfile(REFERENCE_PATH)
    println("\n=== 6. Comparaison avec $REFERENCE_PATH ===")
    ref = CSV.read(REFERENCE_PATH, DataFrame)
    if nrow(ref) != nrow(output)
        println("⚠ Taille différente : ref=$(nrow(ref)), recréé=$(nrow(output))")
    else
        diff = output.PM .- ref.PM
        rmse_diff = sqrt(mean(diff .^ 2))
        max_abs = maximum(abs.(diff))
        mae = mean(abs.(diff))
        cor_val = cor(output.PM, ref.PM)
        println("RMSE(différence)    : $(round(rmse_diff, digits=4))")
        println("MAE(différence)     : $(round(mae, digits=4))")
        println("|diff| max          : $(round(max_abs, digits=4))")
        println("Corrélation PM      : $(round(cor_val, digits=6))")
        println("Moyenne PM recréé   : $(round(mean(output.PM), digits=3))")
        println("Moyenne PM référence: $(round(mean(ref.PM), digits=3))")
        if rmse_diff < 1e-6
            println("✓ Prédictions IDENTIQUES — modèle fidèlement reproduit")
        elseif rmse_diff < 0.5
            println("✓ Prédictions quasi-identiques (différences numériques)")
        else
            println("⚠ Écart non négligeable — vérifier le pipeline de features")
        end
    end
else
    println("\n(Fichier de référence $REFERENCE_PATH introuvable, skip comparaison)")
end

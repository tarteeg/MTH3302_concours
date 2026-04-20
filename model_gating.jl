# =============================================================================
# model_gating.jl — Option A : modèle à 2 régimes avec classifieur logistique
#
# Architecture :
#   1. Classifieur logistique : P(fumée) = f(visibilité, temp, saison, PM_lag, ...)
#   2. GLM_normal : entraîné sur jours PM ≤ 35
#   3. GLM_fumée  : entraîné sur jours PM > 35
#   4. Prédiction finale = P(fumée) × GLM_fumée + (1-P(fumée)) × GLM_normal
#
# Le classifieur peut utiliser PM_lag (récursif en test) car il ne prédit pas PM
# mais une classe, ce qui permet d'intégrer la mémoire temporelle.
# =============================================================================

using CSV, DataFrames, Dates, GLM, Statistics, LinearAlgebra

const TRAIN_AIR   = "data/qualite-de-lair_train.csv"
const TRAIN_METEO = "data/meteo_train.csv"
const TEST_AIR    = "data/qualite-de-lair_test.csv"
const TEST_METEO  = "data/meteo_test.csv"
const OUT_PATH    = "benchmark_predictions_gating.csv"

const SO2_STATIONS   = ["Saint-Jean-Baptiste ", "Saint-Dominique",
                         "Saint-Joseph", "Anjou"]
const FIRE_TAIL_DAYS = 14
const FIRE_MONTHS    = [5, 6, 7, 8, 9]
const WINTER_MONTHS  = [12, 1, 2]
const FUMEE_THRESHOLD = 35.0

# -----------------------------------------------------------------------------
# Helpers (identiques)
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
    end; out
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
# 1. Preprocess train
# -----------------------------------------------------------------------------
println("=== 1. Preprocess train ===")
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

sort!(train, [:stationId, :Date])
transform!(groupby(train, :stationId),
    :PM    => (x -> lag_vec(x, 1))   => :PM_lag1,
    :PM    => (x -> roll_mean(x, 3)) => :PM_roll3,
    :NO2   => (x -> roll_mean(x, 3)) => :NO2_roll3,
    :pluie => jours_sans_pluie_vec   => :jours_sans_pluie)
train.vent_x_visibilite = train.vitesse_vent_moy .* train.visibilite_moy
train.mois_sin = sin.(2π .* train.mois ./ 12)
train.mois_cos = cos.(2π .* train.mois ./ 12)
train.temp_x_mois_sin = train.temp_moy .* train.mois_sin
train.temp_x_mois_cos = train.temp_moy .* train.mois_cos

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
train.is_winter = in.(train.mois, Ref(WINTER_MONTHS))
train.winter_tmin_inter = ifelse.(train.is_winter, train.temp_min_roll3, 0.0)
train.winter_vent_inter = ifelse.(train.is_winter, train.vent_min_roll3, 0.0)

# Label de régime (binaire)
train.is_fumee = train.PM .> FUMEE_THRESHOLD

# -----------------------------------------------------------------------------
# 2. Formules
# -----------------------------------------------------------------------------
# Classifieur : utilise features exogènes + PM_lag (pas PM brute)
formula_clf = @formula(is_fumee ~ PM_roll3 + PM_lag1 + NO2_roll3 +
                                   visibilite_min_roll3 + temp_max_roll7 +
                                   deficit_pluie14 + is_fire_season +
                                   fire_vis_inter + fire_tmax_inter +
                                   hum_rel_moy + vitesse_vent_moy +
                                   mois_sin + mois_cos + is_winter)

# GLM normal : utilise PM_lag1 / PM_roll3 (stable car PM reste bas)
formula_glm_normal = @formula(PM ~ PM_roll3 + PM_lag1 + NO2_roll3 +
                            vitesse_vent_moy + visibilite_moy + hum_rel_moy +
                            temp_moy + mois + vent_x_visibilite + jours_sans_pluie +
                            mois_sin + mois_cos + temp_x_mois_sin + temp_x_mois_cos +
                            visibilite_min_roll3 + temp_max_roll7 + deficit_pluie14 +
                            is_fire_season + fire_vis_inter + fire_tmax_inter +
                            is_winter + temp_min_roll3 + vent_min_roll3 +
                            winter_tmin_inter + winter_vent_inter)

# GLM fumée : SANS PM_lag1/PM_roll3 (sinon feedback exponentiel en récursif)
formula_glm_fumee = @formula(PM ~ NO2_roll3 +
                            vitesse_vent_moy + visibilite_moy + hum_rel_moy +
                            temp_moy + mois + vent_x_visibilite + jours_sans_pluie +
                            mois_sin + mois_cos + temp_x_mois_sin + temp_x_mois_cos +
                            visibilite_min_roll3 + temp_max_roll7 + deficit_pluie14 +
                            is_fire_season + fire_vis_inter + fire_tmax_inter +
                            is_winter + temp_min_roll3 + vent_min_roll3 +
                            winter_tmin_inter + winter_vent_inter)

const PM_CAP = 500.0  # Plafond absolu (max observé en train = 484)

COLS_ALL = [:PM, :is_fumee, :PM_lag1, :PM_roll3, :NO2_roll3,
            :vitesse_vent_moy, :visibilite_moy, :hum_rel_moy,
            :temp_moy, :mois, :vent_x_visibilite, :jours_sans_pluie,
            :mois_sin, :mois_cos, :temp_x_mois_sin, :temp_x_mois_cos,
            :visibilite_min_roll3, :temp_max_roll7, :deficit_pluie14,
            :is_fire_season, :fire_vis_inter, :fire_tmax_inter,
            :is_winter, :temp_min_roll3, :vent_min_roll3,
            :winter_tmin_inter, :winter_vent_inter]

# -----------------------------------------------------------------------------
# 3. Fit des 3 modèles sur un sous-ensemble + prédiction récursive gating
# -----------------------------------------------------------------------------
"""
    fit_gating(train_data) → (clf, glm_normal, glm_fumee)
"""
function fit_gating(train_data::DataFrame)
    td = dropmissing(train_data, COLS_ALL)
    clf = glm(formula_clf, td, Binomial(), LogitLink())

    td_normal = filter(r -> !r.is_fumee, td)
    td_fumee  = filter(r ->  r.is_fumee, td)
    println("  Fit GLM normal : n=$(nrow(td_normal))")
    println("  Fit GLM fumée  : n=$(nrow(td_fumee))")

    glm_normal = lm(formula_glm_normal, td_normal)
    glm_fumee  = lm(formula_glm_fumee,  td_fumee)
    (clf, glm_normal, glm_fumee)
end

"""
    recursive_predict_gating(clf, glm_n, glm_f, target_df, history_df, train_med)
"""
function recursive_predict_gating(clf, glm_n, glm_f,
                                  target_df::DataFrame, history_df::DataFrame,
                                  train_med::Float64)
    n = nrow(target_df)
    preds = zeros(Float64, n)
    p_fumee_all = zeros(Float64, n)
    history_pm = Dict{Any, Vector{Float64}}()
    for sdf in groupby(sort(history_df, [:stationId, :Date]), :stationId)
        sid = first(sdf.stationId)
        history_pm[sid] = [Float64(r.PM) for r in eachrow(sdf) if !ismissing(r.PM)]
    end
    idx_sorted = sortperm(collect(zip(target_df.Date, target_df.stationId)))
    for i in idx_sorted
        row = target_df[i, :]
        sid = row.stationId
        hist = get!(history_pm, sid, Float64[])
        pm_lag1  = isempty(hist) ? train_med : hist[end]
        pm_roll3 = if length(hist) >= 3; mean(hist[end-2:end])
                   elseif !isempty(hist); mean(hist); else; train_med; end
        row_df = DataFrame(target_df[i:i, :])
        row_df.PM_lag1  = [Float64(pm_lag1)]
        row_df.PM_roll3 = [Float64(pm_roll3)]
        p = GLM.predict(clf, row_df)[1]
        yn = GLM.predict(glm_n, row_df)[1]
        yf = GLM.predict(glm_f, row_df)[1]
        yhat = clamp(p * yf + (1 - p) * yn, 0.0, PM_CAP)
        preds[i] = yhat
        p_fumee_all[i] = p
        push!(hist, yhat)
    end
    preds, p_fumee_all
end

function eval_holdout(year_holdout::Int)
    println("\n── Évaluation holdout $year_holdout ──")
    tr = filter(r -> year(r.Date) < year_holdout, train)
    vl = filter(r -> year(r.Date) == year_holdout, train)
    for col in [:NO2_roll3, :visibilite_min_roll3, :temp_max_roll7,
                :temp_min_roll3, :vent_min_roll3]
        med = median(skipmissing(tr[:, col]))
        vl[!, col] = Float64.(coalesce.(vl[:, col], med))
    end
    clf, gn, gf = fit_gating(tr)
    preds, p_fumee = recursive_predict_gating(clf, gn, gf, vl, tr, median(tr.PM))
    actual = Float64.(vl.PM)
    mask = .!ismissing.(actual)
    rmse_g = sqrt(mean((actual[mask] .- preds[mask]) .^ 2))
    mae_g  = mean(abs.(actual[mask] .- preds[mask]))
    is_fum = actual[mask] .> FUMEE_THRESHOLD
    rmse_n = sqrt(mean((actual[mask][.!is_fum] .- preds[mask][.!is_fum]) .^ 2))
    rmse_f = sum(is_fum) > 0 ?
        sqrt(mean((actual[mask][is_fum] .- preds[mask][is_fum]) .^ 2)) : NaN
    # Stats classifieur
    pred_fumee = preds[mask] .> FUMEE_THRESHOLD
    p_fumee_mean_on_fumee = sum(is_fum) > 0 ? mean(p_fumee[mask][is_fum]) : NaN
    p_fumee_mean_on_normal = mean(p_fumee[mask][.!is_fum])
    println("  RMSE global       = $(round(rmse_g, digits=3))")
    println("  MAE               = $(round(mae_g, digits=3))")
    println("  RMSE régime normal (PM ≤ 35) = $(round(rmse_n, digits=3)) ($(sum(.!is_fum)) obs)")
    println("  RMSE régime fumée  (PM > 35) = $(round(rmse_f, digits=3)) ($(sum(is_fum)) obs)")
    println("  Max prédit = $(round(maximum(preds), digits=1)) vs réel = $(round(maximum(actual[mask]), digits=1))")
    println("  P(fumée) moyen | réel fumée  = $(round(p_fumee_mean_on_fumee, digits=3))")
    println("  P(fumée) moyen | réel normal = $(round(p_fumee_mean_on_normal, digits=3))")
    rmse_g
end

println("\n=== 2. CV holdout 2023 et 2024 ===")
rmse_2024 = eval_holdout(2024)
rmse_2023 = eval_holdout(2023)

# -----------------------------------------------------------------------------
# 4. Fit final sur 2007-2024 complet + prédiction 2025
# -----------------------------------------------------------------------------
println("\n=== 3. Fit final sur 2007-2024 ===")
clf_f, glm_n_f, glm_f_f = fit_gating(train)

# Préparation test
println("\n=== 4. Préparation test ===")
test_air   = CSV.read(TEST_AIR,   DataFrame)
test_meteo = CSV.read(TEST_METEO, DataFrame)
test = outerjoin(test_air, test_meteo[:, 3:end], on=:Date)
test = filter(row -> !ismissing(row.stationId), test)
test = treat_so2_missing_values(test, SO2_STATIONS)
sort!(test, [:stationId, :Date])
test.mois = month.(test.Date)
test.vent_x_visibilite = test.vitesse_vent_moy .* test.visibilite_moy
test.mois_sin = sin.(2π .* test.mois ./ 12)
test.mois_cos = cos.(2π .* test.mois ./ 12)
test.temp_x_mois_sin = test.temp_moy .* test.mois_sin
test.temp_x_mois_cos = test.temp_moy .* test.mois_cos

test_date_min = minimum(test.Date)
train_tail = combine(groupby(sort(train, [:stationId, :Date]), :stationId)) do sdf
    last(sort(dropmissing(sdf, :PM), :Date), min(7, nrow(sdf)))
end
train_tail_slim = select(train_tail, [:stationId, :Date, :NO2, :pluie])
test_input = DataFrame(stationId=test.stationId, Date=test.Date, NO2=test.NO2, pluie=test.pluie)
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
    [:visibilite_min => (v -> rolling_mean(v, 3, 2)) => :visibilite_min_roll3,
     :temp_max       => (v -> rolling_mean(v, 7, 4)) => :temp_max_roll7,
     :pluie          => (v -> rolling_sum(v, 14, 7)) => :pluie_cum14],
    [:visibilite_min_roll3, :temp_max_roll7, :pluie_cum14])
for col in [:visibilite_min_roll3, :temp_max_roll7, :pluie_cum14]
    med = median(skipmissing(train[:, col]))
    test[!, col] = Float64.(coalesce.(test[:, col], med))
end
test.is_fire_season = in.(test.mois, Ref(FIRE_MONTHS))
test.deficit_pluie14 = Int.(test.pluie_cum14 .< Q25_PLUIE14)
test.fire_vis_inter  = ifelse.(test.is_fire_season, test.visibilite_min_roll3, 0.0)
test.fire_tmax_inter = ifelse.(test.is_fire_season, test.temp_max_roll7, 0.0)

test = add_rolling_test!(test, train,
    [:stationId, :Date, :temp_min, :vitesse_vent_min],
    [:temp_min         => (v -> rolling_mean(v, 3, 2)) => :temp_min_roll3,
     :vitesse_vent_min => (v -> rolling_mean(v, 3, 2)) => :vent_min_roll3],
    [:temp_min_roll3, :vent_min_roll3])
for col in [:temp_min_roll3, :vent_min_roll3]
    med = median(skipmissing(train[:, col]))
    test[!, col] = Float64.(coalesce.(test[:, col], med))
end
test.is_winter = in.(test.mois, Ref(WINTER_MONTHS))
test.winter_tmin_inter = ifelse.(test.is_winter, test.temp_min_roll3, 0.0)
test.winter_vent_inter = ifelse.(test.is_winter, test.vent_min_roll3, 0.0)
test.PM_lag1  = zeros(Float64, nrow(test))
test.PM_roll3 = zeros(Float64, nrow(test))

println("\n=== 5. Prédiction récursive gating 2025 ===")
preds_2025, p_fumee_2025 = recursive_predict_gating(clf_f, glm_n_f, glm_f_f,
                                                    test, train, median(train.PM))
output = DataFrame(
    ID = [(test.Date[i], test.stationId[i]) for i in 1:nrow(test)],
    PM = preds_2025)
CSV.write(OUT_PATH, output)
println("→ $OUT_PATH ($(nrow(output)) lignes)")
println("  Moyenne PM prédit : $(round(mean(preds_2025), digits=3))")
println("  Max PM prédit     : $(round(maximum(preds_2025), digits=3))")
println("  P(fumée) moyen    : $(round(mean(p_fumee_2025), digits=3))")
println("  Jours avec P(fumée)>0.5 : $(sum(p_fumee_2025 .> 0.5))")

println("\n\n=== RÉSUMÉ GATING ===")
println("                       | 2024 CV | 2023 CV | 2025 Kaggle")
println("  nolag                |   9.36  |  24.34  |  18.25")
println("  recursive+log        |   9.06  |  25.91  |  ?")
println("  gating (3 modèles)   |  $(rpad(round(rmse_2024, digits=2),4))   |  $(rpad(round(rmse_2023, digits=2),4))   |  à tester")

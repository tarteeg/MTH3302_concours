# =============================================================================
# model_weighted.jl — GLM Gaussian(Log) avec pondération temporelle
#
# Idée : la pollution a baissé depuis 2007 (véhicules, normes). Les années
# récentes sont plus représentatives de 2025. On pondère :
#   wt(annee) = exp(-(2024 - annee) / decay)
#
# On teste plusieurs decay (3, 5, 10 ans) et choisit celui qui minimise
# le RMSE CV sur 2024 (année la plus proche de 2025).
# =============================================================================

using CSV, DataFrames, Dates, GLM, Statistics, LinearAlgebra

const TRAIN_AIR   = "data/qualite-de-lair_train.csv"
const TRAIN_METEO = "data/meteo_train.csv"
const TEST_AIR    = "data/qualite-de-lair_test.csv"
const TEST_METEO  = "data/meteo_test.csv"
const OUT_PATH    = "benchmark_predictions_weighted.csv"

const SO2_STATIONS   = ["Saint-Jean-Baptiste ", "Saint-Dominique",
                         "Saint-Joseph", "Anjou"]
const FIRE_TAIL_DAYS = 14
const FIRE_MONTHS    = [5, 6, 7, 8, 9]
const WINTER_MONTHS  = [12, 1, 2]

# --- Helpers (identiques) ---
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
    T = Union{eltype(v), Missing}; out = Vector{T}(missing, length(v))
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

# --- Preprocess (identique) ---
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
    sdf.NO2 = interpolate(sdf.NO2); sdf.pluie = interpolate(sdf.pluie)
    sdf.neige = interpolate(sdf.neige); sdf.precip_tot = interpolate(sdf.precip_tot)
    sdf.O3 = interpolate(sdf.O3)
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
train.PM_pos = max.(Float64.(train.PM), 0.5)
train.annee = year.(train.Date)

formula_full = @formula(PM_pos ~ PM_roll3 + PM_lag1 + NO2_roll3 +
                                  vitesse_vent_moy + visibilite_moy + hum_rel_moy +
                                  temp_moy + mois + vent_x_visibilite + jours_sans_pluie +
                                  mois_sin + mois_cos + temp_x_mois_sin + temp_x_mois_cos +
                                  visibilite_min_roll3 + temp_max_roll7 + deficit_pluie14 +
                                  is_fire_season + fire_vis_inter + fire_tmax_inter +
                                  is_winter + temp_min_roll3 + vent_min_roll3 +
                                  winter_tmin_inter + winter_vent_inter)

COLS_NEEDED = [:PM_pos, :PM_lag1, :PM_roll3, :NO2_roll3,
               :vitesse_vent_moy, :visibilite_moy, :hum_rel_moy,
               :temp_moy, :mois, :vent_x_visibilite, :jours_sans_pluie,
               :mois_sin, :mois_cos, :temp_x_mois_sin, :temp_x_mois_cos,
               :visibilite_min_roll3, :temp_max_roll7, :deficit_pluie14,
               :is_fire_season, :fire_vis_inter, :fire_tmax_inter,
               :is_winter, :temp_min_roll3, :vent_min_roll3,
               :winter_tmin_inter, :winter_vent_inter, :annee]

# --- Prédiction récursive ---
function recursive_predict_glm(model, target_df::DataFrame, history_df::DataFrame,
                               train_med::Float64; cap::Float64=150.0)
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
        yhat = clamp(GLM.predict(model, row_df)[1], 0.1, cap)
        preds[i] = yhat
        push!(hist, yhat)
    end
    preds
end

# --- Évaluation avec pondération ---
function eval_weighted(decay, year_holdout::Int; exclude_2023=true, ref_year=2024)
    tr = filter(r -> year(r.Date) < year_holdout, train)
    vl = filter(r -> year(r.Date) == year_holdout, train)
    tr_clean = dropmissing(tr, COLS_NEEDED)
    if exclude_2023 && year_holdout != 2023
        tr_clean = filter(r -> r.annee != 2023, tr_clean)
    end
    # Poids temporels
    if decay === nothing
        tr_clean.wt = ones(Float64, nrow(tr_clean))
    else
        tr_clean.wt = exp.(-(ref_year .- tr_clean.annee) ./ decay)
    end
    # Normaliser pour que somme des poids = N (évite de changer l'échelle de la vraisemblance)
    tr_clean.wt = tr_clean.wt .* (nrow(tr_clean) / sum(tr_clean.wt))
    model = glm(formula_full, tr_clean, Normal(), LogLink(); wts=tr_clean.wt)
    preds = recursive_predict_glm(model, vl, tr_clean, median(tr_clean.PM_pos))
    actual = Float64.(vl.PM)
    mask = .!ismissing.(actual)
    rmse_g = sqrt(mean((actual[mask] .- preds[mask]) .^ 2))
    is_fum = actual[mask] .> 35
    rmse_n = sqrt(mean((actual[mask][.!is_fum] .- preds[mask][.!is_fum]) .^ 2))
    rmse_f = sum(is_fum) > 0 ?
        sqrt(mean((actual[mask][is_fum] .- preds[mask][is_fum]) .^ 2)) : NaN
    rmse_g, rmse_n, rmse_f, maximum(preds), model
end

# --- Benchmark des decays ---
println("\n=== 2. Grille des decays (holdout 2024) ===")
decay_grid = [nothing, 20.0, 10.0, 7.0, 5.0, 3.0, 2.0]
best_decay = nothing
best_rmse = Inf
for d in decay_grid
    rmse, rn, rf, mp, _ = eval_weighted(d, 2024)
    tag = d === nothing ? "uniforme" : "decay=$d ans"
    println("  $tag : RMSE=$(round(rmse,digits=3)) | normal=$(round(rn,digits=3)) | fumée=$(round(rf,digits=3)) | max=$(round(mp,digits=1))")
    if rmse < best_rmse
        global best_rmse = rmse
        global best_decay = d
    end
end
bd_tag = best_decay === nothing ? "uniforme" : "$best_decay ans"
println("\n→ Meilleur decay : $bd_tag (RMSE=$(round(best_rmse,digits=3)))")

println("\n=== 3. Vérification sur holdout 2023 avec meilleur decay ===")
r23, rn23, rf23, mp23, _ = eval_weighted(best_decay, 2023; exclude_2023=false)
println("  RMSE 2023 = $(round(r23,digits=3)) | normal=$(round(rn23,digits=3)) | fumée=$(round(rf23,digits=3)) | max=$(round(mp23,digits=1))")

# --- Fit final avec le meilleur decay ---
println("\n=== 4. Fit final sur 2007-2024 sans 2023, decay=$bd_tag ===")
train_final = filter(r -> r.annee != 2023, dropmissing(train, COLS_NEEDED))
if best_decay === nothing
    train_final.wt = ones(Float64, nrow(train_final))
else
    train_final.wt = exp.(-(2024 .- train_final.annee) ./ best_decay)
end
train_final.wt = train_final.wt .* (nrow(train_final) / sum(train_final.wt))
model_final = glm(formula_full, train_final, Normal(), LogLink(); wts=train_final.wt)
println("n=$(nrow(train_final)), somme poids=$(round(sum(train_final.wt),digits=1))")

# --- Prépa test ---
println("\n=== 5. Préparation test ===")
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
test.PM_pos   = zeros(Float64, nrow(test))

println("\n=== 6. Prédiction récursive 2025 ===")
preds_2025 = recursive_predict_glm(model_final, test, train_final, median(train_final.PM_pos))
output = DataFrame(
    ID = [(test.Date[i], test.stationId[i]) for i in 1:nrow(test)],
    PM = preds_2025)
CSV.write(OUT_PATH, output)
println("→ $OUT_PATH ($(nrow(output)) lignes)")
println("  Moyenne PM prédit : $(round(mean(preds_2025), digits=3))")
println("  Max PM prédit     : $(round(maximum(preds_2025), digits=3))")

println("\n=== RÉSUMÉ ===")
println("                              | 2024 CV | 2023 CV")
println("  Gaussian(Log) sans 2023     |  9.26   |  24.67")
println("  Gaussian(Log) pondéré $bd_tag |  $(rpad(round(best_rmse,digits=2),4))   |  $(rpad(round(r23,digits=2),4))")

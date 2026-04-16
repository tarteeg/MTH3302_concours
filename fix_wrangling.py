import json

path = "/Users/tidianecisse/PROJET_INFO/MTH3302_concours/projet_h2026_experimentation.ipynb"
with open(path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for i, cell in enumerate(nb.get("cells", [])):
    source = "".join(cell.get("source", []))
    if "test_df = leftjoin(test_air, test_meteo[:, 3:end], on=:Date)" in source:
        new_source = []
        skip = False
        for line in cell["source"]:
            if "test_df = leftjoin(test_air, test_meteo[:, 3:end], on=:Date)" in line:
                new_source.append(line)
                new_source.extend([
                    "\n",
                    "    # =========================\n",
                    "    # Nettoyage TITANIUM: Destruction de tous les \"Missing\" ou \"\" cachés dans Kaggle\n",
                    "    # =========================\n",
                    "    for col in names(test_df)\n",
                    "        if col != \"Date\" && col != \"stationId\" && col != \"nom\" && col != \"is_aero\" && col != \"saison\" && col != \"mois\"\n",
                    "            med_val = 0.0\n",
                    "            if col in names(model_df)\n",
                    "                arr = collect(skipmissing(model_df[!, col]))\n",
                    "                if !isempty(arr)\n",
                    "                    med_val = Float64(median(arr))\n",
                    "                end\n",
                    "            end\n",
                    "            \n",
                    "            # Remplacement brutal ligne par ligne\n",
                    "            test_df[!, col] = [ismissing(x) || x === \"\" || string(x) == \"NA\" || string(x) == \"missing\" || typeof(x) <: AbstractString ? med_val : Float64(x) for x in test_df[!, col]]\n",
                    "        end\n",
                    "    end\n",
                    "\n"
                ])
                skip = True
            elif "test_df.is_aero = test_df.stationId .== 66" in line:
                skip = False
                new_source.append(line)
            elif not skip:
                new_source.append(line)
                
        # Fix the bug I accidentally introduced: `typeof(x) <: AbstractString ? med_val : Float64(x)`
        # If it's a string like "23.4", parsing it is better.
        fixed_source = []
        for s in new_source:
            if "typeof(x) <: AbstractString ? med_val : Float64(x)" in s:
                s = s.replace("typeof(x) <: AbstractString ? med_val : Float64(x)", "typeof(x) <: AbstractString ? (tryparse(Float64, x) === nothing ? med_val : parse(Float64, x)) : Float64(x)")
            fixed_source.append(s)
            
        cell["source"] = fixed_source

with open(path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Titanium wrangling injected.")

# app.py
# Streamlit GAP Simülasyonu (Gerçek ekip konumları entegre) + Trafo GeoJSON yükle + Trafo konumlarını export et

import math
import json
import numpy as np
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium
from ortools.sat.python import cp_model


# ----------------------------
# GERÇEK EKİP KONUMARI (SABİT)
# ----------------------------
TEAMS = [
    ("BEDAŞ Beyoğlu İşletme Müdürlüğü", 41.042942843441594, 28.98187509471993),
    ("BEDAŞ Beyazıt İşletme Müdürlüğü", 41.01255990927693, 28.962134641262114),
    ("BEDAŞ Bayrampaşa İşletme Müdürlüğü", 41.046302182999646, 28.910872668799808),
    ("BEDAŞ Bakırköy İşletme Müdürlüğü", 40.98605787570794, 28.89211399154593),
    ("BEDAŞ Başakşehir İşletme Müdürlüğü", 41.09662872610036, 28.789892375665104),
    ("BEDAŞ Beşyol İşletme Müdürlüğü", 41.02375414632992, 28.790824905276498),
    ("BEDAŞ Çağlayan İşletme Müdürlüğü", 41.07210166553191, 28.982043223356346),
    ("BEDAŞ Güngören İşletme Müdürlüğü", 41.02151226059597, 28.887805521157343),
    ("BEDAŞ Sefaköy İşletme Müdürlüğü", 40.99755768113406, 28.829276198411225),
    ("BEDAŞ Sarıyer İşletme Müdürlüğü", 41.032477772499725, 28.904751568799814),
]

DEFAULT_CENTER = (41.03, 28.90)  # İstanbul


# ----------------------------
# Yardımcılar
# ----------------------------
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def solve_gap(cost, demand, cap):
    """
    cost: (E x T) int
    demand: (T,) int
    cap: (E,) int
    return: assignment matrix X (E x T) 0/1, total_cost
    """
    E, T = cost.shape
    model = cp_model.CpModel()

    x = {}
    for e in range(E):
        for t in range(T):
            x[(e, t)] = model.NewBoolVar(f"x_{e}_{t}")

    # Her trafo tek ekibe atanır
    for t in range(T):
        model.Add(sum(x[(e, t)] for e in range(E)) == 1)

    # Kapasite kısıtları
    for e in range(E):
        model.Add(sum(int(demand[t]) * x[(e, t)] for t in range(T)) <= int(cap[e]))

    # Amaç
    model.Minimize(sum(int(cost[e, t]) * x[(e, t)] for e in range(E) for t in range(T)))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 5.0
    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None, None

    X = np.zeros((E, T), dtype=int)
    for e in range(E):
        for t in range(T):
            X[e, t] = int(solver.Value(x[(e, t)]))
    total = int(solver.ObjectiveValue())
    return X, total


def parse_geojson_points(geojson_obj):
    """
    GeoJSON FeatureCollection içinden Point (lon,lat) çıkarır.
    Dönen: DataFrame [id, name, lat, lon, props_json]
    """
    feats = geojson_obj.get("features", [])
    rows = []
    k = 0

    for f in feats:
        geom = f.get("geometry") or {}
        gtype = geom.get("type")
        coords = geom.get("coordinates")

        if gtype == "Point" and isinstance(coords, (list, tuple)) and len(coords) >= 2:
            lon, lat = coords[0], coords[1]
            props = f.get("properties") or {}
            name = (
                props.get("name")
                or props.get("Name")
                or props.get("TRAFO")
                or props.get("id")
                or props.get("ID")
                or f"Trafo_{k+1}"
            )
            rows.append(
                {
                    "id": k + 1,
                    "name": str(name),
                    "lat": float(lat),
                    "lon": float(lon),
                    "props_json": json.dumps(props, ensure_ascii=False),
                }
            )
            k += 1

    return pd.DataFrame(rows)


# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title="GAP Simülasyonu (Trafo GeoJSON + Gerçek Ekipler)", layout="wide")
st.title("GAP Simülasyonu – Trafo GeoJSON + Gerçek BEDAŞ Ekip Konumları")

with st.sidebar:
    st.header("1) Trafo Verisi")

    up = st.file_uploader("Trafo GeoJSON yükle (.geojson / .json)", type=["geojson", "json"])
    if up is None:
        st.warning("Devam etmek için trafo GeoJSON dosyası yükle.")
        st.stop()

    geojson_obj = json.loads(up.read().decode("utf-8", errors="ignore"))
    trafo_df = parse_geojson_points(geojson_obj)

    if trafo_df.empty:
        st.error("GeoJSON içinde Point tipinde trafo bulunamadı.")
        st.stop()

    st.success(f"Bulunan trafo sayısı: {len(trafo_df)}")

    st.header("2) Simülasyon Parametreleri")

    # Ekip sayısı: sabit 10 konum var, istersen ilk N ekibi kullan
    use_teams = st.slider("Kaç ekip kullanılsın? (ilk N ekip)", 1, len(TEAMS), len(TEAMS), 1)

    st.subheader("Kapasiteler (manuel)")
    caps = []
    for i in range(use_teams):
        caps.append(st.slider(f"Ekip {i+1} kapasite", 1, 200, 20, 1))
    caps = np.array(caps, dtype=int)

    st.subheader("Trafo iş yükü")
    demand_mode = st.selectbox("İş yükü modu", ["Sabit (1)", "Random (1-3)", "Random (1-5)"])
    if demand_mode == "Sabit (1)":
        demands = np.ones(len(trafo_df), dtype=int)
    elif demand_mode == "Random (1-3)":
        np.random.seed(7)
        demands = np.random.randint(1, 4, size=len(trafo_df), dtype=int)
    else:
        np.random.seed(7)
        demands = np.random.randint(1, 6, size=len(trafo_df), dtype=int)

    st.subheader("Maliyet ayarı")
    w_dist = st.slider("Mesafe ağırlığı", 1, 500, 80, 1)
    fixed_team_penalty = st.slider("Ekip sabit cezası (her atama için)", 0, 2000, 0, 10)

    st.header("3) Trafo konumlarını Export")
    export_name = st.text_input("Export dosya adı", "trafo_konumlari.csv")


# ----------------------------
# Verileri hazırla
# ----------------------------
teams_used = TEAMS[:use_teams]
team_names = [t[0] for t in teams_used]
team_coords = [(t[1], t[2]) for t in teams_used]  # (lat,lon)

trafos = list(zip(trafo_df["lat"].tolist(), trafo_df["lon"].tolist()))  # (lat,lon)

# Maliyet matrisi (mesafe tabanlı)
E, T = use_teams, len(trafos)
cost = np.zeros((E, T), dtype=int)
for e in range(E):
    for t in range(T):
        d = haversine_km(team_coords[e][0], team_coords[e][1], trafos[t][0], trafos[t][1])
        cost[e, t] = int(round(d * w_dist)) + int(fixed_team_penalty)

# GAP çöz
X, total_cost = solve_gap(cost, demands, caps)

# Export (trafo koordinatları)
export_df = trafo_df[["id", "name", "lat", "lon"]].copy()
export_df["demand"] = demands.astype(int)

csv_bytes = export_df.to_csv(index=False).encode("utf-8-sig")
st.sidebar.download_button(
    label="Trafo konumlarını indir (CSV)",
    data=csv_bytes,
    file_name=export_name if export_name.lower().endswith(".csv") else export_name + ".csv",
    mime="text/csv",
)


# ----------------------------
# Çıktılar + Harita
# ----------------------------
if X is None:
    st.error("Bu kapasite ayarlarıyla FEASIBLE atama bulunamadı (GAP çözülemedi).")
    st.stop()

col1, col2 = st.columns([1.15, 0.85], gap="large")

with col2:
    st.subheader("Özet")
    st.write(f"Trafo sayısı: **{T}**")
    st.write(f"Ekip sayısı: **{E}**")
    st.write(f"Toplam talep: **{int(demands.sum())}**")
    st.write(f"Toplam kapasite: **{int(caps.sum())}**")
    st.success(f"Toplam maliyet: **{int(total_cost)}**")

    # Atama tablosu
    rows = []
    for t in range(T):
        assigned_team = int(np.argmax(X[:, t]))
        rows.append(
            {
                "Trafo": trafo_df.loc[t, "name"],
                "Trafo_ID": int(trafo_df.loc[t, "id"]),
                "İşYükü": int(demands[t]),
                "Atanan Ekip": team_names[assigned_team],
                "Maliyet": int(cost[assigned_team, t]),
                "Trafo (lat,lon)": f"({trafos[t][0]:.6f},{trafos[t][1]:.6f})",
            }
        )
    df = pd.DataFrame(rows)
    st.subheader("Atama Tablosu")
    st.dataframe(df, use_container_width=True, height=420)

    # Ekip dolulukları
    used = (X * demands.reshape(1, -1)).sum(axis=1).astype(int)
    util = pd.DataFrame(
        {
            "Ekip": team_names,
            "Kullanılan": used,
            "Kapasite": caps,
            "Doluluk%": np.round(100 * used / np.maximum(caps, 1), 1),
        }
    )
    st.subheader("Kapasite Kullanımı")
    st.dataframe(util, use_container_width=True, height=260)

with col1:
    st.subheader("Harita")

    m = folium.Map(location=DEFAULT_CENTER, zoom_start=11, control_scale=True)

    palette = [
        "red", "blue", "green", "purple", "orange",
        "darkred", "cadetblue", "darkgreen", "darkblue", "pink"
    ]

    # Ekip ikonları
    for e in range(E):
        folium.Marker(
            location=(team_coords[e][0], team_coords[e][1]),
            tooltip=f"{team_names[e]} (Kapasite {int(caps[e])})",
            icon=folium.Icon(color=palette[e % len(palette)], icon="users", prefix="fa"),
        ).add_to(m)

    # Trafo ikonları + atama çizgileri
    for t in range(T):
        e = int(np.argmax(X[:, t]))
        color = palette[e % len(palette)]

        folium.Marker(
            location=(trafos[t][0], trafos[t][1]),
            tooltip=f"{trafo_df.loc[t, 'name']} | İşYükü={int(demands[t])} | {team_names[e]}",
            icon=folium.Icon(color=color, icon="bolt", prefix="fa"),
        ).add_to(m)

        folium.PolyLine(
            locations=[(team_coords[e][0], team_coords[e][1]), (trafos[t][0], trafos[t][1])],
            color=color,
            weight=2,
            opacity=0.7,
        ).add_to(m)

    st_folium(m, width=None, height=720)

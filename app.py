import os
import math
import random
import time
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from ortools.constraint_solver import routing_enums_pb2, pywrapcp
import folium
import streamlit.components.v1 as components

st.set_page_config(page_title="AI Talep Tahmini & MDVRP", layout="wide", page_icon="🚚")

REQUIRED_DATA = [
    "hazir_veri/backtest_sonuclari.csv",
    "hazir_veri/aile_bazli_wape.csv",
    "hazir_veri/ozellik_onemleri.csv",
    "hazir_veri/magaza_talep_ozeti.csv",
    "hazir_veri/son_30_gun_tahminler.csv",
]
if not all(os.path.exists(p) for p in REQUIRED_DATA):
    st.error("HATA: 'hazir_veri/' klasörü veya CSV dosyaları bulunamadı. Önce 'python veri_hazirla.py' komutunu çalıştırın.")
    st.stop()

@st.cache_data
def load_prepared_data():
    bt = pd.read_csv("hazir_veri/backtest_sonuclari.csv")
    fam = pd.read_csv("hazir_veri/aile_bazli_wape.csv")
    imp = pd.read_csv("hazir_veri/ozellik_onemleri.csv")
    magaza = pd.read_csv("hazir_veri/magaza_talep_ozeti.csv").sort_values("store_nbr").reset_index(drop=True)
    son30 = pd.read_csv("hazir_veri/son_30_gun_tahminler.csv")
    return bt, fam, imp, magaza, son30

bt_df, fam_df, imp_df, magaza_df, son30_df = load_prepared_data()

def haversine_distance_m(lat1, lon1, lat2, lon2):
    R = 6371000.0  # metre
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return int(round(R * c * 1.3))  # Yol sapma katsayisi alpha = 1.3 (metre)

# ==========================================
# GENETİK ALGORİTMA MODÜLÜ (METRE + MALİYET AMAÇLI + CEZA KONTROLLÜ)
# ==========================================
def run_genetic_algorithm(store_indices, depot_idx, dist_matrix_m, demands, capacity, max_dist_m, pop_size=40, gen_count=50):
    random.seed(42)
    population = [random.sample(list(store_indices), len(store_indices)) for _ in range(pop_size)]

    def calculate_fitness(ind):
        routes, curr_route, curr_load, curr_dist = [], [], 0, 0
        prev_node = depot_idx
        for store in ind:
            leg_dist = dist_matrix_m[prev_node][store]
            ret_dist = dist_matrix_m[store][depot_idx]

            if curr_load + demands[store] <= capacity and (curr_dist + leg_dist + ret_dist) <= max_dist_m:
                curr_route.append(store)
                curr_load += demands[store]
                curr_dist += leg_dist
                prev_node = store
            else:
                if curr_route:
                    routes.append(curr_route)
                curr_route = [store]
                curr_load = demands[store]
                curr_dist = dist_matrix_m[depot_idx][store]
                prev_node = store
        if curr_route:
            routes.append(curr_route)

        total_dist_m = 0
        penalty = 0.0
        for r in routes:
            if not r:
                continue
            r_dist = dist_matrix_m[depot_idx][r[0]]
            r_load = sum(demands[s] for s in r)
            for i in range(len(r) - 1):
                r_dist += dist_matrix_m[r[i]][r[i + 1]]
            r_dist += dist_matrix_m[r[-1]][depot_idx]

            if r_dist > max_dist_m:
                penalty += ((r_dist - max_dist_m) / 1000.0) * 1000.0
            if r_load > capacity:
                penalty += (r_load - capacity) * 1000.0
            total_dist_m += r_dist

        # Ortak amaç fonksiyonu: Yakıt maliyeti (1.25 USD/km) + Araç sabit çıkış maliyeti (45 USD/araç)
        total_cost = (total_dist_m / 1000.0) * 1.25 + len(routes) * 45.0
        return total_cost + penalty, total_dist_m, routes, penalty

    best_ind, best_obj, best_dist_m, best_routes, best_penalty = None, float("inf"), 0, [], float("inf")
    for _ in range(gen_count):
        pop_fitness = []
        for ind in population:
            obj, dist_m, routes, pen = calculate_fitness(ind)
            pop_fitness.append((obj, ind, dist_m, routes, pen))
            if obj < best_obj:
                best_obj, best_ind, best_dist_m, best_routes, best_penalty = obj, ind, dist_m, routes, pen

        new_pop = [best_ind]
        t_size = min(3, len(pop_fitness))
        while len(new_pop) < pop_size:
            p1 = min(random.sample(pop_fitness, t_size), key=lambda x: x[0])[1]
            p2 = min(random.sample(pop_fitness, t_size), key=lambda x: x[0])[1]
            if len(p1) >= 2:
                start, end = sorted(random.sample(range(len(p1)), 2))
                child = [None] * len(p1)
                child[start:end] = p1[start:end]
                p2_filtered = [x for x in p2 if x not in child]
                idx = 0
                for i in range(len(child)):
                    if child[i] is None:
                        child[i] = p2_filtered[idx]
                        idx += 1
                if random.random() < 0.15:
                    m1, m2 = random.sample(range(len(child)), 2)
                    child[m1], child[m2] = child[m2], child[m1]
            else:
                child = list(p1)
            new_pop.append(child)
        population = new_pop

    return best_dist_m, best_routes, best_penalty

# ==========================================
# KAPASİTELİ DEPO ATAMA VE ORTAK MDVRP HATTI
# ==========================================
def kapasiteli_atama(dist_m, k, demands, depo_kap):
    kalan = [depo_kap] * k
    atama = {d: [] for d in range(k)}
    for s in sorted(range(k, len(demands)), key=lambda i: -demands[i]):
        for d in sorted(range(k), key=lambda d: dist_m[s][d]):
            if kalan[d] >= demands[s]:
                atama[d].append(s)
                kalan[d] -= demands[s]
                break
        else:
            return None  # bu K uygulanabilir değil
    return atama

def build_ga_outputs(atama, dist_matrix_m, demands_full, all_coords, k, arac_kap, max_mesafe_m, depo_maliyet, magaza_df_ref):
    t0 = time.time()
    ga_dist_m, ga_veh = 0, 0
    ga_routes_table, ga_folium_routes, ga_store_indices = [], [], []
    feasible = True
    total_penalty = 0.0

    for d_idx in range(k):
        stores = atama.get(d_idx, [])
        if not stores:
            continue
        b_dist_m, b_routes, b_penalty = run_genetic_algorithm(
            stores, d_idx, dist_matrix_m, demands_full, arac_kap, max_mesafe_m
        )
        total_penalty += b_penalty
        if b_penalty > 0:
            feasible = False

        for r in b_routes:
            if not r:
                continue
            r_coords = [all_coords[d_idx]]
            r_str = f"Depo {d_idx + 1}"
            r_load = 0
            r_dist_v_m = 0
            r_stores = []
            prev_n = d_idx
            for s_node in r:
                leg_m = dist_matrix_m[prev_n][s_node]
                r_dist_v_m += leg_m
                r_load += demands_full[s_node]
                store_real_nbr = int(magaza_df_ref.loc[s_node - k, "store_nbr"])
                r_str += f" ➔ M{store_real_nbr} [{leg_m / 1000.0:.1f}km]"
                r_coords.append(all_coords[s_node])
                r_stores.append(s_node - k)
                prev_n = s_node
            ret_m = dist_matrix_m[prev_n][d_idx]
            r_dist_v_m += ret_m
            r_coords.append(all_coords[d_idx])

            if r_dist_v_m > max_mesafe_m or r_load > arac_kap:
                feasible = False

            ga_dist_m += r_dist_v_m
            ga_veh += 1
            ga_folium_routes.append(r_coords)
            ga_store_indices.append(r_stores)
            ga_routes_table.append({
                "Araç": f"Araç {ga_veh}",
                "Güzergah": r_str + f" ➔ Dönüş [{ret_m / 1000.0:.1f}km]",
                "Mesafe (km)": round(r_dist_v_m / 1000.0, 2),
                "Yük (br)": r_load,
                "Doluluk (%)": round(r_load / arac_kap * 100, 1),
            })

    ga_time = time.time() - t0
    ga_dist_km = ga_dist_m / 1000.0
    ga_lojistik = ga_dist_km * 1.25 + ga_veh * 45.0
    ga_tesis = k * depo_maliyet
    ga_toplam = ga_lojistik + ga_tesis

    return {
        "feasible": feasible,
        "penalty": total_penalty,
        "dist_m": ga_dist_m,
        "dist_km": ga_dist_km,
        "veh": ga_veh,
        "lojistik_maliyet": ga_lojistik,
        "tesis_maliyet": ga_tesis,
        "toplam_maliyet": ga_toplam,
        "routes_table": ga_routes_table,
        "folium_routes_data": ga_folium_routes,
        "routes_store_indices": ga_store_indices,
        "time": ga_time,
    }

def solve_mdvrp_pipeline(
    store_coords,
    guvenlikli_talepler,
    k,
    depo_kap,
    arac_kap,
    max_mesafe_km,
    depo_maliyet,
    magaza_df_ref,
    seed=42,
    time_limit_sec=2,
):
    max_mesafe_m = int(max_mesafe_km * 1000)

    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    km.fit(store_coords, sample_weight=np.array(guvenlikli_talepler))
    depot_coords = km.cluster_centers_
    all_coords = np.vstack((depot_coords, store_coords))
    total_nodes = len(all_coords)

    distance_matrix_m = np.zeros((total_nodes, total_nodes), dtype=int)
    for i in range(total_nodes):
        for j in range(total_nodes):
            if i != j:
                distance_matrix_m[i][j] = haversine_distance_m(
                    all_coords[i][0], all_coords[i][1], all_coords[j][0], all_coords[j][1]
                )

    demands_full = [0] * k + list(guvenlikli_talepler)
    atama = kapasiteli_atama(distance_matrix_m, k, demands_full, depo_kap)

    base_info = {
        "depot_coords": depot_coords,
        "all_coords": all_coords,
        "distance_matrix_m": distance_matrix_m,
        "demands_full": demands_full,
        "atama": atama,
    }

    if atama is None or max(guvenlikli_talepler) > arac_kap:
        return {
            **base_info,
            "feasible": False,
            "reason": "Kapasite aşımı",
        }

    # 1. GENETİK ALGORİTMA (Aynı atama + metre + maliyet amaçlı)
    ga_res = build_ga_outputs(
        atama=atama,
        dist_matrix_m=distance_matrix_m,
        demands_full=demands_full,
        all_coords=all_coords,
        k=k,
        arac_kap=arac_kap,
        max_mesafe_m=max_mesafe_m,
        depo_maliyet=depo_maliyet,
        magaza_df_ref=magaza_df_ref,
    )

    # 2. GOOGLE OR-TOOLS (Aynı atama + sent maliyet + 45 USD sabit araç maliyeti + metre mesafe)
    start_time = time.time()
    max_veh_per_depot = max(2, int(math.ceil(depo_kap / arac_kap)) + 2)
    num_vehicles = k * max_veh_per_depot
    starts = [int(d) for d in range(k) for _ in range(max_veh_per_depot)]

    manager = pywrapcp.RoutingIndexManager(total_nodes, num_vehicles, starts, starts)
    routing = pywrapcp.RoutingModel(manager)

    # Amaç fonksiyonu callback'i: SENT cinsinden (1 m = 0.125 cent => 1 km = 125 cent = 1.25 USD)
    def cost_cent_cb(f, t):
        d_m = distance_matrix_m[manager.IndexToNode(f)][manager.IndexToNode(t)]
        return int(round(d_m * 0.125))

    cost_idx = routing.RegisterTransitCallback(cost_cent_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(cost_idx)
    routing.SetFixedCostOfAllVehicles(4500)  # Araç başı 45.00 USD = 4500 sent

    # Mesafe kısıtı callback'i: METRE cinsinden
    def dist_m_cb(f, t):
        return int(distance_matrix_m[manager.IndexToNode(f)][manager.IndexToNode(t)])

    dist_idx = routing.RegisterTransitCallback(dist_m_cb)
    routing.AddDimension(dist_idx, 0, max_mesafe_m, True, "Distance")

    # Kapasite kısıtı callback'i: BİRİM (br) cinsinden
    def dem_cb(f):
        return int(demands_full[manager.IndexToNode(f)])

    dem_idx = routing.RegisterUnaryTransitCallback(dem_cb)
    routing.AddDimensionWithVehicleCapacity(dem_idx, 0, [arac_kap] * num_vehicles, True, "Capacity")

    # OR-Tools'u GA ile birebir aynı depo-mağaza atamasına (atama) bağla
    veh_of = {d: [int(v) for v in range(num_vehicles) if starts[v] == d] for d in range(k)}
    for d, stores in atama.items():
        allowed_vehs = veh_of[d]
        for s in stores:
            node_idx = int(manager.NodeToIndex(int(s)))
            try:
                routing.SetAllowedVehiclesForIndex(allowed_vehs, node_idx)
            except TypeError:
                routing.VehicleVar(node_idx).SetValues(allowed_vehs)

    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_params.time_limit.seconds = time_limit_sec

    solution = routing.SolveWithParameters(search_params)
    or_time = time.time() - start_time

    if not solution:
        or_res = {
            "feasible": False,
            "time": or_time,
            "reason": "OR-Tools zaman sınırında uygulanabilir çözüm bulamadı.",
        }
    else:
        or_dist_m = 0
        or_veh = 0
        routes_table, folium_routes_data, routes_store_indices = [], [], []

        for v_id in range(num_vehicles):
            index = routing.Start(v_id)
            route_coords = []
            route_str = f"Depo {manager.IndexToNode(index) + 1}"
            r_load, r_dist_v_m, r_stores = 0, 0, []
            ret_leg_m = 0
            while not routing.IsEnd(index):
                n_idx = manager.IndexToNode(index)
                route_coords.append(all_coords[n_idx])
                index = solution.Value(routing.NextVar(index))
                n_next = manager.IndexToNode(index)
                leg_m = int(distance_matrix_m[n_idx][n_next])
                r_dist_v_m += leg_m
                if n_next >= k:
                    store_real_nbr = int(magaza_df_ref.loc[n_next - k, "store_nbr"])
                    r_load += demands_full[n_next]
                    route_str += f" ➔ M{store_real_nbr} [{leg_m / 1000.0:.1f}km]"
                    r_stores.append(n_next - k)
                else:
                    ret_leg_m = leg_m
            route_coords.append(all_coords[manager.IndexToNode(index)])
            if len(route_coords) > 2:
                or_dist_m += r_dist_v_m
                or_veh += 1
                folium_routes_data.append(route_coords)
                routes_table.append({
                    "Araç": f"Araç {or_veh}",
                    "Güzergah": route_str + f" ➔ Dönüş [{ret_leg_m / 1000.0:.1f}km]",
                    "Mesafe (km)": round(r_dist_v_m / 1000.0, 2),
                    "Yük (br)": r_load,
                    "Doluluk (%)": round(r_load / arac_kap * 100, 1),
                })
                routes_store_indices.append(r_stores)

        or_dist_km = or_dist_m / 1000.0
        or_lojistik = or_dist_km * 1.25 + or_veh * 45.0
        or_tesis = k * depo_maliyet
        or_toplam = or_lojistik + or_tesis

        or_res = {
            "feasible": True,
            "dist_m": or_dist_m,
            "dist_km": or_dist_km,
            "veh": or_veh,
            "lojistik_maliyet": or_lojistik,
            "tesis_maliyet": or_tesis,
            "toplam_maliyet": or_toplam,
            "routes_table": routes_table,
            "folium_routes_data": folium_routes_data,
            "routes_store_indices": routes_store_indices,
            "time": or_time,
        }

    candidates = []
    if or_res["feasible"]:
        candidates.append(("Google OR-Tools", or_res))
    if ga_res["feasible"]:
        candidates.append(("Genetik Algoritma", ga_res))

    if not candidates:
        return {
            **base_info,
            "feasible": False,
            "reason": "Rota/mesafe kısıtı sağlanamadı (OR-Tools ve GA uygulanabilir çözüm üretemedi)",
            "or_res": or_res,
            "ga_res": ga_res,
        }

    winner_name, winner_res = min(candidates, key=lambda item: item[1]["toplam_maliyet"])

    return {
        **base_info,
        "feasible": True,
        "winner_name": winner_name,
        "winner": winner_res,
        "or_res": or_res,
        "ga_res": ga_res,
    }

# --- YAN MENÜ (GERÇEK FAVORITA ÖLÇEĞİNE UYARLANMIŞ SINIRLAR) ---
st.sidebar.markdown("### ⚙️ Tahmin & Filo Ayarları")
z_katsayisi = st.sidebar.slider("Emniyet Stoğu Z Katsayısı (1.64 ≈ %95 Hizmet):", 0.0, 2.58, 1.64, 0.01)
talep_baz = st.sidebar.selectbox("Planlama Talep Bazı:", ["Son 30 Günlük Ortalama Tahmin", "Son Gün (15.08.2017) Tahmini"])
arac_kapasitesi = int(st.sidebar.number_input("Araç Taşıma Kapasitesi (br):", 5000, 50000, 20000, 1000))
max_mesafe = int(st.sidebar.number_input("Araç Günlük Maks. Mesafe (km):", 200, 1500, 600, 50))

st.sidebar.markdown("---")
st.sidebar.markdown("### 🏢 Tesis Yerleşimi (MDVRP)")
depo_sayisi = st.sidebar.slider("Açılacak Depo Sayısı (K-Means):", 1, 8, 3, 1)
depo_kapasitesi = int(st.sidebar.number_input("Depo Hacmi (br):", 20000, 300000, 160000, 5000))
depo_acma_maliyeti = int(st.sidebar.number_input("Depo İşletme Maliyeti (USD):", 500, 10000, 2000, 500))

st.title("🚚 AI Destekli Gıda Tahmini ve Kapasiteli Rotalama (MDVRP)")
tab1, tab2 = st.tabs(["📊 1. Makine Öğrenmesi & Doğrulama", "🗺️ 2. Lojistik Ağı & Tesis Yerleşimi"])

# ==========================================
# TAB 1: MAKİNE ÖĞRENMESİ (GERÇEK FAVORITA SONUÇLARI)
# ==========================================
with tab1:
    st.info("📌 **Veri Kapsamı:** Bu sekmedeki sonuçlar Corporación Favorita veri setindeki 54 mağazanın 4 bozulabilir gıda grubu (PRODUCE, DAIRY, EGGS, MEATS) üzerinde gerçekleştirilen 4 katmanlı ilerleyen tahmin başlangıçlı doğrulama (rolling backtest) çıktılarını göstermektedir.")

    st.subheader("🔬 Zaman Serisi Çapraz Doğrulama Sonuçları (4 Katmanlı Rolling Backtest)")

    # En düşük MAE ortalamasına sahip modeli dinamik seç ve fark fold sapmasından küçükse "Eşit" yaz
    model_stats = [
        {
            "key": "rf",
            "name": "Random Forest",
            "pred_col": "rf_pred",
            "mae_mean": float(bt_df["RF_MAE"].mean()),
            "mae_std": float(bt_df["RF_MAE"].std()),
            "wape_mean": float(bt_df["RF_WAPE"].mean()),
            "wape_std": float(bt_df["RF_WAPE"].std()),
        },
        {
            "key": "xgb",
            "name": "XGBoost",
            "pred_col": "xgb_pred",
            "mae_mean": float(bt_df["XGB_MAE"].mean()),
            "mae_std": float(bt_df["XGB_MAE"].std()),
            "wape_mean": float(bt_df["XGB_WAPE"].mean()),
            "wape_std": float(bt_df["XGB_WAPE"].std()),
        },
        {
            "key": "naive",
            "name": "Sezonsal Naif",
            "pred_col": "naive_pred",
            "mae_mean": float(bt_df["Naif_MAE"].mean()),
            "mae_std": float(bt_df["Naif_MAE"].std()),
            "wape_mean": float(bt_df["Naif_WAPE"].mean()),
            "wape_std": float(bt_df["Naif_WAPE"].std()),
        },
    ]

    best_model = min(model_stats, key=lambda m: m["mae_mean"])
    sorted_models = sorted(model_stats, key=lambda m: m["mae_mean"])
    mae_diff = sorted_models[1]["mae_mean"] - sorted_models[0]["mae_mean"]
    is_stat_tie = mae_diff < best_model["mae_std"]

    for m_item in model_stats:
        if m_item["key"] == best_model["key"]:
            m_item["display_name"] = f"🏆 {m_item['name']} (Eşit)" if is_stat_tie else f"🏆 {m_item['name']}"
        elif is_stat_tie and m_item["key"] == sorted_models[1]["key"]:
            m_item["display_name"] = f"⚖️ {m_item['name']} (Eşit)"
        else:
            m_item["display_name"] = m_item["name"]

    c1, c2, c3 = st.columns(3)
    for col_box, m_item in zip([c1, c2, c3], model_stats):
        col_box.metric(
            m_item["display_name"],
            f"MAE: {m_item['mae_mean']:.2f}",
            f"WAPE: %{m_item['wape_mean']:.2f} (±{m_item['wape_std']:.2f})",
            "inverse",
        )

    with st.expander("📊 Tüm Backtest Katmanları ve Ürün Ailesi Analizleri", expanded=True):
        c_tab1, c_tab2 = st.columns([1.6, 1])
        c_tab1.dataframe(bt_df, hide_index=True, use_container_width=True)
        c_tab2.dataframe(fam_df, hide_index=True, use_container_width=True)

    st.markdown("---")
    col_chart1, col_chart2 = st.columns([2, 1])

    kazanan_ad = best_model["name"]
    kazanan_col = best_model["pred_col"] if best_model["pred_col"] in son30_df.columns else "rf_pred"

    with col_chart1:
        st.write(f"📈 **Ağ Geneli Gerçekleşen vs. {kazanan_ad} Tahmini (17 Temmuz – 15 Ağustos 2017)**")
        daily_agg = (
            son30_df.groupby("date", as_index=True)[["y_true", kazanan_col]]
            .sum()
            .rename(columns={"y_true": "Gerçek Satışlar", kazanan_col: f"{kazanan_ad} Tahmini"})
        )
        st.line_chart(daily_agg, color=["#1f77b4", "#ff7f0e"])

    ozellik_tr_map = {
        "lag_7": "Geçen Hafta Satışı (t-7)",
        "rolling_mean_7": "7 Günlük Hareketli Ortalama",
        "lag_14": "2 Hafta Önceki Satış (t-14)",
        "lag_28": "4 Hafta Önceki Satış (t-28)",
        "dayofweek": "Haftanın Günü",
        "store_nbr": "Mağaza Numarası",
        "dayofyear": "Yılın Günü",
        "is_weekend": "Hafta Sonu Durumu",
        "month": "Ay Bilgisi",
        "pay_day": "Maaş Günü (15 / Ay Sonu)",
        "is_holiday": "Resmi Tatil / Bayram",
    }

    with col_chart2:
        st.write(f"🎯 **{kazanan_ad} Özellik Önemi**")
        imp_tr_df = imp_df.copy()
        imp_tr_df["Ozellik_TR"] = imp_tr_df["Ozellik"].map(ozellik_tr_map).fillna(imp_tr_df["Ozellik"])
        feat_chart = imp_tr_df.set_index("Ozellik_TR")[["Onem"]].sort_values("Onem", ascending=True)
        st.bar_chart(feat_chart, horizontal=True, color="#2ca02c")

# ==========================================
# TAB 2: ARAÇ ROTALAMA (GERÇEK TAHMİN + EMNİYET STOĞU)
# ==========================================
with tab2:
    city_coords = {
        "Quito": (-0.1807, -78.4678), "Guayaquil": (-2.1962, -79.8862),
        "Santo Domingo": (-0.2530, -79.1754), "Cuenca": (-2.9001, -79.0059),
        "Manta": (-0.9677, -80.7127), "Latacunga": (-0.9316, -78.6158),
        "Machala": (-3.2581, -79.9554), "Ambato": (-1.2491, -78.6168),
        "Guaranda": (-1.5916, -79.0022), "Ibarra": (0.3517, -78.1223),
        "Babahoyo": (-1.8022, -79.5344), "Salinas": (-2.2145, -80.9515),
        "Daule": (-1.8622, -79.9777), "Loja": (-3.9931, -79.2042),
        "Playas": (-2.6285, -80.3800), "Puyo": (-1.4837, -78.0026),
        "Riobamba": (-1.6698, -78.6471), "Cayambe": (0.0384, -78.1510),
        "El Carmen": (-0.2741, -79.4604), "Quevedo": (-1.0286, -79.4635),
        "Libertad": (-2.2330, -80.9104), "Sangolqui": (-0.3344, -78.4503),
        "Duran": (-2.1701, -79.8220), "Esmeraldas": (0.9592, -79.6536),
    }

    np.random.seed(42)
    store_lats, store_lons = [], []
    for _, row in magaza_df.iterrows():
        base_lat, base_lon = city_coords[row["city"]]
        store_lats.append(base_lat + np.random.uniform(-0.02, 0.02))
        store_lons.append(base_lon + np.random.uniform(-0.02, 0.02))
    store_coords = np.column_stack((store_lats, store_lons))

    if talep_baz == "Son 30 Günlük Ortalama Tahmin":
        ham_talepler = magaza_df["ort_30g_tahmin"].tolist()
    else:
        ham_talepler = magaza_df["son_gun_tahmin"].tolist()

    hata_std_list = magaza_df["hata_std"].tolist()
    emniyet_stoklari = [z_katsayisi * std for std in hata_std_list]
    guvenlikli_talepler = [int(math.ceil(y + ss)) for y, ss in zip(ham_talepler, emniyet_stoklari)]
    demands_full = [0] * depo_sayisi + guvenlikli_talepler

    # U-Eğrisi Analizi (Sonuçlar st.session_state içinde korunur)
    with st.expander("📊 Optimum Depo Sayısı (K) ve Maliyet U-Eğrisi Analizi", expanded="u_curve" in st.session_state):
        st.write("Sistem K=1'den K=8'e kadar her senaryoyu talep ağırlıklı K-Means, kapasiteli depo ataması ve aynı atama üzerinde çalışan OR-Tools ile Genetik Algoritma çözücüleriyle 4 farklı başlangıç tohumunda (random_state) çalıştırır; en düşük maliyetli uygulanabilir çözümü baz alır.")
        u_seeds = [42, 100, 2026, 7]
        u_param_key = (z_katsayisi, talep_baz, arac_kapasitesi, max_mesafe, depo_kapasitesi, depo_acma_maliyeti)

        if st.button("U-Eğrisini Oluştur"):
            k_results = []
            prog_bar = st.progress(0)
            status_txt = st.empty()
            total_runs = 8 * len(u_seeds)
            step_cnt = 0

            for k_test in range(1, 9):
                seed_runs = []
                for sd in u_seeds:
                    step_cnt += 1
                    status_txt.text(f"K = {k_test}/8 | Tohum = {sd} ({step_cnt}/{total_runs}) çözülüyor...")
                    res_k = solve_mdvrp_pipeline(
                        store_coords=store_coords,
                        guvenlikli_talepler=guvenlikli_talepler,
                        k=k_test,
                        depo_kap=depo_kapasitesi,
                        arac_kap=arac_kapasitesi,
                        max_mesafe_km=max_mesafe,
                        depo_maliyet=depo_acma_maliyeti,
                        magaza_df_ref=magaza_df,
                        seed=sd,
                        time_limit_sec=2,
                    )
                    if res_k["feasible"]:
                        seed_runs.append(res_k)
                    prog_bar.progress(step_cnt / total_runs)

                if not seed_runs:
                    k_results.append({
                        "Depo Sayısı (K)": k_test,
                        "Durum": "❌ Uygunsuz",
                        "Ort. Mesafe (km)": None,
                        "Ort. Araç": None,
                        "Tesis Maliyeti (USD)": round(k_test * depo_acma_maliyeti, 2),
                        "Ort. Lojistik Maliyeti (USD)": None,
                        "Ort. Toplam Maliyet (USD)": None,
                        "Maliyet Std. Sapma (±USD)": None,
                        "Toplam Maliyet (Ort. ± Sapma)": "Uygunsuz",
                    })
                else:
                    tot_costs = [r["winner"]["toplam_maliyet"] for r in seed_runs]
                    log_costs = [r["winner"]["lojistik_maliyet"] for r in seed_runs]
                    dists = [r["winner"]["dist_km"] for r in seed_runs]
                    vehs = [r["winner"]["veh"] for r in seed_runs]
                    mean_tot = float(np.mean(tot_costs))
                    std_tot = float(np.std(tot_costs, ddof=1)) if len(tot_costs) > 1 else 0.0

                    k_results.append({
                        "Depo Sayısı (K)": k_test,
                        "Durum": f"✅ Uygun ({len(seed_runs)}/{len(u_seeds)})",
                        "Ort. Mesafe (km)": round(float(np.mean(dists)), 1),
                        "Ort. Araç": round(float(np.mean(vehs)), 1),
                        "Tesis Maliyeti (USD)": round(k_test * depo_acma_maliyeti, 2),
                        "Ort. Lojistik Maliyeti (USD)": round(float(np.mean(log_costs)), 2),
                        "Ort. Toplam Maliyet (USD)": round(mean_tot, 2),
                        "Maliyet Std. Sapma (±USD)": round(std_tot, 2),
                        "Toplam Maliyet (Ort. ± Sapma)": f"{mean_tot:,.2f} ± {std_tot:,.2f} USD",
                    })

            status_txt.empty()
            prog_bar.empty()
            st.session_state["u_curve"] = {
                "df": pd.DataFrame(k_results),
                "key": u_param_key,
            }

        if "u_curve" in st.session_state:
            if st.session_state["u_curve"]["key"] != u_param_key:
                st.warning("⚠️ Sol menüdeki parametreler değişti. Gösterilen U-eğrisi önceki ayarlara aittir; güncellemek için 'U-Eğrisini Oluştur' butonuna tekrar basın.")
            df_k = st.session_state["u_curve"]["df"]
            st.dataframe(df_k, hide_index=True, use_container_width=True)
            valid_k = df_k.dropna(subset=["Ort. Toplam Maliyet (USD)"])
            if not valid_k.empty:
                st.line_chart(
                    valid_k.set_index("Depo Sayısı (K)")[["Ort. Toplam Maliyet (USD)", "Tesis Maliyeti (USD)", "Ort. Lojistik Maliyeti (USD)"]],
                    color=["#ff0000", "#0000ff", "#00ff00"],
                )
                best_row = valid_k.loc[valid_k["Ort. Toplam Maliyet (USD)"].idxmin()]
                opt_k = int(best_row["Depo Sayısı (K)"])
                st.success(f"**Sonuç:** Kapasite ve mesafe kısıtlarını sağlayan senaryolar içinde toplam maliyeti en küçükleyen optimum depo sayısı **K = {opt_k}** ({best_row['Toplam Maliyet (Ort. ± Sapma)']}) olarak belirlenmiştir.")
            else:
                st.error("⚠️ Hiçbir depo sayısı mevcut kapasite/mesafe parametreleriyle uygulanabilir çözüm üretemedi.")

    st.markdown("---")

    # Talep Ağırlıklı K-Means Depo Konumlandırma ve Kapasiteli Atama (Ana Çözüm Hattı)
    kmeans = KMeans(n_clusters=depo_sayisi, random_state=42, n_init=10)
    kmeans.fit(store_coords, sample_weight=np.array(guvenlikli_talepler))
    depot_coords = kmeans.cluster_centers_
    all_coords = np.vstack((depot_coords, store_coords))
    total_nodes = len(all_coords)

    distance_matrix_m = np.zeros((total_nodes, total_nodes), dtype=int)
    for i in range(total_nodes):
        for j in range(total_nodes):
            if i != j:
                distance_matrix_m[i][j] = haversine_distance_m(all_coords[i][0], all_coords[i][1], all_coords[j][0], all_coords[j][1])

    atama_sonucu = kapasiteli_atama(distance_matrix_m, depo_sayisi, demands_full, depo_kapasitesi)
    optimizasyon_kilitli = False

    if atama_sonucu is None:
        optimizasyon_kilitli = True
        depot_store_map = {i: [] for i in range(depo_sayisi)}
        depot_loads = {i: 0 for i in range(depo_sayisi)}
        for s_idx in range(depo_sayisi, total_nodes):
            nearest = min(range(depo_sayisi), key=lambda d: distance_matrix_m[s_idx][d])
            depot_store_map[nearest].append(s_idx)
            depot_loads[nearest] += demands_full[s_idx]
    else:
        depot_store_map = atama_sonucu
        depot_loads = {d: sum(demands_full[s] for s in atama_sonucu[d]) for d in range(depo_sayisi)}

    st.subheader("🏢 Depo Başına Kapasite ve Yük Analizi (Ortak Atama)")
    st.caption(f"Toplam Ağ Günlük Planlanan Sevkiyatı (Tahmin + Z={z_katsayisi:.2f} Emniyet Stoğu): **{sum(guvenlikli_talepler):,} br** | En Yüksek Tekil Mağaza Talebi: **{max(guvenlikli_talepler):,} br**")

    depot_table_data = []
    for d in range(depo_sayisi):
        load = depot_loads[d]
        if load > depo_kapasitesi or atama_sonucu is None:
            status = "🔴 KAPASİTE AŞIMI" if load > depo_kapasitesi else "🟡 ATAMA YETERSİZ"
        else:
            status = "🟢 UYGUN"
        depot_table_data.append({
            "Depo": f"Depo {d+1}",
            "Atanan Mağaza Sayısı": len(depot_store_map[d]),
            "Atanan Yük (br)": load,
            "Kapasite (br)": depo_kapasitesi,
            "Doluluk (%)": round(load / depo_kapasitesi * 100, 1),
            "Durum": status,
        })
    st.dataframe(pd.DataFrame(depot_table_data), hide_index=True, use_container_width=True)

    if max(guvenlikli_talepler) > arac_kapasitesi:
        optimizasyon_kilitli = True
        st.error(f"⚠️ En yüksek mağaza talebi ({max(guvenlikli_talepler):,} br), araç kapasitesini ({arac_kapasitesi:,} br) aşıyor! Araç kapasitesini artırın.")
    elif optimizasyon_kilitli:
        st.error("⚠️ Depo kapasitesi aşımı! Depo sayısını veya depo hacmini artırın.")

    current_params = {
        "z_katsayisi": z_katsayisi,
        "talep_baz": talep_baz,
        "arac_kapasitesi": arac_kapasitesi,
        "max_mesafe": max_mesafe,
        "depo_sayisi": depo_sayisi,
        "depo_kapasitesi": depo_kapasitesi,
        "depo_acma_maliyeti": depo_acma_maliyeti,
    }

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🚀 Rotalama Yöntemlerini Çalıştır ve Karşılaştır", disabled=optimizasyon_kilitli, type="primary"):
        with st.spinner("Google OR-Tools ve Genetik Algoritma aynı depo ataması ve maliyet amaç fonksiyonu üzerinde çalıştırılıyor..."):
            res = solve_mdvrp_pipeline(
                store_coords=store_coords,
                guvenlikli_talepler=guvenlikli_talepler,
                k=depo_sayisi,
                depo_kap=depo_kapasitesi,
                arac_kap=arac_kapasitesi,
                max_mesafe_km=max_mesafe,
                depo_maliyet=depo_acma_maliyeti,
                magaza_df_ref=magaza_df,
                seed=42,
                time_limit_sec=2,
            )

            if not res["feasible"]:
                st.session_state["sonuc"] = {
                    "feasible": False,
                    "reason": res.get("reason", ""),
                    "or_res": res.get("or_res", {"feasible": False}),
                    "ga_res": res.get("ga_res", {"feasible": False}),
                    "params": current_params,
                }
            else:
                winner = res["winner"]
                best_veh = winner["veh"]
                routes_store_indices = winner["routes_store_indices"]

                # Gerçek 30 Gün Stres Testi (son30_df gerçek satışları üzerinden)
                pv = son30_df.pivot_table(index="date", columns="store_nbr", values="y_true", aggfunc="sum")
                nbr = magaza_df["store_nbr"].to_dict()
                toplam_gun = len(pv)
                tasan_gun = 0
                tasan_sefer = 0
                toplam_sefer = toplam_gun * max(1, best_veh)
                tasma_miktarlari = []

                arac_gunluk_yukler = {idx: [] for idx in range(len(routes_store_indices))}
                arac_asim_gunleri = {idx: 0 for idx in range(len(routes_store_indices))}

                for _, satir in pv.iterrows():
                    ihlal = False
                    for idx, rs in enumerate(routes_store_indices):
                        gercek_yuk = float(sum(satir.get(nbr[s], 0.0) for s in rs))
                        arac_gunluk_yukler[idx].append(gercek_yuk)
                        if gercek_yuk > arac_kapasitesi:
                            tasan_sefer += 1
                            arac_asim_gunleri[idx] += 1
                            tasma_miktarlari.append(gercek_yuk - arac_kapasitesi)
                            ihlal = True
                    tasan_gun += int(ihlal)

                stres_tablosu = []
                for idx, rs in enumerate(routes_store_indices):
                    plan_yuk = sum(guvenlikli_talepler[s] for s in rs)
                    yuk_serisi = arac_gunluk_yukler[idx]
                    ort_gercek = int(round(np.mean(yuk_serisi))) if yuk_serisi else 0
                    maks_gercek = int(round(np.max(yuk_serisi))) if yuk_serisi else 0
                    asim_say = arac_asim_gunleri[idx]
                    status = f"🔴 AŞIM ({asim_say}/{toplam_gun} gün)" if asim_say > 0 else "✅ GÜVENLİ"
                    stres_tablosu.append({
                        "Araç": f"Araç {idx+1}",
                        "Planlanan Yük (br)": plan_yuk,
                        "30g Ort. Gerçek (br)": ort_gercek,
                        "30g Maks. Gerçek (br)": maks_gercek,
                        "Araç Kapasitesi (br)": arac_kapasitesi,
                        "Durum": status,
                    })

                ihlal_orani = (tasan_sefer / toplam_sefer) * 100.0 if toplam_sefer else 0.0
                ort_tasma = float(np.mean(tasma_miktarlari)) if tasma_miktarlari else 0.0

                st.session_state["sonuc"] = {
                    "feasible": True,
                    "res": res,
                    "depot_coords": depot_coords,
                    "store_coords": store_coords,
                    "guvenlikli_talepler": guvenlikli_talepler,
                    "params": current_params,
                    "stres": {
                        "stres_tablosu": stres_tablosu,
                        "toplam_gun": toplam_gun,
                        "tasan_gun": tasan_gun,
                        "tasan_sefer": tasan_sefer,
                        "toplam_sefer": toplam_sefer,
                        "ihlal_orani": ihlal_orani,
                        "ort_tasma": ort_tasma,
                    },
                }

    # EKRANA YAZDIRMA (Buton dışından st.session_state["sonuc"] üzerinden okunur)
    if "sonuc" in st.session_state:
        sonuc = st.session_state["sonuc"]
        saved_p = sonuc["params"]

        if saved_p != current_params:
            st.warning("⚠️ Sol menüdeki parametreler değişti. Gösterilen rotalar ve maliyetler önceki ayarlara aittir; güncellemek için '🚀 Rotalama Yöntemlerini Çalıştır ve Karşılaştır' butonuna tekrar basın.")

        if not sonuc["feasible"]:
            st.error(f"⚠️ Çözücüler mevcut kısıtlarla uygulanabilir rota üretemedi: {sonuc.get('reason', '')}")
        else:
            res = sonuc["res"]
            or_res = res["or_res"]
            ga_res = res["ga_res"]
            winner_name = res["winner_name"]
            winner = res["winner"]
            saved_depot_coords = sonuc["depot_coords"]
            saved_store_coords = sonuc["store_coords"]
            saved_talepler = sonuc["guvenlikli_talepler"]
            saved_arac_kap = saved_p["arac_kapasitesi"]
            saved_depo_sayisi = saved_p["depo_sayisi"]
            saved_depo_maliyet = saved_p["depo_acma_maliyeti"]
            stres = sonuc["stres"]

            if not or_res["feasible"]:
                st.error("⚠️ Google OR-Tools uygulanabilir bir çözüm bulamadı (kısıtlar sağlanamadı veya süre sınırı aşıldı).")
            if not ga_res["feasible"]:
                st.error("⚠️ Genetik Algoritma cezasız (kapasite ve mesafe kısıtlarını sağlayan) uygulanabilir bir çözüm bulamadı.")

            # KARŞILAŞTIRMA TABLOSU
            st.subheader("⚖️ Çözücü Yöntemleri Karşılaştırması (Aynı Atama & Ortak Maliyet Amaçlı)")
            toplam_yuk = sum(saved_talepler)
            comp_rows = []

            if ga_res["feasible"]:
                comp_rows.append({
                    "Yöntem": "Genetik Algoritma (Permütasyon Tabanlı)",
                    "Toplam Mesafe (km)": round(ga_res["dist_km"], 2),
                    "Kullanılan Araç": ga_res["veh"],
                    "Ort. Araç Doluluğu (%)": round(toplam_yuk / (ga_res["veh"] * saved_arac_kap) * 100, 1) if ga_res["veh"] else 0,
                    "Çözüm Süresi (sn)": f"{ga_res['time']:.2f}",
                    "Tahmini CO₂ Salımı (kg)": round(ga_res["dist_km"] / 100.0 * 15 * 2.65, 1),
                    "Toplam Günlük Maliyet (USD)": round(ga_res["toplam_maliyet"], 2),
                })

            if or_res["feasible"]:
                comp_rows.append({
                    "Yöntem": "Google OR-Tools (Guided Local Search)",
                    "Toplam Mesafe (km)": round(or_res["dist_km"], 2),
                    "Kullanılan Araç": or_res["veh"],
                    "Ort. Araç Doluluğu (%)": round(toplam_yuk / (or_res["veh"] * saved_arac_kap) * 100, 1) if or_res["veh"] else 0,
                    "Çözüm Süresi (sn)": f"{or_res['time']:.2f}",
                    "Tahmini CO₂ Salımı (kg)": round(or_res["dist_km"] / 100.0 * 15 * 2.65, 1),
                    "Toplam Günlük Maliyet (USD)": round(or_res["toplam_maliyet"], 2),
                })

            st.dataframe(pd.DataFrame(comp_rows), hide_index=True, use_container_width=True)
            st.info(f"🏅 **Seçilen En İyi Çözücü:** **{winner_name}** (Toplam Günlük Maliyet: **{winner['toplam_maliyet']:,.2f} USD**). Aşağıdaki harita, rota tablosu ve stres testi bu çözücünün rotaları üzerinden oluşturulmuştur.")

            # MALİYET VE KARBON AYAK İZİ ÖZET KARTLARI
            best_dist_km = winner["dist_km"]
            best_veh = winner["veh"]
            yakit_maliyeti = best_dist_km * 1.25
            filo_cikis_maliyeti = best_veh * 45.0
            tesis_maliyeti = saved_depo_sayisi * saved_depo_maliyet
            genel_toplam = winner["toplam_maliyet"]
            best_co2 = (best_dist_km / 100.0) * 15 * 2.65

            if or_res["feasible"] and ga_res["feasible"]:
                fark_maliyet = abs(ga_res["toplam_maliyet"] - or_res["toplam_maliyet"])
                fark_etiket = f"Diğer çözücüye göre -{fark_maliyet:,.2f} USD"
            else:
                fark_etiket = winner_name

            st.subheader(f"💰 Günlük Lojistik Maliyetleri ve Karbon Ayak İzi Analizi ({winner_name})")
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("⛽ Günlük Yakıt Maliyeti", f"{yakit_maliyeti:,.2f} USD", f"{best_dist_km:,.1f} km (1.25 USD/km)", "off")
            c2.metric("🚚 Araç Çıkış Maliyeti", f"{filo_cikis_maliyeti:,.0f} USD", f"{best_veh} Araç (45 USD/araç)", "off")
            c3.metric("🏢 Tesis Sabit Maliyeti", f"{tesis_maliyeti:,.0f} USD", f"{saved_depo_sayisi} Depo", "off")
            c4.metric("🌱 Karbon Ayak İzi (CO₂)", f"{best_co2:,.1f} kg", "DEFRA 15L/100km", "off")
            c5.metric("💵 TOPLAM MALİYET", f"{genel_toplam:,.2f} USD", fark_etiket, "normal")

            # HARİTA
            m = folium.Map(location=[-1.8312, -78.1834], zoom_start=6, tiles="OpenStreetMap")
            for i, (lat, lon) in enumerate(saved_depot_coords):
                folium.Marker([lat, lon], popup=f"DEPO {i+1}", icon=folium.Icon(color="red", icon="home")).add_to(m)
            for i, (lat, lon) in enumerate(saved_store_coords):
                s_nbr = int(magaza_df.loc[i, "store_nbr"])
                folium.CircleMarker([lat, lon], radius=6, popup=f"M{s_nbr} ({magaza_df.loc[i, 'city']})", color="#1f77b4", fill=True).add_to(m)
            colors = ["red", "green", "purple", "orange", "darkred", "cadetblue", "darkblue"] * 15
            for idx, rc in enumerate(winner["folium_routes_data"]):
                folium.PolyLine(rc, color=colors[idx % len(colors)], weight=3.5).add_to(m)
            components.html(m._repr_html_(), height=550)

            # DETAYLI ARAÇ ROTALARI + CSV İNDİRME
            st.subheader(f"📋 Detaylı Araç Rotaları ({winner_name})")
            df_routes = pd.DataFrame(winner["routes_table"])
            st.dataframe(df_routes, hide_index=True, use_container_width=True)

            csv = df_routes.to_csv(index=False).encode("utf-8")
            st.download_button(label="📥 Rota Tablosunu CSV Olarak İndir", data=csv, file_name="arac_rotalari.csv", mime="text/csv")

            # GERÇEK 30 GÜN STRES TESTİ
            st.markdown("---")
            st.subheader("🧪 Gerçek Gün Stres Testi (Son 30 Günlük Gerçekleşen Satışlar: 17 Temmuz – 15 Ağustos 2017)")
            st.caption("Sentetik normal dağılım yerine, ilk 3 fold dışında tutulan 30 gerçek günün fiili mağaza satışları üzerinden hafta sonu tepeleri ve mağazalar arası eşzamanlı dalgalanmalar doğrudan test edilmiştir.")
            toplam_gun = stres["toplam_gun"]
            tasan_gun = stres["tasan_gun"]
            dayaniklilik = (1.0 - tasan_gun / max(1, toplam_gun)) * 100.0

            c_s1, c_s2, c_s3, c_s4 = st.columns(4)
            c_s1.metric("📅 Aşım Yaşanan Gün", f"{tasan_gun} / {toplam_gun} Gün", f"{toplam_gun - tasan_gun} gün tam uyum", "off")
            c_s2.metric("🛡️ 30 Günlük Filo Dayanıklılığı", f"%{dayaniklilik:.1f}", f"{toplam_gun - tasan_gun}/{toplam_gun} gün sorunsuz")
            c_s3.metric("⚠️ Sefer Bazlı İhlal Oranı", f"%{stres['ihlal_orani']:.2f}", f"{stres['tasan_sefer']} / {stres['toplam_sefer']} sefer")
            c_s4.metric("📦 Ortalama Taşma Miktarı", f"{stres['ort_tasma']:.1f} br", "İhlal oluşan seferlerde")
            st.dataframe(pd.DataFrame(stres["stres_tablosu"]), hide_index=True, use_container_width=True)

            with st.expander("ℹ️ Kabul ve Maliyet Varsayımları (Akademik Raporlama)"):
                st.markdown("""
- **Rotalama Mesafesi:** Araç günlük maks. mesafe sınırı 600 km (sol menüden değiştirilebilir). Kuş uçuşu mesafelere **1,3 yol sapma katsayısı** uygulanmıştır.
- **Zaman Serisi Doğrulama:** Gerçek Favorita verisi üzerinde 4 katmanlı kayan pencere (rolling backtest) ile sızıntısız 30 günlük tahminler üretilmiştir.
- **Emniyet Stoğu:** Her mağaza için ilk 3 fold tahmin hata standart sapması ($\sigma_i$) üzerinden $SS_i = Z \cdot \sigma_i$ ($Z = 1,64 \Rightarrow \%95$ hizmet düzeyi) uygulanmıştır.
- **Emisyon ve Maliyet:** 100 km'de 15 L yakıt tüketimi, litrede 2,65 kg $CO_2$ salımı (DEFRA standardı); araç sabit çıkış maliyeti **45 USD**, kilometre başı yakıt/operasyon maliyeti **1,25 USD** kabul edilmiştir.
                """)
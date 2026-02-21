import streamlit as st
import pandas as pd
import numpy as np
import joblib
import folium
from streamlit_folium import st_folium

# ===============================
# CONFIG + ESTILO
# ===============================
st.set_page_config(page_title="Straterra — Evaluación", layout="wide")

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background-color: #E6EBE0; }

h1, h2, h3 { color: #ED6A5A !important; }

.stButton > button {
    background-color: #ED6A5A !important;
    color: white !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
}

.card {
    background: rgba(255,255,255,0.75);
    border-radius: 14px;
    padding: 16px;
    margin-bottom: 16px;
    border: 1px solid rgba(0,0,0,0.06);
}

.result-box {
    background-color: #F4F1BB;
    padding: 18px;
    border-radius: 12px;
    font-weight: 900;
    font-size: 18px;
    border: 1px solid rgba(0,0,0,0.08);
}

.kpi {
    background: rgba(255,255,255,0.75);
    border-radius: 14px;
    padding: 16px;
    border: 1px solid rgba(0,0,0,0.06);
    height: 100%;
}
.kpi-title {
    font-weight: 800;
    font-size: 15px;
    margin-bottom: 6px;
}
.kpi-value {
    font-size: 40px;
    font-weight: 900;
    line-height: 1;
    margin-bottom: 8px;
}
.badge {
    display:inline-block;
    padding: 6px 10px;
    border-radius: 999px;
    font-weight: 800;
    font-size: 12px;
    border: 1px solid rgba(0,0,0,0.10);
    margin-bottom: 10px;
}
.badge-low { background: #CFE3D6; }
.badge-mid { background: #F4F1BB; }
.badge-high { background: #F6D6D2; }

.small-muted { color: rgba(0,0,0,0.65); font-size: 13px; }
</style>
""", unsafe_allow_html=True)

# ===============================
# HEADER
# ===============================
st.markdown("<h1>Straterra</h1>", unsafe_allow_html=True)
st.markdown("<h3>Evaluación de Ubicación</h3>", unsafe_allow_html=True)

# ===============================
# CARGAR MODELOS / DATA
# ===============================
rf = joblib.load("models/rf_model.joblib")
kmeans = joblib.load("models/kmeans_model.joblib")
tree = joblib.load("models/balltree.joblib")
scaler_index = joblib.load("models/scaler_index.joblib")
scaler_cluster = joblib.load("models/scaler_cluster.joblib")
rest = pd.read_parquet("models/rest_base.parquet")

# ===============================
# FUNCIONES
# ===============================
def cluster_label(idx):
    mapping = {0: "Emergente", 1: "Premium", 2: "Consolidada"}
    return mapping.get(idx, "Zona")

def cluster_color(label):
    # barra zona (turquesa premium como ya te gustó)
    if label == "Premium": return "#36C9C6"
    if label == "Consolidada": return "#CFE3D6"
    if label == "Emergente": return "#F4F1BB"
    return "#DDDDDD"

def crear_features(lat, lon):
    coords = np.radians([[lat, lon]])
    EARTH = 6371000
    r500 = 500 / EARTH
    r1000 = 1000 / EARTH

    n500 = tree.query_radius(coords, r=r500)[0]
    n1k = tree.query_radius(coords, r=r1000)[0]

    comp500 = len(n500)
    comp1k = len(n1k)

    global_mean = rest["per_ocu_num"].mean()

    if len(n1k) > 0:
        tam_vals = rest.iloc[n1k]["per_ocu_num"].dropna()
        tam = tam_vals.mean() if len(tam_vals) > 0 else global_mean
    else:
        tam = global_mean

    dens = comp1k / (np.pi * 1**2)

    df = pd.DataFrame([{
        "competidores_500m": float(comp500),
        "competidores_1km": float(comp1k),
        "densidad_1km_km2": float(dens),
        "tam_prom_1km": float(tam)
    }])

    scaled = scaler_index.transform(df[["densidad_1km_km2","competidores_1km","tam_prom_1km"]])
    df["indice_saturacion"] = float(np.mean(scaled))

    return df

def recomendacion_final(p_peq, p_med, zona):
    # Regla estable y coherente (no la cambio)
    if zona == "Premium":
        if p_med >= 0.60:
            return "Expansión fuerte (zona premium consolidada)"
        else:
            return "Entrada estratégica diferenciada (zona premium)"

    if zona == "Consolidada":
        if p_med >= 0.50:
            return "Expansión mediana estable"
        else:
            return "Negocio pequeño eficiente"

    if zona == "Emergente":
        if p_med >= 0.50:
            return "Expansión con validación (piloto recomendado)"
        else:
            return "Ideal para negocio pequeño"

    return "Evaluación adicional necesaria"

def nivel_badge(value, t1, t2):
    # devuelve (nivel, clase_css)
    if value < t1:
        return "Bajo", "badge-low"
    elif value < t2:
        return "Medio", "badge-mid"
    else:
        return "Alto", "badge-high"

def kpi_card(title, value_str, nivel, badge_css, desc):
    st.markdown(f"""
    <div class="kpi">
        <div class="kpi-title">{title}</div>
        <div class="kpi-value">{value_str}</div>
        <div class="badge {badge_css}">{nivel}</div>
        <div class="small-muted">{desc}</div>
    </div>
    """, unsafe_allow_html=True)

def build_explicacion_recos_riesgos(X, zona, reco, p_peq, p_med):
    # thresholds simples (heurísticos) -> interpretabilidad
    comp500 = float(X["competidores_500m"].iloc[0])
    comp1k = float(X["competidores_1km"].iloc[0])
    dens = float(X["densidad_1km_km2"].iloc[0])
    tam = float(X["tam_prom_1km"].iloc[0])
    sat = float(X["indice_saturacion"].iloc[0])

    # ¿Por qué? (máximo 3 bullets)
    bullets = []
    if p_med >= 0.55:
        bullets.append("Alta probabilidad de operar como negocio mediano–grande (más inversión).")
    else:
        bullets.append("El modelo no ve una ventaja clara para operación mediano–grande (competencia/capacidad).")

    if zona == "Premium":
        bullets.append("Zona Premium: suele exigir diferenciación (marca, experiencia, servicio).")
    elif zona == "Emergente":
        bullets.append("Zona Emergente: oportunidad de capturar demanda temprana, pero con incertidumbre.")
    else:
        bullets.append("Zona Consolidada: demanda más estable, competencia ya asentada.")

    # tercer bullet basado en presión competitiva
    if comp500 >= 80 or sat >= 0.55:
        bullets.append("Presión competitiva alta en el entorno inmediato: clave diferenciarse.")
    else:
        bullets.append("Presión competitiva moderada: hay espacio para posicionarte con buena ejecución.")

    # Recomendaciones rápidas (según recomendación y zona)
    recos = []
    if "Entrada estratégica" in reco:
        recos += [
            "Diferenciar propuesta (menú/servicio) antes de abrir.",
            "Negociar renta con periodo de gracia o escalonada.",
            "Abrir con piloto de 2–4 semanas (horas pico + ticket promedio)."
        ]
    elif "Expansión fuerte" in reco:
        recos += [
            "Asegurar capacidad operativa (personal y procesos) desde el día 1.",
            "Invertir en visibilidad local (señalización + campañas geo).",
            "Monitorear competencia directa y ajustar precios/combos."
        ]
    elif "Expansión con validación" in reco:
        recos += [
            "Abrir con formato controlado (costos fijos bajos).",
            "Validar demanda 4–8 semanas antes de escalar.",
            "Construir awareness local (alianzas + redes)."
        ]
    else:
        recos += [
            "Optimizar costos y operaciones (formato ligero).",
            "Elegir nicho claro vs. competencia cercana.",
            "Probar ubicaciones alternativas cercanas."
        ]

    # Riesgos a vigilar
    riesgos = []
    if comp500 >= 100:
        riesgos.append("Competencia inmediata muy alta: riesgo de guerra de precios.")
    if tam >= 12:
        riesgos.append("Entorno con negocios más grandes: requiere mayor inversión para competir.")
    if sat >= 0.60:
        riesgos.append("Saturación alta: riesgo de demanda insuficiente si no hay diferenciación.")
    if zona == "Emergente":
        riesgos.append("Demanda inestable: depende mucho de crecimiento/flujo de la zona.")

    # mantener conciso
    riesgos = riesgos[:4]

    return bullets[:3], recos[:4], riesgos

# ===============================
# MAPA (no cambiar)
# ===============================
st.subheader("1) Selecciona la ubicación en el mapa")

default_lat, default_lon = 25.676079, -100.318542
if "lat" not in st.session_state:
    st.session_state["lat"] = default_lat
if "lon" not in st.session_state:
    st.session_state["lon"] = default_lon

m = folium.Map(location=[st.session_state["lat"], st.session_state["lon"]], zoom_start=12)
folium.Marker([st.session_state["lat"], st.session_state["lon"]]).add_to(m)

map_data = st_folium(m, height=600, use_container_width=True)

if map_data and map_data.get("last_clicked"):
    st.session_state["lat"] = map_data["last_clicked"]["lat"]
    st.session_state["lon"] = map_data["last_clicked"]["lng"]
    st.success(f"Ubicación seleccionada: {st.session_state['lat']:.6f}, {st.session_state['lon']:.6f}")

st.divider()

# ===============================
# EVALUAR
# ===============================
if st.button("Evaluar ubicación"):
    lat = st.session_state["lat"]
    lon = st.session_state["lon"]

    X = crear_features(lat, lon)

    # Probas
    proba = rf.predict_proba(X)[0]
    probas = dict(zip(rf.classes_, proba))
    p_peq = float(probas.get("Micro", 0.0))
    p_med = float(probas.get("No-Micro", 0.0))

    # Cluster
    cluster_scaled = scaler_cluster.transform(X[["indice_saturacion","densidad_1km_km2","tam_prom_1km"]])
    cl_idx = int(kmeans.predict(cluster_scaled)[0])
    zona = cluster_label(cl_idx)

    # Recomendación (no cambiar lógica)
    reco = recomendacion_final(p_peq, p_med, zona)

    # Texto explicable
    bullets_por_que, acciones, riesgos = build_explicacion_recos_riesgos(X, zona, reco, p_peq, p_med)

    # ===============================
    # RESULTADOS
    # ===============================
    st.subheader("Resultados")
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.write("### Probabilidades")
        st.write(f"**Negocio pequeño:** {round(p_peq*100)}%")
        st.progress(p_peq)
        st.write(f"**Negocio mediano–grande:** {round(p_med*100)}%")
        st.progress(p_med)

        st.markdown(
            f"<div style='background-color:{cluster_color(zona)}; padding:12px; border-radius:12px; font-weight:900;'>Tipo de zona: {zona}</div>",
            unsafe_allow_html=True
        )
        st.markdown("<div class='small-muted'>Tip: “pequeño” = operación ligera/bajo capital. “mediano–grande” = mayor inversión.</div>", unsafe_allow_html=True)

    with col2:
        st.markdown(f"<div class='result-box'>Recomendación: {reco}</div>", unsafe_allow_html=True)

        st.markdown("<div class='card'><h3 style='margin-top:0'>¿Por qué?</h3></div>", unsafe_allow_html=True)
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        for b in bullets_por_que:
            st.write(f"• {b}")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card'><h3 style='margin-top:0'>Acciones sugeridas (rápidas)</h3></div>", unsafe_allow_html=True)
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        for a in acciones:
            st.write(f"• {a}")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card'><h3 style='margin-top:0'>Riesgos a vigilar</h3></div>", unsafe_allow_html=True)
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        if len(riesgos) == 0:
            st.write("• Riesgos no dominantes con la señal actual (vigilar ejecución).")
        else:
            for r in riesgos:
                st.write(f"• {r}")
        st.markdown("</div>", unsafe_allow_html=True)

    st.divider()

    # ===============================
    # KPI INDICADORES (bonitos y entendibles)
    # ===============================
    st.markdown("## Indicadores del entorno (en lenguaje de negocio)")

    comp500 = float(X["competidores_500m"].iloc[0])
    comp1k = float(X["competidores_1km"].iloc[0])
    dens = float(X["densidad_1km_km2"].iloc[0])
    tam = float(X["tam_prom_1km"].iloc[0])
    sat = float(X["indice_saturacion"].iloc[0])

    # Umbrales heurísticos (si quieres los ajustamos con percentiles)
    n1, n2 = 30, 80       # competencia 500m
    d1, d2 = 20, 60       # densidad
    t1, t2 = 5, 12        # tamaño prom
    s1, s2 = 0.40, 0.60   # saturación

    lvl_c, css_c = nivel_badge(comp500, n1, n2)
    lvl_d, css_d = nivel_badge(dens, d1, d2)
    lvl_t, css_t = nivel_badge(tam, t1, t2)
    lvl_s, css_s = nivel_badge(sat, s1, s2)

    k1, k2, k3, k4 = st.columns(4, gap="large")

    with k1:
        kpi_card(
            "Competencia cercana (500 m)",
            f"{int(round(comp500))}",
            lvl_c, css_c,
            "Cuántos restaurantes hay caminando alrededor (zona inmediata)."
        )
    with k2:
        kpi_card(
            "Competencia ampliada (1 km)",
            f"{int(round(comp1k))}",
            *nivel_badge(comp1k, 80, 200),
            "Cuántos restaurantes hay en un radio más grande (zona de influencia)."
        )
    with k3:
        kpi_card(
            "Densidad de restaurantes (1 km)",
            f"{dens:.1f}",
            lvl_d, css_d,
            "Restaurantes por km² (qué tan “cargada” está la zona)."
        )
    with k4:
        kpi_card(
            "Saturación (0–1)",
            f"{sat:.2f}",
            lvl_s, css_s,
            "Resumen de presión competitiva (más cerca de 1 = más competido)."
        )

    k5, k6 = st.columns(2, gap="large")
    with k5:
        kpi_card(
            "Tamaño típico del entorno",
            f"{tam:.1f}",
            lvl_t, css_t,
            "Promedio de tamaño de negocios cercanos (proxy de inversión/escala)."
        )
    with k6:
        st.markdown("""
        <div class="kpi">
          <div class="kpi-title">Lectura rápida</div>
          <div class="small-muted">
            • <b>Bajo</b> = más espacio para entrar<br/>
            • <b>Medio</b> = compites si ejecutas bien<br/>
            • <b>Alto</b> = necesitas diferenciación clara
          </div>
        </div>
        """, unsafe_allow_html=True)

st.caption("Straterra · Evaluación territorial para restaurantes (NL).")




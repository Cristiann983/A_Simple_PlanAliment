"""
app.py
─────────────────────────────────────────────────────────────────────────────
Dashboard Interactivo en Streamlit:
"Sistema de optimización de planes de alimentación diarios de mínimo costo
mediante un Algoritmo Genético Simple"

Implementa:
- Catálogo inicial de 20 productos con precios del mercado local mexicano (Profeco).
- Estructura de Dos Listas (lista_productos + lista_porciones / cromosoma,
  y poblacion_actual + nueva_poblacion).
- Las 6 etapas del Algoritmo Genético Simple:
  1. Inicialización
  2. Evaluación (Fitness = Costo + Penalizaciones, y Aptitud de adaptación)
  3. Selección (Ruleta, Torneo, Ranking)
  4. Cruce (Crossover de 1 punto a la mitad, 1 punto aleatorio, 2 puntos, uniforme)
  5. Mutación (~5% de probabilidad)
  6. Reemplazo y Criterio de Parada (Generaciones o Convergencia)
- Tablas canónicas 6.1 (Tiempo 0) y 6.2 (Tiempo 1), curvas de adaptación
  y métricas On-line / Off-line de De Jong del documento de Algoritmos Genéticos.
"""

from __future__ import annotations
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from catalogo_productos import (
    obtener_catalogo_base,
    generar_catalogo_escalado,
)
from algoritmo_genetico import (
    ObjetivosNutricionales,
    PesosPenalizacion,
    ParametrosAGS,
    ejecutar_algoritmo_genetico_simple,
)

# ─── Configuración de la página ──────────────────────────────────────────────
st.set_page_config(
    page_title="Optimizador Nutricional · Algoritmo Genético Simple",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Estilos CSS personalizados ──────────────────────────────────────────────
st.markdown(
    """
    <style>
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 60%, #0d9488 100%);
        padding: 1.4rem 1.8rem;
        border-radius: 14px;
        color: #ffffff;
        margin-bottom: 1.2rem;
        box-shadow: 0 8px 20px rgba(15, 23, 42, 0.15);
    }
    .main-header h1 {
        margin: 0;
        font-size: 1.75rem;
        font-weight: 700;
        color: #ffffff;
    }
    .main-header p {
        margin: 0.35rem 0 0 0;
        font-size: 0.96rem;
        color: #e2e8f0;
    }
    .kpi-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-left: 5px solid #0d9488;
        border-radius: 12px;
        padding: 1rem 1.15rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
        margin-bottom: 0.8rem;
    }
    .kpi-card-cost {
        border-left-color: #16a34a;
        background: linear-gradient(180deg, #f0fdf4 0%, #ffffff 100%);
    }
    .kpi-card-blue {
        border-left-color: #2563eb;
    }
    .kpi-card-amber {
        border-left-color: #d97706;
    }
    .kpi-card-purple {
        border-left-color: #7c3aed;
    }
    .kpi-title {
        font-size: 0.80rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: #64748b;
        font-weight: 600;
        margin-bottom: 0.25rem;
    }
    .kpi-value {
        font-size: 1.55rem;
        font-weight: 800;
        color: #0f172a;
        margin: 0;
    }
    .kpi-sub {
        font-size: 0.80rem;
        color: #475569;
        margin-top: 0.25rem;
    }
    .step-box {
        background: #f8fafc;
        border: 1px solid #cbd5e1;
        border-radius: 10px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 0.75rem;
    }
    .step-badge {
        display: inline-block;
        background: #1e3a8a;
        color: white;
        font-weight: 700;
        font-size: 0.78rem;
        padding: 0.18rem 0.6rem;
        border-radius: 999px;
        margin-bottom: 0.35rem;
    }
    .gene-box {
        display: inline-block;
        padding: 0.35rem 0.55rem;
        margin: 0.18rem;
        border-radius: 6px;
        font-family: monospace;
        font-size: 0.88rem;
        font-weight: 700;
        text-align: center;
    }
    .gene-active {
        background: #dcfce7;
        border: 1px solid #16a34a;
        color: #14532d;
    }
    .gene-zero {
        background: #f1f5f9;
        border: 1px solid #cbd5e1;
        color: #94a3b8;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─── Inicialización de estado (Catálogo de 20 productos editable) ────────────
if "catalogo_productos" not in st.session_state:
    st.session_state["catalogo_productos"] = obtener_catalogo_base()

# ─── Barra Lateral: Parámetros del Usuario y del Algoritmo Genético ──────────
with st.sidebar:
    st.markdown("## Objetivos Nutricionales")

    perfil = st.selectbox(
        "Perfil nutricional rápido",
        options=[
            "Deportista / Corredor (PDF: 2000 kcal, 150g prot)",
            "Déficit Calórico / Definición (1700 kcal)",
            "Volumen Muscular (2600 kcal, 180g prot)",
            "Mantenimiento Económico (2000 kcal, 110g prot)",
            "Personalizado",
        ],
        index=0,
    )

    perfiles_map = {
        "Deportista / Corredor (PDF: 2000 kcal, 150g prot)": (2000.0, 150.0, 200.0, 65.0),
        "Déficit Calórico / Definición (1700 kcal)": (1700.0, 140.0, 155.0, 55.0),
        "Volumen Muscular (2600 kcal, 180g prot)": (2600.0, 180.0, 290.0, 80.0),
        "Mantenimiento Económico (2000 kcal, 110g prot)": (2000.0, 110.0, 245.0, 65.0),
    }

    if perfil in perfiles_map:
        def_kcal, def_prot, def_carb, def_grasa = perfiles_map[perfil]
    else:
        def_kcal, def_prot, def_carb, def_grasa = (2000.0, 150.0, 200.0, 65.0)

    c1, c2 = st.columns(2)
    with c1:
        obj_kcal = st.number_input("Calorías (kcal)", min_value=800.0, max_value=5000.0, value=def_kcal, step=50.0)
        obj_carb = st.number_input("Carbohidratos (g)", min_value=30.0, max_value=600.0, value=def_carb, step=5.0)
    with c2:
        obj_prot = st.number_input("Proteínas (g)", min_value=30.0, max_value=350.0, value=def_prot, step=5.0)
        obj_grasa = st.number_input("Grasas (g)", min_value=20.0, max_value=200.0, value=def_grasa, step=5.0)

    kcal_teoricas = 4.0 * obj_prot + 4.0 * obj_carb + 9.0 * obj_grasa
    st.caption(
        f"💡 Equivalencia calórica de tus macros (4P + 4C + 9G): **{kcal_teoricas:.0f} kcal**"
    )

    st.markdown("---")
    st.markdown("##  Parámetros del AGS")

    metodo_sel_label = st.selectbox(
        "3. Método de Selección",
        options=[
            "Ruleta (Proporcional a la Aptitud)",
            "Torneo",
            "Ranking (Proporcional al Rango)",
        ],
        index=0,
        help="Métodos de selección del Algoritmo Genético Simple (Sección 3.3 del PDF).",
    )
    metodo_seleccion_map = {
        "Ruleta (Proporcional a la Aptitud)": "ruleta",
        "Torneo": "torneo",
        "Ranking (Proporcional al Rango)": "ranking",
    }
    metodo_seleccion = metodo_seleccion_map[metodo_sel_label]

    tipo_cruce_label = st.selectbox(
        "4. Operador de Cruce (Crossover)",
        options=[
            "Un punto (Corte a la mitad - PDF 1)",
            "Un punto (Punto aleatorio - PDF 2)",
            "Dos puntos (PDF 2)",
            "Uniforme (Máscara aleatoria - PDF 2)",
        ],
        index=0,
        help="Intercambio de segmentos genéticos entre dos listas padre.",
    )
    tipo_cruce_map = {
        "Un punto (Corte a la mitad - PDF 1)": "un_punto_mitad",
        "Un punto (Punto aleatorio - PDF 2)": "un_punto_aleatorio",
        "Dos puntos (PDF 2)": "dos_puntos",
        "Uniforme (Máscara aleatoria - PDF 2)": "uniforme",
    }
    tipo_cruce = tipo_cruce_map[tipo_cruce_label]

    p1, p2 = st.columns(2)
    with p1:
        tam_poblacion = st.slider("Población (λ)", min_value=5, max_value=300, value=120, step=5)
        prob_cruce = st.slider("Prob. Cruce (pc)", min_value=0.40, max_value=1.00, value=0.85, step=0.05)
    with p2:
        num_generaciones = st.slider("Generaciones (T)", min_value=20, max_value=500, value=250, step=10)
        tasa_mutacion_pct = st.slider("5. Mutación (pm %)", min_value=1.0, max_value=20.0, value=5.0, step=0.5)

    max_porciones = st.slider("Máx. porciones por producto", min_value=2, max_value=10, value=6, step=1)

    with st.expander("⚖️ Pesos de Penalización y Parada"):
        w_kcal = st.number_input("Peso desviación Kcal", min_value=0.1, max_value=100.0, value=2.5, step=0.5)
        w_prot = st.number_input("Peso desviación Proteínas", min_value=0.5, max_value=1000.0, value=15.0, step=1.0)
        w_carb = st.number_input("Peso desviación Carbohidratos", min_value=0.5, max_value=1000.0, value=10.0, step=1.0)
        w_grasa = st.number_input("Peso desviación Grasas", min_value=0.5, max_value=1000.0, value=12.0, step=1.0)
        usar_elitismo = st.checkbox("Preservar mejor individuo (Elitismo)", value=True)
        detener_conv = st.checkbox("Detener por criterio de convergencia", value=True)
        usar_semilla_fija = st.checkbox("Usar semilla fija (reproducibilidad)", value=True)
        valor_semilla = st.number_input("Semilla aleatoria", min_value=1, max_value=99999, value=42, step=1)

    btn_ejecutar = st.button(" Ejecutar Algoritmo Genético", type="primary", use_container_width=True)

# ─── Encabezado Principal ────────────────────────────────────────────────────
st.markdown(
    """
    <div class="main-header">
        <h1> Sistema de Optimización de Planes de Alimentación Diarios de Mínimo Costo</h1>
        <p>Algoritmo Genético Simple (Canónico) · Catálogo de 20 Productos del Mercado Local (MXN) · Modelo de Dos Listas</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─── Ejecución del Algoritmo Genético Simple ─────────────────────────────────
objetivos = ObjetivosNutricionales(
    kcal=obj_kcal,
    proteinas_g=obj_prot,
    carbohidratos_g=obj_carb,
    grasas_g=obj_grasa,
)
pesos = PesosPenalizacion(
    peso_kcal=w_kcal,
    peso_proteinas=w_prot,
    peso_carbohidratos=w_carb,
    peso_grasas=w_grasa,
)
parametros = ParametrosAGS(
    tamano_poblacion=tam_poblacion,
    num_generaciones=num_generaciones,
    prob_cruce=prob_cruce,
    tasa_mutacion=tasa_mutacion_pct / 100.0,
    max_porciones_por_producto=max_porciones,
    metodo_seleccion=metodo_seleccion,
    tipo_cruce=tipo_cruce,
    elitismo=usar_elitismo,
    detener_por_convergencia=detener_conv,
    semilla=int(valor_semilla) if usar_semilla_fija else None,
)

resultado = ejecutar_algoritmo_genetico_simple(
    lista_productos=st.session_state["catalogo_productos"],
    objetivos=objetivos,
    pesos=pesos,
    parametros=parametros,
)

mejor = resultado["mejor_individuo"]
df_menu = pd.DataFrame(resultado["desglose_menu"])
df_dos_listas = pd.DataFrame(resultado["mapeo_dos_listas"])
df_hist = pd.DataFrame(resultado["historial"])

# ─── Pestañas Principales del Dashboard ──────────────────────────────────────
tab_menu, tab_proceso, tab_convergencia, tab_catalogo, tab_validacion = st.tabs(
    [
        " 1. Menú Óptimo y Desglose",
        " 2. Dos Listas y Proceso del AGS (6 Fases)",
        " 3. Convergencia y Tablas Canónicas (PDF 2)",
        " 4. Catálogo de 20 Productos (Precios MXN)",
        " 5. Validación Múltiple y Escalamiento",
    ]
)

# =============================================================================
# PESTAÑA 1: MENÚ ÓPTIMO Y DESGLOSE NUTRICIONAL
# =============================================================================
with tab_menu:
    col_k1, col_k2, col_k3, col_k4, col_k5, col_k6 = st.columns(6)

    with col_k1:
        st.markdown(
            f"""
            <div class="kpi-card kpi-card-cost">
                <div class="kpi-title">Costo Total Diario</div>
                <p class="kpi-value">${mejor.costo_total:,.2f} MXN</p>
                <div class="kpi-sub">{len(df_menu)} productos activos ({sum(mejor.cromosoma)} porciones)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_k2:
        pct_kcal = (mejor.kcal_totales / objetivos.kcal * 100) if objetivos.kcal else 0
        st.markdown(
            f"""
            <div class="kpi-card kpi-card-blue">
                <div class="kpi-title">Calorías</div>
                <p class="kpi-value">{mejor.kcal_totales:,.0f} kcal</p>
                <div class="kpi-sub">Meta: {objetivos.kcal:,.0f} ({pct_kcal:.1f}% · Δ {mejor.desviacion_kcal:.0f})</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_k3:
        pct_prot = (mejor.proteinas_totales / objetivos.proteinas_g * 100) if objetivos.proteinas_g else 0
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Proteínas</div>
                <p class="kpi-value">{mejor.proteinas_totales:,.1f} g</p>
                <div class="kpi-sub">Meta: {objetivos.proteinas_g:,.0f} g ({pct_prot:.1f}% · Δ {mejor.desviacion_proteinas:.1f}g)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_k4:
        pct_carb = (mejor.carbohidratos_totales / objetivos.carbohidratos_g * 100) if objetivos.carbohidratos_g else 0
        st.markdown(
            f"""
            <div class="kpi-card kpi-card-amber">
                <div class="kpi-title">Carbohidratos</div>
                <p class="kpi-value">{mejor.carbohidratos_totales:,.1f} g</p>
                <div class="kpi-sub">Meta: {objetivos.carbohidratos_g:,.0f} g ({pct_carb:.1f}% · Δ {mejor.desviacion_carbohidratos:.1f}g)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_k5:
        pct_grasa = (mejor.grasas_totales / objetivos.grasas_g * 100) if objetivos.grasas_g else 0
        st.markdown(
            f"""
            <div class="kpi-card kpi-card-purple">
                <div class="kpi-title">Grasas</div>
                <p class="kpi-value">{mejor.grasas_totales:,.1f} g</p>
                <div class="kpi-sub">Meta: {objetivos.grasas_g:,.0f} g ({pct_grasa:.1f}% · Δ {mejor.desviacion_grasas:.1f}g)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_k6:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Fitness / Ejecución</div>
                <p class="kpi-value">{mejor.fitness_penalizado:,.1f}</p>
                <div class="kpi-sub">Gen {resultado['generacion_final']} · {resultado['tiempo_ejecucion_s']*1000:.0f} ms</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.info(f" **Criterio de parada cumplido:** {resultado['motivo_parada']}.")

    col_left, col_right = st.columns([1.35, 1.0])

    with col_left:
        st.markdown("###  Plan de Alimentación Diario Óptimo (Mínimo Costo)")
        if not df_menu.empty:
            df_mostrar = df_menu[
                [
                    "id",
                    "producto",
                    "porciones",
                    "porcion_unitaria",
                    "precio_unitario_mxn",
                    "subtotal_mxn",
                    "kcal",
                    "proteinas_g",
                    "carbohidratos_g",
                    "grasas_g",
                ]
            ].rename(
                columns={
                    "id": "ID",
                    "producto": "Ingrediente",
                    "porciones": "Porciones (Gen)",
                    "porcion_unitaria": "Medida por Porción",
                    "precio_unitario_mxn": "Precio Unit. ($)",
                    "subtotal_mxn": "Subtotal ($ MXN)",
                    "kcal": "Kcal",
                    "proteinas_g": "Prot (g)",
                    "carbohidratos_g": "Carb (g)",
                    "grasas_g": "Grasa (g)",
                }
            )
            st.dataframe(
                df_mostrar,
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.warning("El cromosoma generado no seleccionó porciones.")

        # Visualización rápida del vector Cromosoma (Lista 2)
        st.markdown("####  Cromosoma Óptimo (Arreglo de 20 Enteros = Porciones por Producto)")
        chips_html = ""
        for item in resultado["mapeo_dos_listas"]:
            clase = "gene-active" if item["porciones"] > 0 else "gene-zero"
            id_prod = item["id"]
            nom_prod = item["producto"]
            porc_prod = item["porciones"]
            chips_html += (
                f'<div class="gene-box {clase}" title="ID {id_prod}: {nom_prod}">'
                f'ID {id_prod}<br>{nom_prod[:10]}<br><b>{porc_prod}</b></div>'
            )
        st.markdown(chips_html, unsafe_allow_html=True)

    with col_right:
        st.markdown("### Cumplimiento de Objetivos Nutricionales")
        df_comp = pd.DataFrame(
            {
                "Nutriente": ["Calorías (÷10)", "Proteínas (g)", "Carbohidratos (g)", "Grasas (g)"],
                "Objetivo": [
                    objetivos.kcal / 10.0,
                    objetivos.proteinas_g,
                    objetivos.carbohidratos_g,
                    objetivos.grasas_g,
                ],
                "Menú Obtenido": [
                    mejor.kcal_totales / 10.0,
                    mejor.proteinas_totales,
                    mejor.carbohidratos_totales,
                    mejor.grasas_totales,
                ],
            }
        )
        fig_bar = go.Figure()
        fig_bar.add_trace(
            go.Bar(
                name="Objetivo del Usuario",
                x=df_comp["Nutriente"],
                y=df_comp["Objetivo"],
                marker_color="#1e3a8a",
                text=[f"{v:.1f}" for v in df_comp["Objetivo"]],
                textposition="auto",
            )
        )
        fig_bar.add_trace(
            go.Bar(
                name="Menú Generado (AGS)",
                x=df_comp["Nutriente"],
                y=df_comp["Menú Obtenido"],
                marker_color="#0d9488",
                text=[f"{v:.1f}" for v in df_comp["Menú Obtenido"]],
                textposition="auto",
            )
        )
        fig_bar.update_layout(
            barmode="group",
            height=330,
            margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        if not df_menu.empty:
            paleta_menu = [
                "#0d9488", "#1e3a8a", "#2563eb", "#16a34a", "#0284c7",
                "#7c3aed", "#d97706", "#059669", "#4f46e5", "#0891b2",
            ]
            fig_pie = px.pie(
                df_menu,
                names="producto",
                values="subtotal_mxn",
                hole=0.45,
                title="Distribución del Costo Diario ($ MXN) por Ingrediente",
                color_discrete_sequence=paleta_menu,
            )
            fig_pie.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=10))
            st.plotly_chart(fig_pie, use_container_width=True)


# =============================================================================
# PESTAÑA 2: LAS DOS LISTAS Y LAS 6 FASES DEL ALGORITMO GENÉTICO SIMPLE
# =============================================================================
with tab_proceso:
    st.markdown("### Estructura de Dos Listas en el Algoritmo Genético Simple")
    col_l1, col_l2 = st.columns(2)

    with col_l1:
        st.markdown(
            """
            <div class="step-box">
                <span class="step-badge">PAR DE LISTAS 1 · REPRESENTACIÓN DEL PROBLEMA</span>
                <h4>1) <code>lista_productos</code> ↔ 2) <code>lista_porciones</code> (Cromosoma)</h4>
                <p>El modelo utiliza dos listas paralelas de longitud <b>N = 20</b>:</p>
                <ul>
                    <li><b>Lista 1 (<code>lista_productos</code>):</b> Almacena los 20 productos del catálogo con su precio en MXN, calorías, proteínas, carbohidratos y grasas por porción.</li>
                    <li><b>Lista 2 (<code>lista_porciones</code> / Cromosoma):</b> Lista de 20 números enteros <code>[p₀, p₁, ..., p₁₉]</code> donde la posición <code>i</code> indica cuántas porciones del producto <code>i</code> se incluyen en el menú diario.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_l2:
        st.markdown(
            """
            <div class="step-box">
                <span class="step-badge">PAR DE LISTAS 2 · CICLO REPRODUCTIVO GENERACIONAL</span>
                <h4>1) <code>poblacion_actual</code> (Padres) ↔ 2) <code>nueva_poblacion</code> (Hijos)</h4>
                <p>Durante la evolución (Figura 6.1 del PDF de Algoritmos Genéticos):</p>
                <ul>
                    <li><b>Lista A (<code>poblacion_actual</code>):</b> Contiene los <code>λ</code> individuos de la generación actual <code>t</code> que compiten en la selección.</li>
                    <li><b>Lista B (<code>nueva_poblacion</code>):</b> Almacena los nuevos descendientes producidos tras el cruce y la mutación. En el paso de <b>Reemplazo</b>, sustituye a <code>poblacion_actual</code>.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("####  Correspondencia 1 a 1 entre `lista_productos` (Lista 1) y `lista_porciones` (Lista 2 - Cromosoma)")
    st.dataframe(
        df_dos_listas.rename(
            columns={
                "id": "Índice (Posición i)",
                "producto": "Lista 1: Producto",
                "categoria": "Categoría",
                "porcion_unitaria": "Porción Estándar",
                "precio_unitario_mxn": "Lista 1: Precio ($ MXN)",
                "porciones": "Lista 2: Gen / Porciones",
                "subtotal_mxn": "Costo Resultante ($)",
                "kcal": "Kcal Resultantes",
                "proteinas_g": "Proteína (g)",
                "carbohidratos_g": "Carbohidratos (g)",
                "grasas_g": "Grasas (g)",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("---")
    st.markdown("###  Las 6 Fases del Proceso del Algoritmo Genético Simple")

    f1, f2, f3 = st.columns(3)
    with f1:
        st.markdown(
            f"""
            <div class="step-box">
                <span class="step-badge">FASE 1 · INICIALIZACIÓN</span>
                <p>Se genera <code>poblacion_actual</code> con <b>λ = {tam_poblacion}</b> individuos (cromosomas de 20 enteros entre <code>0</code> y <code>{max_porciones}</code>) de forma aleatoria.</p>
                <p><b>Ejemplo Individuo #1 en t=0:</b><br><code>{resultado['tabla_tiempo_0'][0].cromosoma}</code></p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div class="step-box">
                <span class="step-badge">FASE 4 · CRUCE (CROSSOVER)</span>
                <p>Con probabilidad <b>p_c = {prob_cruce:.2f}</b> ({tipo_cruce_label}), se combinan los cromosomas de dos padres intercambiando segmentos para generar 2 hijos.</p>
                <p><b>Punto de corte en Pareja #1 (t=1):</b> Posición <code>{resultado['tabla_tiempo_1'][0]['punto_cruce']}</code></p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with f2:
        st.markdown(
            f"""
            <div class="step-box">
                <span class="step-badge">FASE 2 · EVALUACIÓN (FITNESS)</span>
                <p>Se calcula para cada individuo:<br>
                <code>Fitness = Costo + Penalizaciones</code><br>
                <code>Aptitud g(x) = 10000 / (1 + Fitness)</code></p>
                <p><b>Mejor Fitness inicial (t=0):</b> <code>{df_hist.iloc[0]['mejor_fitness']:,.2f}</code><br>
                <b>Mejor Fitness final (t={resultado['generacion_final']}):</b> <code>{mejor.fitness_penalizado:,.2f}</code></p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div class="step-box">
                <span class="step-badge">FASE 5 · MUTACIÓN</span>
                <p>Con baja probabilidad <b>p_m = {tasa_mutacion_pct:.1f}%</b> por gen, se suma o resta aleatoriamente 1 porción para mantener la diversidad genética y evitar óptimos locales.</p>
                <p><b>Genes esperados mutados por hijo:</b> ~<code>{20 * (tasa_mutacion_pct/100.0):.1f}</code> de 20 genes</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with f3:
        st.markdown(
            f"""
            <div class="step-box">
                <span class="step-badge">FASE 3 · SELECCIÓN</span>
                <p>Método activo: <b>{metodo_sel_label}</b>. Los individuos con mejor adaptación tienen mayor probabilidad <code>p_j = g(x_j)/∑g(x)</code> de reproducirse.</p>
                <p><b>Parejas seleccionadas por generación:</b> <code>{tam_poblacion // 2}</code> parejas</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div class="step-box">
                <span class="step-badge">FASE 6 · REEMPLAZO Y PARADA</span>
                <p>La lista <code>nueva_poblacion</code> reemplaza a <code>poblacion_actual</code> y el ciclo se repite hasta cumplir el número de generaciones o convergencia.</p>
                <p><b>Generaciones ejecutadas:</b> <code>{resultado['generacion_final']} / {num_generaciones}</code></p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# =============================================================================
# PESTAÑA 3: CONVERGENCIA Y TABLAS CANÓNICAS (PDF 2 ALGORITMOS GENÉTICOS)
# =============================================================================
with tab_convergencia:
    st.markdown("###  Curvas de Evolución y Convergencia (Figura 6.4 y Sección 3.8 del PDF 2)")

    g_col1, g_col2 = st.columns(2)

    with g_col1:
        # Gráfica equivalente a la Figura 6.4 del PDF 2: Mejor Adaptación vs Adaptación Media
        fig_adapt = go.Figure()
        fig_adapt.add_trace(
            go.Scatter(
                x=df_hist["generacion"],
                y=df_hist["mejor_aptitud"],
                mode="lines",
                name="Mejor Adaptación g*(t)",
                line=dict(color="#16a34a", width=3),
            )
        )
        fig_adapt.add_trace(
            go.Scatter(
                x=df_hist["generacion"],
                y=df_hist["aptitud_media"],
                mode="lines",
                name="Adaptación Media ḡ(t)",
                line=dict(color="#2563eb", width=2, dash="dash"),
            )
        )
        fig_adapt.update_layout(
            title="Figura 6.4 (PDF 2): Adaptación Media y Mejor Adaptación por Generación",
            xaxis_title="Generaciones (t)",
            yaxis_title="Función de Adaptación g(x) [Mayor es mejor]",
            height=360,
            margin=dict(l=20, r=20, t=45, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_adapt, use_container_width=True)

    with g_col2:
        # Gráfica de minimización del Fitness Penalizado (Costo + Penalización, PDF 1 Sección 2.7.3)
        fig_fit = go.Figure()
        fig_fit.add_trace(
            go.Scatter(
                x=df_hist["generacion"],
                y=df_hist["mejor_fitness"],
                mode="lines",
                name="Mejor Fitness Penalizado (Costo + Penalización)",
                line=dict(color="#0d9488", width=3),
            )
        )
        fig_fit.add_trace(
            go.Scatter(
                x=df_hist["generacion"],
                y=df_hist["mejor_costo_mxn"],
                mode="lines",
                name="Costo del Mejor Menú ($ MXN)",
                line=dict(color="#d97706", width=2),
            )
        )
        fig_fit.update_layout(
            title="Minimización del Fitness Penalizado y Costo ($ MXN)",
            xaxis_title="Generaciones (t)",
            yaxis_title="Valor Penalizado / Costo MXN [Menor es mejor]",
            height=360,
            margin=dict(l=20, r=20, t=45, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_fit, use_container_width=True)

    g_col3, g_col4 = st.columns(2)
    with g_col3:
        # Medidas de evaluación de De Jong (1975) - Sección 3.8 PDF 2 (pág. 14-15)
        fig_dejong = go.Figure()
        fig_dejong.add_trace(
            go.Scatter(
                x=df_hist["generacion"],
                y=df_hist["dejong_offline_aptitud"],
                mode="lines",
                name="Evaluación Off-line v_off-line(T)",
                line=dict(color="#7c3aed", width=3),
            )
        )
        fig_dejong.add_trace(
            go.Scatter(
                x=df_hist["generacion"],
                y=df_hist["dejong_online_aptitud"],
                mode="lines",
                name="Evaluación On-line v_on-line(T)",
                line=dict(color="#ec4899", width=2, dash="dot"),
            )
        )
        fig_dejong.update_layout(
            title="Sección 3.8 (PDF 2): Evaluación On-line y Off-line de De Jong",
            xaxis_title="Generaciones (T)",
            yaxis_title="Adaptación Acumulada Promedio",
            height=330,
            margin=dict(l=20, r=20, t=45, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_dejong, use_container_width=True)

    with g_col4:
        # Progresión hacia la uniformidad genética (Convergencia de De Jong, pág. 4 PDF 2)
        fig_conv = go.Figure()
        fig_conv.add_trace(
            go.Scatter(
                x=df_hist["generacion"],
                y=df_hist["convergencia_genes"],
                mode="lines",
                name="% Uniformidad Genética",
                line=dict(color="#1e3a8a", width=3),
                fill="tozeroy",
            )
        )
        fig_conv.add_hline(
            y=95.0,
            line_dash="dash",
            line_color="#dc2626",
            annotation_text="Umbral 95% De Jong",
        )
        fig_conv.update_layout(
            title="Progresión hacia la Uniformidad Genética (% de Convergencia)",
            xaxis_title="Generaciones (t)",
            yaxis_title="Uniformidad Promedio por Gen (%)",
            yaxis=dict(range=[0, 105]),
            height=330,
            margin=dict(l=20, r=20, t=45, b=20),
        )
        st.plotly_chart(fig_conv, use_container_width=True)

    st.markdown("---")
    st.markdown("###  Tabla 6.1 (PDF 2): Población Inicial en el Tiempo $t = 0$ y Ruleta Sesgada")
    st.caption(
        "Muestra los individuos de la población inicial, su cromosoma (ristra de 20 enteros), "
        "su costo, fitness penalizado, función de adaptación f(x), probabilidad de selección "
        "f(x)/∑f(x) y probabilidad acumulada."
    )

    filas_t0 = []
    for ind in resultado["tabla_tiempo_0"]:
        filas_t0.append(
            {
                "Individuo": ind.id_individuo,
                "Cromosoma (20 genes = porciones)": str(ind.cromosoma),
                "Costo ($ MXN)": ind.costo_total,
                "Kcal": ind.kcal_totales,
                "Prot (g)": ind.proteinas_totales,
                "Fitness Penalizado": ind.fitness_penalizado,
                "f(x) Adaptación": round(ind.aptitud, 4),
                "f(x)/∑f(x) (Prob. Selección)": round(ind.prob_seleccion, 4),
                "Probabilidad Acumulada": round(ind.prob_acumulada, 4),
            }
        )
    df_t0 = pd.DataFrame(filas_t0)
    st.dataframe(df_t0.head(20), use_container_width=True, hide_index=True)

    suma_f_t0 = sum(ind.aptitud for ind in resultado["tabla_tiempo_0"])
    media_f_t0 = suma_f_t0 / len(resultado["tabla_tiempo_0"])
    mejor_f_t0 = max(ind.aptitud for ind in resultado["tabla_tiempo_0"])
    st.markdown(
        f"**Resumen Tabla 6.1 ($t=0$):** &nbsp; "
        f"**Suma $\\sum f(x)$:** `{suma_f_t0:.4f}` &nbsp;|&nbsp; "
        f"**Media $\\bar{{f}}(x)$:** `{media_f_t0:.4f}` &nbsp;|&nbsp; "
        f"**Mejor $f^*(x)$:** `{mejor_f_t0:.4f}`"
    )

    st.markdown("---")
    st.markdown("###  Tabla 6.2 (PDF 2): Operadores de Selección, Cruce y Mutación en el Tiempo $t = 1$")
    st.caption(
        "Muestra el emparejamiento de padres seleccionados, el punto de cruce, los descendientes "
        "obtenidos tras el cruce y la nueva población de descendientes mutados."
    )
    filas_t1 = []
    for op in resultado["tabla_tiempo_1"][:20]:
        filas_t1.append(
            {
                "Pareja": op["pareja"],
                "Padre ID": op["id_padre"],
                "Aleatorio u": op["aleatorio_u"],
                "Cromosoma Padre": str(op["cromosoma_padre"]),
                "Punto de Cruce": op["punto_cruce"],
                "Descendiente Cruzado": str(op["descendiente_cruzado"]),
                "Descendiente Mutado": str(op["descendiente_mutado"]),
                "Genes Mutados (Índices)": str(op["genes_mutados"]) if op["genes_mutados"] else "Ninguno",
                "Costo Hijo ($)": op["costo_mxn"],
                "Fitness Penalizado": op["fitness_penalizado"],
                "f(x) Adaptación": round(op["aptitud"], 4),
            }
        )
    df_t1 = pd.DataFrame(filas_t1)
    st.dataframe(df_t1, use_container_width=True, hide_index=True)


# =============================================================================
# PESTAÑA 4: CATÁLOGO DE 20 PRODUCTOS (PRECIOS DE MERCADO LOCAL)
# =============================================================================
with tab_catalogo:
    st.markdown("###  Catálogo de 20 Productos del Mercado Local Mexicano (Profeco / PACIC)")
    st.caption(
        "Puedes editar directamente cualquier precio o valor nutricional en la tabla inferior "
        "para simular cambios de precios en el mercado local. Los primeros 3 productos corresponden "
        "exactamente a los ejemplos de la sección 2.7.2 del PDF Generalidades del Proyecto."
    )

    df_cat = pd.DataFrame(st.session_state["catalogo_productos"])
    df_editado = st.data_editor(
        df_cat,
        use_container_width=True,
        hide_index=True,
        num_rows="fixed",
        column_config={
            "id": st.column_config.NumberColumn("ID", disabled=True),
            "nombre": st.column_config.TextColumn("Producto / Ingrediente"),
            "porcion_desc": st.column_config.TextColumn("Tamaño de Porción"),
            "precio_mxn": st.column_config.NumberColumn("Precio por Porción ($ MXN)", min_value=0.5, max_value=500.0, step=0.5, format="$%.2f"),
            "kcal": st.column_config.NumberColumn("Calorías (kcal)", min_value=0, max_value=1500, step=5),
            "proteinas_g": st.column_config.NumberColumn("Proteínas (g)", min_value=0.0, max_value=100.0, step=0.5),
            "carbohidratos_g": st.column_config.NumberColumn("Carbohidratos (g)", min_value=0.0, max_value=150.0, step=0.5),
            "grasas_g": st.column_config.NumberColumn("Grasas (g)", min_value=0.0, max_value=100.0, step=0.5),
            "categoria": st.column_config.TextColumn("Categoría"),
            "referencia_mercado": st.column_config.TextColumn("Referencia Precio Mercado Local"),
        },
    )

    col_btn_cat1, col_btn_cat2, _ = st.columns([1, 1, 2])
    with col_btn_cat1:
        if st.button(" Aplicar Cambios al Catálogo", use_container_width=True):
            st.session_state["catalogo_productos"] = df_editado.to_dict(orient="records")
            st.rerun()
    with col_btn_cat2:
        if st.button(" Restaurar 20 Productos Originales", use_container_width=True):
            st.session_state["catalogo_productos"] = obtener_catalogo_base()
            st.rerun()

    st.markdown("---")
    st.markdown("####  Eficiencia Económico-Nutricional de los 20 Productos")
    df_ef = df_editado.copy()
    df_ef["Proteína por cada $10 MXN (g)"] = (df_ef["proteinas_g"] / df_ef["precio_mxn"] * 10.0).round(2)
    df_ef["Calorías por cada $1 MXN (kcal)"] = (df_ef["kcal"] / df_ef["precio_mxn"]).round(1)

    c_ef1, c_ef2 = st.columns(2)
    with c_ef1:
        fig_ef_prot = px.bar(
            df_ef.sort_values("Proteína por cada $10 MXN (g)", ascending=True),
            x="Proteína por cada $10 MXN (g)",
            y="nombre",
            color="categoria",
            orientation="h",
            title="Gramos de Proteína obtenidos por cada $10 MXN",
        )
        fig_ef_prot.update_layout(height=450, margin=dict(l=20, r=20, t=40, b=20), yaxis_title="")
        st.plotly_chart(fig_ef_prot, use_container_width=True)

    with c_ef2:
        fig_ef_kcal = px.bar(
            df_ef.sort_values("Calorías por cada $1 MXN (kcal)", ascending=True),
            x="Calorías por cada $1 MXN (kcal)",
            y="nombre",
            color="categoria",
            orientation="h",
            title="Calorías obtenidas por cada $1 MXN",
        )
        fig_ef_kcal.update_layout(height=450, margin=dict(l=20, r=20, t=40, b=20), yaxis_title="")
        st.plotly_chart(fig_ef_kcal, use_container_width=True)


# =============================================================================
# PESTAÑA 5: VALIDACIÓN MÚLTIPLE Y ESCALAMIENTO (SECCIONES 2.7.6 Y 2.7.7)
# =============================================================================
with tab_validacion:
    st.markdown("### 🧪 Validación en Múltiples Ejecuciones y Prueba de Escalamiento (Secciones 2.7.6 y 2.7.7)")

    v_col1, v_col2 = st.columns(2)

    with v_col1:
        st.markdown("#### 1. Validación Estocástica en Múltiples Ejecuciones (Sección 2.7.6)")
        st.caption(
            "Ejecuta el Algoritmo Genético Simple con distintas semillas aleatorias para "
            "comprobar la consistencia del costo y el cumplimiento nutricional."
        )
        num_corridas = st.slider("Número de ejecuciones independientes", min_value=3, max_value=15, value=5)
        if st.button(" Ejecutar Validación Múltiple", use_container_width=True):
            resultados_mult = []
            for corrida in range(1, num_corridas + 1):
                params_c = ParametrosAGS(
                    tamano_poblacion=tam_poblacion,
                    num_generaciones=num_generaciones,
                    prob_cruce=prob_cruce,
                    tasa_mutacion=tasa_mutacion_pct / 100.0,
                    max_porciones_por_producto=max_porciones,
                    metodo_seleccion=metodo_seleccion,
                    tipo_cruce=tipo_cruce,
                    elitismo=usar_elitismo,
                    detener_por_convergencia=detener_conv,
                    semilla=100 + corrida * 17,
                )
                r_c = ejecutar_algoritmo_genetico_simple(
                    lista_productos=st.session_state["catalogo_productos"],
                    objetivos=objetivos,
                    pesos=pesos,
                    parametros=params_c,
                )
                m_c = r_c["mejor_individuo"]
                resultados_mult.append(
                    {
                        "Ejecución": f"Corrida #{corrida}",
                        "Semilla": 100 + corrida * 17,
                        "Costo ($ MXN)": m_c.costo_total,
                        "Kcal": m_c.kcal_totales,
                        "Proteína (g)": m_c.proteinas_totales,
                        "Carbohidratos (g)": m_c.carbohidratos_totales,
                        "Grasas (g)": m_c.grasas_totales,
                        "Fitness Penalizado": m_c.fitness_penalizado,
                        "Gen. Parada": r_c["generacion_final"],
                        "Tiempo (ms)": round(r_c["tiempo_ejecucion_s"] * 1000, 1),
                    }
                )
            df_mult = pd.DataFrame(resultados_mult)
            st.dataframe(df_mult, use_container_width=True, hide_index=True)
            st.success(
                f"**Costo Promedio:** ${df_mult['Costo ($ MXN)'].mean():.2f} MXN "
                f"(Mín: ${df_mult['Costo ($ MXN)'].min():.2f} · Máx: ${df_mult['Costo ($ MXN)'].max():.2f}) | "
                f"**Fitness Promedio:** {df_mult['Fitness Penalizado'].mean():.2f}"
            )

    with v_col2:
        st.markdown("#### 2. Escalamiento del Catálogo hasta 100 Ingredientes (Sección 2.7.7)")
        st.caption(
            "Prueba la escalabilidad controlada del algoritmo (Sección 2.2 y 2.7.7 del PDF 1) "
            "ampliando el catálogo de 15 → 20 → 50 → 100 productos sin modificar la lógica base."
        )
        if st.button(" Ejecutar Prueba de Escalamiento (15 a 100 Productos)", use_container_width=True):
            tamanos_cat = [15, 20, 50, 100]
            filas_esc = []
            for n_prod in tamanos_cat:
                cat_n = generar_catalogo_escalado(n_prod)
                params_e = ParametrosAGS(
                    tamano_poblacion=tam_poblacion,
                    num_generaciones=num_generaciones,
                    prob_cruce=prob_cruce,
                    tasa_mutacion=tasa_mutacion_pct / 100.0,
                    max_porciones_por_producto=max_porciones,
                    metodo_seleccion=metodo_seleccion,
                    tipo_cruce=tipo_cruce,
                    elitismo=usar_elitismo,
                    detener_por_convergencia=detener_conv,
                    semilla=42,
                )
                r_e = ejecutar_algoritmo_genetico_simple(
                    lista_productos=cat_n,
                    objetivos=objetivos,
                    pesos=pesos,
                    parametros=params_e,
                )
                m_e = r_e["mejor_individuo"]
                filas_esc.append(
                    {
                        "Tamaño Catálogo (Longitud Cromosoma)": f"{n_prod} productos",
                        "Espacio de Búsqueda": f"{max_porciones + 1}^{n_prod}",
                        "Costo Óptimo ($ MXN)": m_e.costo_total,
                        "Kcal": m_e.kcal_totales,
                        "Proteína (g)": m_e.proteinas_totales,
                        "Fitness Penalizado": m_e.fitness_penalizado,
                        "Tiempo Ejecución (ms)": round(r_e["tiempo_ejecucion_s"] * 1000, 1),
                    }
                )
            df_esc = pd.DataFrame(filas_esc)
            st.dataframe(df_esc, use_container_width=True, hide_index=True)

            fig_esc = px.bar(
                df_esc,
                x="Tamaño Catálogo (Longitud Cromosoma)",
                y="Tiempo Ejecución (ms)",
                text="Tiempo Ejecución (ms)",
                color="Costo Óptimo ($ MXN)",
                title="Tiempo de Ejecución (ms) y Costo Óptimo al Escalar de 15 a 100 Productos",
            )
            fig_esc.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_esc, use_container_width=True)

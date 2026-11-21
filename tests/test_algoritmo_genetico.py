"""
tests/test_algoritmo_genetico.py
─────────────────────────────────────────────────────────────────────────────
Pruebas unitarias para verificar las 6 fases del Algoritmo Genético Simple,
el catálogo de 20 productos y el uso de las dos listas.
"""

import random
from catalogo_productos import obtener_catalogo_base, generar_catalogo_escalado
from algoritmo_genetico import (
    ObjetivosNutricionales,
    PesosPenalizacion,
    ParametrosAGS,
    inicializar_poblacion,
    evaluar_individuo,
    evaluar_poblacion,
    seleccionar_padre,
    cruzar_padres,
    mutar_cromosoma,
    reemplazar_poblacion,
    ejecutar_algoritmo_genetico_simple,
)


def test_catalogo_20_productos():
    """Verifica que el catálogo base contenga exactamente 20 productos con los 3 ejemplos del PDF."""
    catalogo = obtener_catalogo_base()
    assert len(catalogo) == 20
    # Verificar los 3 ejemplos de la sección 2.7.2 del PDF
    assert catalogo[0]["nombre"] == "Pechuga de pollo"
    assert catalogo[0]["precio_mxn"] == 15.0
    assert catalogo[0]["kcal"] == 165
    assert catalogo[1]["nombre"] == "Arroz blanco"
    assert catalogo[1]["precio_mxn"] == 4.0
    assert catalogo[2]["nombre"] == "Huevo de gallina"
    assert catalogo[2]["precio_mxn"] == 3.0


def test_seis_fases_y_dos_listas():
    """Verifica Inicialización, Evaluación, Selección, Cruce, Mutación y Reemplazo."""
    rng = random.Random(42)
    lista_productos = obtener_catalogo_base()
    objetivos = ObjetivosNutricionales(2000, 150, 200, 65)
    pesos = PesosPenalizacion()

    # 1. Inicialización
    poblacion_actual = inicializar_poblacion(20, lista_productos, max_porciones=6, rng=rng)
    assert len(poblacion_actual) == 20
    assert all(len(crom) == len(lista_productos) for crom in poblacion_actual)

    # 2. Evaluación
    evaluados = evaluar_poblacion(poblacion_actual, lista_productos, objetivos, pesos)
    assert len(evaluados) == 20
    assert abs(evaluados[-1].prob_acumulada - 1.0) < 1e-6

    # 3. Selección (Ruleta, Torneo, Ranking)
    for metodo in ["ruleta", "torneo", "ranking"]:
        padre, u = seleccionar_padre(evaluados, metodo, rng)
        assert len(padre.cromosoma) == 20

    # 4. Cruce (Crossover)
    h1, h2, punto = cruzar_padres(
        evaluados[0].cromosoma,
        evaluados[1].cromosoma,
        prob_cruce=1.0,
        tipo_cruce="un_punto_mitad",
        rng=rng,
    )
    assert punto == 10
    assert h1 == evaluados[0].cromosoma[:10] + evaluados[1].cromosoma[10:]
    assert h2 == evaluados[1].cromosoma[:10] + evaluados[0].cromosoma[10:]

    # 5. Mutación
    mutado, indices = mutar_cromosoma(h1, tasa_mutacion=0.5, max_porciones=6, rng=rng)
    assert len(mutado) == 20
    assert all(0 <= g <= 6 for g in mutado)

    # 6. Reemplazo
    nueva_poblacion_eval = evaluar_poblacion([mutado] * 20, lista_productos, objetivos, pesos)
    reemplazada = reemplazar_poblacion(evaluados, nueva_poblacion_eval, elitismo=True)
    assert len(reemplazada) == 20


def test_convergencia_menu_optimo():
    """Verifica que el AGS converja a una solución que cumpla los objetivos nutricionales."""
    catalogo = obtener_catalogo_base()
    objetivos = ObjetivosNutricionales(2000, 150, 200, 65)
    pesos = PesosPenalizacion()
    params = ParametrosAGS(tamano_poblacion=80, num_generaciones=150, semilla=42)

    res = ejecutar_algoritmo_genetico_simple(catalogo, objetivos, pesos, params)
    mejor = res["mejor_individuo"]

    assert len(res["lista_productos"]) == 20
    assert len(res["lista_porciones_optima"]) == 20
    assert mejor.costo_total > 0
    # El fitness final debe ser mucho mejor (menor) que el fitness inicial
    assert res["historial"][-1]["mejor_fitness"] < res["historial"][0]["mejor_fitness"]
    # Desviaciones nutricionales bajas
    assert mejor.desviacion_kcal < 60
    assert mejor.desviacion_proteinas < 10


def test_escalamiento_100_ingredientes():
    """Verifica que el catálogo escale a 100 ingredientes sin cambiar la lógica base."""
    cat_100 = generar_catalogo_escalado(100)
    assert len(cat_100) == 100
    res = ejecutar_algoritmo_genetico_simple(
        cat_100,
        ObjetivosNutricionales(2000, 150, 200, 65),
        PesosPenalizacion(),
        ParametrosAGS(tamano_poblacion=40, num_generaciones=40, semilla=7),
    )
    assert len(res["mejor_individuo"].cromosoma) == 100

"""
algoritmo_genetico.py
─────────────────────────────────────────────────────────────────────────────
Implementación del Algoritmo Genético Simple (Canónico) para el:
"Sistema de optimización de planes de alimentación diarios de mínimo costo
mediante un Algoritmo Genético Simple".

Basado en:
1. Generalidades del Proyecto (Secciones 2.1 a 2.7)
2. Fundamentos de Algoritmos Genéticos (Algoritmo Genético Simple / Canónico,
   Figura 6.1, Tablas 6.1 y 6.2, Secciones 2 y 3).

Estructura de Dos Listas:
- A nivel de representación del problema:
  1) `lista_productos`: Lista de N productos (20 por defecto) con precio y nutrientes.
  2) `lista_porciones` (Cromosoma): Lista de N enteros [p_0, p_1, ..., p_{N-1}]
     donde la posición i representa el número de porciones del producto i.
- A nivel del ciclo evolutivo generacional:
  1) `poblacion_actual`: Lista de individuos (padres) en la generación t.
  2) `nueva_poblacion`: Lista de individuos (hijos) generados para la generación t+1
     que reemplaza a `poblacion_actual` en la fase de Reemplazo.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional
import random
import time
import copy


@dataclass
class ObjetivosNutricionales:
    """Requerimientos nutricionales diarios definidos por el usuario."""
    kcal: float = 2000.0
    proteinas_g: float = 150.0
    carbohidratos_g: float = 200.0
    grasas_g: float = 65.0


@dataclass
class PesosPenalizacion:
    """
    Pesos de penalización por cada unidad de desviación respecto al objetivo
    nutricional (Sección 2.5.1 y 2.7.3 del PDF Generalidades).
    """
    peso_kcal: float = 2.5
    peso_proteinas: float = 15.0
    peso_carbohidratos: float = 10.0
    peso_grasas: float = 12.0


@dataclass
class ParametrosAGS:
    """Parámetros de configuración del Algoritmo Genético Simple."""
    tamano_poblacion: int = 120
    num_generaciones: int = 250
    prob_cruce: float = 0.85
    tasa_mutacion: float = 0.05  # ~5% como indica Sección 2.3.2 y 2.7.4
    max_porciones_por_producto: int = 6
    metodo_seleccion: str = "ruleta"  # "ruleta", "torneo", "ranking"
    tipo_cruce: str = "un_punto_mitad"  # "un_punto_mitad", "un_punto_aleatorio", "dos_puntos", "uniforme"
    tamano_torneo: int = 3
    elitismo: bool = True  # Garantiza conservar al mejor individuo (Rudolph 1994, Sección 4.3 PDF 2)
    detener_por_convergencia: bool = True
    umbral_convergencia_genes: float = 0.95  # 95% criterio De Jong (1975) pág. 4 PDF 2
    paciencia_convergencia: int = 100
    semilla: Optional[int] = None


@dataclass
class IndividuoEvaluado:
    """
    Representa un individuo (cromosoma) junto con su evaluación nutricional,
    su costo económico y sus métricas de adaptación para el AGS.
    """
    id_individuo: int
    cromosoma: List[int]  # Lista de enteros (porciones por cada producto de lista_productos)
    costo_total: float = 0.0
    kcal_totales: float = 0.0
    proteinas_totales: float = 0.0
    carbohidratos_totales: float = 0.0
    grasas_totales: float = 0.0
    desviacion_kcal: float = 0.0
    desviacion_proteinas: float = 0.0
    desviacion_carbohidratos: float = 0.0
    desviacion_grasas: float = 0.0
    penalizacion_total: float = 0.0
    fitness_penalizado: float = 0.0  # f(x) = Costo + Penalizaciones (menor / cercano a 0 es mejor)
    aptitud: float = 0.0             # g(x) = 10000 / (1 + f(x)) (mayor es mejor, para ruleta)
    prob_seleccion: float = 0.0      # f(x) / sum(f(x)) (Tabla 6.1 PDF 2)
    prob_acumulada: float = 0.0      # Probabilidad acumulada (Tabla 6.1 PDF 2)


# ─────────────────────────────────────────────────────────────────────────────
# 1. INICIALIZACIÓN
# ─────────────────────────────────────────────────────────────────────────────
def generar_cromosoma_aleatorio(
    num_productos: int,
    max_porciones: int,
    rng: random.Random,
    objetivo_kcal: float = 2000.0,
) -> List[int]:
    """
    Fase 1 (Inicialización): Genera un individuo (cromosoma) aleatorio como
    una lista de enteros de tamaño fijo `num_productos`.
    Cada posición corresponde a un producto de `lista_productos` y su valor
    a la cantidad de porciones enteras en [0, max_porciones].
    """
    cromosoma = [0] * num_productos
    # Estimamos cuántas porciones activas requiere un día típico (~120 kcal promedio por porción)
    porciones_estimadas = max(4, int(round(objetivo_kcal / 120.0)))
    # Variamos la densidad inicial aleatoriamente para cubrir todo el espacio de búsqueda
    modo = rng.random()
    if modo < 0.70:
        # Selección aleatoria de ingredientes activos hasta acercarse al rango energético diario
        intentos = rng.randint(max(3, porciones_estimadas - 6), porciones_estimadas + 8)
        for _ in range(intentos):
            idx = rng.randint(0, num_productos - 1)
            if cromosoma[idx] < max_porciones:
                cromosoma[idx] += 1
    else:
        # Generación aleatoria uniforme por gen con probabilidad de inclusión
        prob_activo = min(0.65, max(0.20, (porciones_estimadas / (num_productos * 2.0))))
        for i in range(num_productos):
            if rng.random() < prob_activo:
                cromosoma[i] = rng.randint(1, min(max_porciones, 4))
    return cromosoma


def inicializar_poblacion(
    tamano_poblacion: int,
    lista_productos: List[Dict[str, Any]],
    max_porciones: int,
    rng: random.Random,
    objetivo_kcal: float = 2000.0,
) -> List[List[int]]:
    """
    Fase 1: Inicialización.
    Genera la lista `poblacion_actual` con `tamano_poblacion` cromosomas aleatorios.
    """
    num_productos = len(lista_productos)
    poblacion_actual: List[List[int]] = [
        generar_cromosoma_aleatorio(num_productos, max_porciones, rng, objetivo_kcal)
        for _ in range(tamano_poblacion)
    ]
    return poblacion_actual


# ─────────────────────────────────────────────────────────────────────────────
# 2. EVALUACIÓN (FUNCIÓN DE APTITUD / FITNESS)
# ─────────────────────────────────────────────────────────────────────────────
def evaluar_individuo(
    cromosoma: List[int],
    lista_productos: List[Dict[str, Any]],
    objetivos: ObjetivosNutricionales,
    pesos: PesosPenalizacion,
    id_individuo: int = 0,
) -> IndividuoEvaluado:
    """
    Fase 2: Evaluación.
    Relaciona las DOS LISTAS principales del modelo:
      - `lista_productos`: catálogo con precios y nutrientes de cada producto.
      - `cromosoma` (`lista_porciones`): cantidad de porciones enteras de cada producto.

    Calcula:
      - Costo total (MXN) y suma de macronutrientes (kcal, proteínas, carbohidratos, grasas).
      - Fitness penalizado (Sección 2.7.3 PDF 1):
          f(x) = Costo_Total + Penalizaciones_por_desviación (más cercano a 0 es mejor).
      - Función de adaptación / aptitud (Sección 3.2 y Tabla 6.1 PDF 2):
          g(x) = 10000 / (1 + f(x)) (mayor valor indica mayor adaptación para la ruleta).
    """
    costo_total = 0.0
    kcal_totales = 0.0
    proteinas_totales = 0.0
    carbohidratos_totales = 0.0
    grasas_totales = 0.0

    # Recorrido paralelo de las dos listas: lista_productos y cromosoma (lista_porciones)
    for producto, porciones in zip(lista_productos, cromosoma):
        if porciones > 0:
            costo_total += porciones * float(producto["precio_mxn"])
            kcal_totales += porciones * float(producto["kcal"])
            proteinas_totales += porciones * float(producto["proteinas_g"])
            carbohidratos_totales += porciones * float(producto["carbohidratos_g"])
            grasas_totales += porciones * float(producto["grasas_g"])

    desv_kcal = abs(objetivos.kcal - kcal_totales)
    desv_prot = abs(objetivos.proteinas_g - proteinas_totales)
    desv_carb = abs(objetivos.carbohidratos_g - carbohidratos_totales)
    desv_grasa = abs(objetivos.grasas_g - grasas_totales)

    penalizacion = (
        desv_kcal * pesos.peso_kcal
        + desv_prot * pesos.peso_proteinas
        + desv_carb * pesos.peso_carbohidratos
        + desv_grasa * pesos.peso_grasas
    )

    # Si el menú está completamente vacío (0 porciones), penalizar fuertemente
    if sum(cromosoma) == 0:
        penalizacion += 50000.0

    fitness_penalizado = costo_total + penalizacion
    aptitud = 10000.0 / (1.0 + fitness_penalizado)

    return IndividuoEvaluado(
        id_individuo=id_individuo,
        cromosoma=list(cromosoma),
        costo_total=round(costo_total, 2),
        kcal_totales=round(kcal_totales, 2),
        proteinas_totales=round(proteinas_totales, 2),
        carbohidratos_totales=round(carbohidratos_totales, 2),
        grasas_totales=round(grasas_totales, 2),
        desviacion_kcal=round(desv_kcal, 2),
        desviacion_proteinas=round(desv_prot, 2),
        desviacion_carbohidratos=round(desv_carb, 2),
        desviacion_grasas=round(desv_grasa, 2),
        penalizacion_total=round(penalizacion, 2),
        fitness_penalizado=round(fitness_penalizado, 2),
        aptitud=aptitud,
    )


def evaluar_poblacion(
    poblacion: List[List[int]],
    lista_productos: List[Dict[str, Any]],
    objetivos: ObjetivosNutricionales,
    pesos: PesosPenalizacion,
) -> List[IndividuoEvaluado]:
    """
    Evalúa todos los individuos de la población y calcula su probabilidad de
    selección proporcional a la aptitud f(x)/sum(f(x)) y su probabilidad
    acumulada, tal como se especifica en la Tabla 6.1 del PDF de Algoritmos Genéticos.
    """
    evaluados: List[IndividuoEvaluado] = [
        evaluar_individuo(crom, lista_productos, objetivos, pesos, id_individuo=i + 1)
        for i, crom in enumerate(poblacion)
    ]

    suma_aptitud = sum(ind.aptitud for ind in evaluados)
    acumulada = 0.0
    n = len(evaluados)

    for ind in evaluados:
        if suma_aptitud > 0:
            ind.prob_seleccion = ind.aptitud / suma_aptitud
        else:
            ind.prob_seleccion = 1.0 / n if n > 0 else 0.0
        acumulada += ind.prob_seleccion
        ind.prob_acumulada = min(1.0, acumulada)

    if evaluados:
        evaluados[-1].prob_acumulada = 1.0

    return evaluados


# ─────────────────────────────────────────────────────────────────────────────
# 3. SELECCIÓN (RULETA, TORNEO, RANKING)
# ─────────────────────────────────────────────────────────────────────────────
def seleccion_ruleta(
    poblacion_evaluada: List[IndividuoEvaluado],
    rng: random.Random,
) -> Tuple[IndividuoEvaluado, float]:
    """
    Selección por Ruleta Sesgada (Proporcional a la función de adaptación).
    PDF 2, Sección 3.3 y Tabla 6.1: Cada individuo tiene probabilidad p_j = g(I_j)/sum(g).
    Se extrae un número aleatorio u en [0, 1] y se compara con la columna de
    probabilidad acumulada.
    """
    u = rng.random()
    for ind in poblacion_evaluada:
        if u <= ind.prob_acumulada:
            return ind, u
    return poblacion_evaluada[-1], u


def seleccion_torneo(
    poblacion_evaluada: List[IndividuoEvaluado],
    rng: random.Random,
    tamano_torneo: int = 3,
) -> Tuple[IndividuoEvaluado, float]:
    """
    Selección por Torneo (PDF 2, Sección 3.3, pág. 10):
    Escoge al azar `tamano_torneo` individuos de la población y selecciona
    al que posee la mejor adaptación (menor fitness_penalizado / mayor aptitud).
    """
    k = min(max(2, tamano_torneo), len(poblacion_evaluada))
    competidores = rng.sample(poblacion_evaluada, k)
    ganador = min(competidores, key=lambda ind: ind.fitness_penalizado)
    return ganador, ganador.prob_seleccion


def seleccion_ranking(
    poblacion_evaluada: List[IndividuoEvaluado],
    rng: random.Random,
) -> Tuple[IndividuoEvaluado, float]:
    """
    Selección proporcional al Rango (Ranking, PDF 2, Sección 3.3, Figura 6.6, pág. 9):
    Los individuos se ordenan de peor a mejor adaptación; el peor recibe rango 1
    y el mejor recibe rango lambda. La probabilidad de selección es:
        p_j^rango = rango(g(I_j)) / (lambda * (lambda + 1) / 2)
    """
    lam = len(poblacion_evaluada)
    ordenados_peor_a_mejor = sorted(
        poblacion_evaluada, key=lambda ind: ind.aptitud
    )
    constante_norm = lam * (lam + 1) / 2.0
    u = rng.random()
    acum = 0.0
    for rango, ind in enumerate(ordenados_peor_a_mejor, start=1):
        acum += rango / constante_norm
        if u <= acum:
            return ind, u
    return ordenados_peor_a_mejor[-1], u


def seleccionar_padre(
    poblacion_evaluada: List[IndividuoEvaluado],
    metodo: str,
    rng: random.Random,
    tamano_torneo: int = 3,
) -> Tuple[IndividuoEvaluado, float]:
    """Fase 3: Selección de un padre según el método configurado."""
    metodo_norm = metodo.lower().strip()
    if metodo_norm == "torneo":
        return seleccion_torneo(poblacion_evaluada, rng, tamano_torneo)
    elif metodo_norm == "ranking":
        return seleccion_ranking(poblacion_evaluada, rng)
    else:
        return seleccion_ruleta(poblacion_evaluada, rng)


# ─────────────────────────────────────────────────────────────────────────────
# 4. CRUCE (CROSSOVER)
# ─────────────────────────────────────────────────────────────────────────────
def cruzar_padres(
    padre_1: List[int],
    padre_2: List[int],
    prob_cruce: float,
    tipo_cruce: str,
    rng: random.Random,
) -> Tuple[List[int], List[int], int]:
    """
    Fase 4: Cruce (Crossover).
    Combina las listas (cromosomas) de dos padres para generar dos hijos,
    intercambiando segmentos de información genética con probabilidad `prob_cruce`.
    Devuelve (hijo_1, hijo_2, punto_cruce).
    """
    longitud = len(padre_1)
    if longitud < 2 or rng.random() > prob_cruce:
        # Si no se aplica cruce, los descendientes duplican a los padres (PDF 2, pág. 3)
        return list(padre_1), list(padre_2), 0

    tipo = tipo_cruce.lower().strip()
    if tipo == "un_punto_mitad":
        # Sección 2.7.4 PDF 1: "corte por la mitad e intercambio de segmentos entre padres"
        punto = longitud // 2
        hijo_1 = padre_1[:punto] + padre_2[punto:]
        hijo_2 = padre_2[:punto] + padre_1[punto:]
        return hijo_1, hijo_2, punto

    elif tipo == "dos_puntos":
        # Sección 3.4 PDF 2 (Figura 6.9): Cruce basado en dos puntos
        p1, p2 = sorted(rng.sample(range(1, longitud), 2))
        hijo_1 = padre_1[:p1] + padre_2[p1:p2] + padre_1[p2:]
        hijo_2 = padre_2[:p1] + padre_1[p1:p2] + padre_2[p2:]
        return hijo_1, hijo_2, p1

    elif tipo == "uniforme":
        # Sección 3.4 PDF 2 (Figura 6.10): Cruce uniforme con máscara de Bernoulli(1/2)
        hijo_1, hijo_2 = [], []
        for g1, g2 in zip(padre_1, padre_2):
            if rng.random() < 0.5:
                hijo_1.append(g1)
                hijo_2.append(g2)
            else:
                hijo_1.append(g2)
                hijo_2.append(g1)
        return hijo_1, hijo_2, longitud // 2

    else:
        # Cruce canónico de 1 punto aleatorio entre 1 y l-1 (PDF 2, Sección 2.2 y Ecuaciones 1-2)
        punto = rng.randint(1, longitud - 1)
        hijo_1 = padre_1[:punto] + padre_2[punto:]
        hijo_2 = padre_2[:punto] + padre_1[punto:]
        return hijo_1, hijo_2, punto


# ─────────────────────────────────────────────────────────────────────────────
# 5. MUTACIÓN
# ─────────────────────────────────────────────────────────────────────────────
def mutar_cromosoma(
    cromosoma: List[int],
    tasa_mutacion: float,
    max_porciones: int,
    rng: random.Random,
) -> Tuple[List[int], List[int]]:
    """
    Fase 5: Mutación.
    Altera aleatoriamente los genes del cromosoma con una baja probabilidad
    (`tasa_mutacion`, ej. ~5% = 0.05) para mantener la diversidad genética
    y evitar óptimos locales.
    Sección 2.7.4 PDF 1: "con ~5 % de probabilidad se suma o resta una porción a un ingrediente".
    Devuelve (cromosoma_mutado, indices_genes_mutados).
    """
    mutado = list(cromosoma)
    genes_mutados: List[int] = []

    for i in range(len(mutado)):
        if rng.random() < tasa_mutacion:
            genes_mutados.append(i)
            valor_actual = mutado[i]
            if valor_actual == 0:
                mutado[i] = 1
            elif valor_actual >= max_porciones:
                mutado[i] = max_porciones - 1
            else:
                # Suma o resta 1 porción (o con 15% de probabilidad apaga el ingrediente a 0)
                if rng.random() < 0.15:
                    mutado[i] = 0
                else:
                    delta = 1 if rng.random() < 0.5 else -1
                    mutado[i] = max(0, min(max_porciones, valor_actual + delta))

    return mutado, genes_mutados


# ─────────────────────────────────────────────────────────────────────────────
# 6. REEMPLAZO Y CRITERIO DE CONVERGENCIA
# ─────────────────────────────────────────────────────────────────────────────
def calcular_convergencia_genetica(poblacion: List[List[int]]) -> float:
    """
    Criterio práctico de convergencia de De Jong (1975) (PDF 2, Sección 2.1, pág. 4):
    Mide la progresión hacia la uniformidad genética de la población.
    Para cada gen j, calcula la proporción máxima de individuos que comparten
    el mismo valor y devuelve el promedio de uniformidad sobre todos los genes.
    """
    if not poblacion or not poblacion[0]:
        return 0.0
    tam_pob = len(poblacion)
    longitud = len(poblacion[0])
    suma_proporciones = 0.0

    for j in range(longitud):
        conteos: Dict[int, int] = {}
        for ind in poblacion:
            val = ind[j]
            conteos[val] = conteos.get(val, 0) + 1
        max_frecuencia = max(conteos.values())
        suma_proporciones += max_frecuencia / tam_pob

    return suma_proporciones / longitud


def reemplazar_poblacion(
    poblacion_actual_eval: List[IndividuoEvaluado],
    nueva_poblacion_eval: List[IndividuoEvaluado],
    elitismo: bool = True,
) -> List[IndividuoEvaluado]:
    """
    Fase 6: Reemplazo (Reducción, Sección 3.6 y 4.3 del PDF 2).
    La lista `nueva_poblacion` reemplaza a la lista `poblacion_actual`.
    Si `elitismo=True`, se preserva al mejor individuo de la generación anterior
    en caso de que ningún descendiente lo haya superado (garantía de convergencia
    de Rudolph 1994, Sección 4.3 PDF 2).
    """
    siguiente_generacion = list(nueva_poblacion_eval)

    if elitismo and poblacion_actual_eval and siguiente_generacion:
        mejor_anterior = min(poblacion_actual_eval, key=lambda x: x.fitness_penalizado)
        mejor_nuevo = min(siguiente_generacion, key=lambda x: x.fitness_penalizado)

        if mejor_anterior.fitness_penalizado < mejor_nuevo.fitness_penalizado:
            # Reemplaza al peor individuo de la nueva generación por el mejor de la anterior
            idx_peor = max(
                range(len(siguiente_generacion)),
                key=lambda i: siguiente_generacion[i].fitness_penalizado,
            )
            siguiente_generacion[idx_peor] = copy.deepcopy(mejor_anterior)

    # Recalcular probabilidades de selección y acumuladas para la nueva generación
    suma_aptitud = sum(ind.aptitud for ind in siguiente_generacion)
    acum = 0.0
    n = len(siguiente_generacion)
    for idx, ind in enumerate(siguiente_generacion):
        ind.id_individuo = idx + 1
        ind.prob_seleccion = (ind.aptitud / suma_aptitud) if suma_aptitud > 0 else (1.0 / n)
        acum += ind.prob_seleccion
        ind.prob_acumulada = min(1.0, acum)
    if siguiente_generacion:
        siguiente_generacion[-1].prob_acumulada = 1.0

    return siguiente_generacion


# ─────────────────────────────────────────────────────────────────────────────
# CICLO PRINCIPAL DEL ALGORITMO GENÉTICO SIMPLE (FIGURA 6.1 PDF 2)
# ─────────────────────────────────────────────────────────────────────────────
def ejecutar_algoritmo_genetico_simple(
    lista_productos: List[Dict[str, Any]],
    objetivos: ObjetivosNutricionales,
    pesos: PesosPenalizacion,
    parametros: ParametrosAGS,
) -> Dict[str, Any]:
    """
    Ejecuta el Algoritmo Genético Simple completo siguiendo el pseudocódigo
    de la Figura 6.1 del PDF de Algoritmos Genéticos y las 6 fases del proyecto:
      1. Inicialización
      2. Evaluación
      3. Selección
      4. Cruce (Crossover)
      5. Mutación
      6. Reemplazo (hasta cumplir criterio de parada)
    """
    t_inicio = time.perf_counter()
    rng = random.Random(parametros.semilla)

    # Asegurar que el tamaño de población sea par para el ciclo FOR Tamaño_poblacion/2 DO
    tam_pob = max(4, parametros.tamano_poblacion)
    if tam_pob % 2 != 0:
        tam_pob += 1

    # ── PASO 1: INICIALIZACIÓN ───────────────────────────────────────────────
    # Lista A del ciclo evolutivo: poblacion_actual (cromosomas de la generación t)
    poblacion_actual: List[List[int]] = inicializar_poblacion(
        tamano_poblacion=tam_pob,
        lista_productos=lista_productos,
        max_porciones=parametros.max_porciones_por_producto,
        rng=rng,
        objetivo_kcal=objetivos.kcal,
    )

    # ── PASO 2: EVALUACIÓN DE LA POBLACIÓN INICIAL ───────────────────────────
    poblacion_evaluada = evaluar_poblacion(
        poblacion=poblacion_actual,
        lista_productos=lista_productos,
        objetivos=objetivos,
        pesos=pesos,
    )

    # Guardar instantánea de la Población Inicial (Tabla 6.1 del PDF 2, Tiempo t=0)
    tabla_tiempo_0 = copy.deepcopy(poblacion_evaluada)
    tabla_tiempo_1_operadores: List[Dict[str, Any]] = []

    mejor_global = copy.deepcopy(
        min(poblacion_evaluada, key=lambda ind: ind.fitness_penalizado)
    )

    historial_generaciones: List[Dict[str, Any]] = []
    suma_acumulada_online_fitness = 0.0
    suma_acumulada_offline_fitness = 0.0
    suma_acumulada_online_aptitud = 0.0
    suma_acumulada_offline_aptitud = 0.0

    generaciones_sin_mejora = 0
    motivo_parada = f"Se alcanzó el máximo de {parametros.num_generaciones} generaciones"
    generacion_final = 0

    # Registro de la generación 0
    mejor_gen_0 = min(poblacion_evaluada, key=lambda x: x.fitness_penalizado)
    media_fitness_0 = sum(x.fitness_penalizado for x in poblacion_evaluada) / tam_pob
    media_aptitud_0 = sum(x.aptitud for x in poblacion_evaluada) / tam_pob
    conv_0 = calcular_convergencia_genetica(poblacion_actual)

    suma_acumulada_online_fitness += media_fitness_0
    suma_acumulada_offline_fitness += mejor_global.fitness_penalizado
    suma_acumulada_online_aptitud += media_aptitud_0
    suma_acumulada_offline_aptitud += mejor_global.aptitud

    historial_generaciones.append(
        {
            "generacion": 0,
            "mejor_fitness": mejor_global.fitness_penalizado,
            "fitness_medio": round(media_fitness_0, 2),
            "mejor_aptitud": mejor_global.aptitud,
            "aptitud_media": media_aptitud_0,
            "mejor_costo_mxn": mejor_global.costo_total,
            "mejor_kcal": mejor_global.kcal_totales,
            "mejor_proteinas": mejor_global.proteinas_totales,
            "mejor_carbohidratos": mejor_global.carbohidratos_totales,
            "mejor_grasas": mejor_global.grasas_totales,
            "convergencia_genes": round(conv_0 * 100.0, 2),
            "dejong_online_fitness": round(suma_acumulada_online_fitness, 2),
            "dejong_offline_fitness": round(suma_acumulada_offline_fitness, 2),
            "dejong_online_aptitud": suma_acumulada_online_aptitud,
            "dejong_offline_aptitud": suma_acumulada_offline_aptitud,
        }
    )

    # ── CICLO EVOLUTIVO: WHILE NOT Terminado DO ──────────────────────────────
    for gen in range(1, parametros.num_generaciones + 1):
        generacion_final = gen
        # Lista B del ciclo evolutivo: nueva_poblacion (descendientes para t+1)
        nueva_poblacion: List[List[int]] = []

        # FOR Tamaño_poblacion / 2 DO (Ciclo Reproductivo de Figura 6.1 PDF 2)
        for pareja_idx in range(tam_pob // 2):
            # ── PASO 3: SELECCIÓN ────────────────────────────────────────────
            padre_1_eval, u1 = seleccionar_padre(
                poblacion_evaluada,
                parametros.metodo_seleccion,
                rng,
                parametros.tamano_torneo,
            )
            padre_2_eval, u2 = seleccionar_padre(
                poblacion_evaluada,
                parametros.metodo_seleccion,
                rng,
                parametros.tamano_torneo,
            )

            # ── PASO 4: CRUCE (CROSSOVER) ────────────────────────────────────
            hijo_1_cruzado, hijo_2_cruzado, punto_cruce = cruzar_padres(
                padre_1=padre_1_eval.cromosoma,
                padre_2=padre_2_eval.cromosoma,
                prob_cruce=parametros.prob_cruce,
                tipo_cruce=parametros.tipo_cruce,
                rng=rng,
            )

            # ── PASO 5: MUTACIÓN ─────────────────────────────────────────────
            hijo_1_mutado, genes_mut_1 = mutar_cromosoma(
                cromosoma=hijo_1_cruzado,
                tasa_mutacion=parametros.tasa_mutacion,
                max_porciones=parametros.max_porciones_por_producto,
                rng=rng,
            )
            hijo_2_mutado, genes_mut_2 = mutar_cromosoma(
                cromosoma=hijo_2_cruzado,
                tasa_mutacion=parametros.tasa_mutacion,
                max_porciones=parametros.max_porciones_por_producto,
                rng=rng,
            )

            # Insertar los dos descendientes mutados en la nueva generación
            nueva_poblacion.append(hijo_1_mutado)
            nueva_poblacion.append(hijo_2_mutado)

            # Registrar detalle de la primera generación (Tabla 6.2 del PDF 2, Tiempo t=1)
            if gen == 1:
                eval_h1 = evaluar_individuo(hijo_1_mutado, lista_productos, objetivos, pesos)
                eval_h2 = evaluar_individuo(hijo_2_mutado, lista_productos, objetivos, pesos)
                tabla_tiempo_1_operadores.append(
                    {
                        "pareja": pareja_idx + 1,
                        "id_padre": padre_1_eval.id_individuo,
                        "aleatorio_u": round(u1, 3),
                        "cromosoma_padre": list(padre_1_eval.cromosoma),
                        "punto_cruce": punto_cruce,
                        "descendiente_cruzado": list(hijo_1_cruzado),
                        "descendiente_mutado": list(hijo_1_mutado),
                        "genes_mutados": genes_mut_1,
                        "costo_mxn": eval_h1.costo_total,
                        "fitness_penalizado": eval_h1.fitness_penalizado,
                        "aptitud": eval_h1.aptitud,
                    }
                )
                tabla_tiempo_1_operadores.append(
                    {
                        "pareja": pareja_idx + 1,
                        "id_padre": padre_2_eval.id_individuo,
                        "aleatorio_u": round(u2, 3),
                        "cromosoma_padre": list(padre_2_eval.cromosoma),
                        "punto_cruce": punto_cruce,
                        "descendiente_cruzado": list(hijo_2_cruzado),
                        "descendiente_mutado": list(hijo_2_mutado),
                        "genes_mutados": genes_mut_2,
                        "costo_mxn": eval_h2.costo_total,
                        "fitness_penalizado": eval_h2.fitness_penalizado,
                        "aptitud": eval_h2.aptitud,
                    }
                )

        # Evaluar la nueva población de descendientes
        nueva_poblacion_evaluada = evaluar_poblacion(
            poblacion=nueva_poblacion,
            lista_productos=lista_productos,
            objetivos=objetivos,
            pesos=pesos,
        )

        # ── PASO 6: REEMPLAZO ────────────────────────────────────────────────
        poblacion_evaluada = reemplazar_poblacion(
            poblacion_actual_eval=poblacion_evaluada,
            nueva_poblacion_eval=nueva_poblacion_evaluada,
            elitismo=parametros.elitismo,
        )
        poblacion_actual = [ind.cromosoma for ind in poblacion_evaluada]

        # Actualizar el mejor individuo global y métricas de la generación
        mejor_gen = min(poblacion_evaluada, key=lambda x: x.fitness_penalizado)
        if mejor_gen.fitness_penalizado < mejor_global.fitness_penalizado - 1e-6:
            mejor_global = copy.deepcopy(mejor_gen)
            generaciones_sin_mejora = 0
        else:
            generaciones_sin_mejora += 1

        media_fitness = sum(x.fitness_penalizado for x in poblacion_evaluada) / tam_pob
        media_aptitud = sum(x.aptitud for x in poblacion_evaluada) / tam_pob
        convergencia_genes = calcular_convergencia_genetica(poblacion_actual)

        # Métricas de De Jong (1975) - Sección 3.8 PDF 2 (pág. 14-15)
        suma_acumulada_online_fitness += media_fitness
        suma_acumulada_offline_fitness += mejor_global.fitness_penalizado
        suma_acumulada_online_aptitud += media_aptitud
        suma_acumulada_offline_aptitud += mejor_global.aptitud

        t_steps = gen + 1
        historial_generaciones.append(
            {
                "generacion": gen,
                "mejor_fitness": mejor_global.fitness_penalizado,
                "fitness_medio": round(media_fitness, 2),
                "mejor_aptitud": mejor_global.aptitud,
                "aptitud_media": media_aptitud,
                "mejor_costo_mxn": mejor_global.costo_total,
                "mejor_kcal": mejor_global.kcal_totales,
                "mejor_proteinas": mejor_global.proteinas_totales,
                "mejor_carbohidratos": mejor_global.carbohidratos_totales,
                "mejor_grasas": mejor_global.grasas_totales,
                "convergencia_genes": round(convergencia_genes * 100.0, 2),
                "dejong_online_fitness": round(suma_acumulada_online_fitness / t_steps, 2),
                "dejong_offline_fitness": round(suma_acumulada_offline_fitness / t_steps, 2),
                "dejong_online_aptitud": suma_acumulada_online_aptitud / t_steps,
                "dejong_offline_aptitud": suma_acumulada_offline_aptitud / t_steps,
            }
        )

        # Verificar criterio de parada por convergencia (IF la poblacion ha convergido THEN Terminado := TRUE)
        if parametros.detener_por_convergencia:
            if convergencia_genes >= parametros.umbral_convergencia_genes:
                motivo_parada = (
                    f"Convergencia genética alcanzada ({convergencia_genes*100:.1f}% >= "
                    f"{parametros.umbral_convergencia_genes*100:.0f}%, Criterio de De Jong) "
                    f"en la generación {gen}"
                )
                break
            if generaciones_sin_mejora >= parametros.paciencia_convergencia and gen >= 30:
                motivo_parada = (
                    f"Convergencia por estabilidad ({generaciones_sin_mejora} generaciones "
                    f"consecutivas sin cambio en el óptimo) en la generación {gen}"
                )
                break

    tiempo_ejecucion_s = time.perf_counter() - t_inicio

    # Construir desglose del menú diario a partir de las DOS LISTAS (lista_productos y mejor_global.cromosoma)
    desglose_menu: List[Dict[str, Any]] = []
    mapeo_dos_listas: List[Dict[str, Any]] = []

    for producto, porciones in zip(lista_productos, mejor_global.cromosoma):
        subtotal_mxn = round(porciones * float(producto["precio_mxn"]), 2)
        kcal_item = round(porciones * float(producto["kcal"]), 1)
        prot_item = round(porciones * float(producto["proteinas_g"]), 1)
        carb_item = round(porciones * float(producto["carbohidratos_g"]), 1)
        grasa_item = round(porciones * float(producto["grasas_g"]), 1)

        fila = {
            "id": producto["id"],
            "producto": producto["nombre"],
            "categoria": producto.get("categoria", "General"),
            "porcion_unitaria": producto["porcion_desc"],
            "precio_unitario_mxn": float(producto["precio_mxn"]),
            "porciones": int(porciones),
            "subtotal_mxn": subtotal_mxn,
            "kcal": kcal_item,
            "proteinas_g": prot_item,
            "carbohidratos_g": carb_item,
            "grasas_g": grasa_item,
        }
        mapeo_dos_listas.append(fila)
        if porciones > 0:
            desglose_menu.append(fila)

    return {
        "mejor_individuo": mejor_global,
        "desglose_menu": desglose_menu,
        "mapeo_dos_listas": mapeo_dos_listas,
        "historial": historial_generaciones,
        "tabla_tiempo_0": tabla_tiempo_0,
        "tabla_tiempo_1": tabla_tiempo_1_operadores,
        "poblacion_final": poblacion_evaluada,
        "generacion_final": generacion_final,
        "motivo_parada": motivo_parada,
        "tiempo_ejecucion_s": round(tiempo_ejecucion_s, 4),
        "lista_productos": lista_productos,
        "lista_porciones_optima": list(mejor_global.cromosoma),
    }

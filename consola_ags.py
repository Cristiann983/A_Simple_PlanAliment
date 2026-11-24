"""
consola_ags.py
─────────────────────────────────────────────────────────────────────────────
Ejecución interactiva en CONSOLA del Algoritmo Genético Simple (Canónico)
para la optimización de planes de alimentación diarios de mínimo costo.

Características:
- Captura MANUAL por consola de los objetivos nutricionales y de los
  parámetros principales del AGS (población, generaciones, selección,
  cruce, probabilidad de cruce, tasa de mutación y máx. porciones).
- Los pesos de penalización y los criterios de parada están preconfigurados
  automáticamente (no se solicitan manualmente).
- Muestra paso a paso en consola las Dos Listas y las 6 fases del proceso:
  1. Inicialización
  2. Evaluación (Tabla 6.1 en t=0)
  3. Selección
  4. Cruce (Crossover)
  5. Mutación (Tabla 6.2 en t=1)
  6. Reemplazo y Evolución generacional hasta el criterio de parada.
"""

from __future__ import annotations
import sys
import random
import time
import copy
from typing import List, Dict, Any

from catalogo_productos import obtener_catalogo_base
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
    calcular_convergencia_genetica,
)

# Asegurar salida UTF-8 en consolas de Windows (PowerShell / CMD)
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


LINEA_DOBLE = "=" * 96
LINEA_SIMPLE = "-" * 96


def pedir_float(mensaje: str, minimo: float, maximo: float, sugerido: float) -> float:
    """Solicita un número decimal validado por consola (Enter usa el valor sugerido)."""
    while True:
        entrada = input(f"  > {mensaje} [{minimo} - {maximo}] (Enter = {sugerido}): ").lstrip("\ufeff").strip()
        if entrada == "":
            return float(sugerido)
        try:
            valor = float(entrada)
            if minimo <= valor <= maximo:
                return valor
            print(f"    [!] Por favor ingresa un valor entre {minimo} y {maximo}.")
        except ValueError:
            print("    [!] Entrada inválida. Ingresa un número válido.")


def pedir_int(mensaje: str, minimo: int, maximo: int, sugerido: int) -> int:
    """Solicita un número entero validado por consola (Enter usa el valor sugerido)."""
    while True:
        entrada = input(f"  > {mensaje} [{minimo} - {maximo}] (Enter = {sugerido}): ").lstrip("\ufeff").strip()
        if entrada == "":
            return int(sugerido)
        try:
            valor = int(entrada)
            if minimo <= valor <= maximo:
                return valor
            print(f"    [!] Por favor ingresa un entero entre {minimo} y {maximo}.")
        except ValueError:
            print("    [!] Entrada inválida. Ingresa un número entero.")


def pedir_opcion(mensaje: str, opciones: Dict[str, str], sugerido: str) -> str:
    """Solicita elegir una opción de un menú numerado."""
    while True:
        entrada = input(f"  > {mensaje} (Enter = {sugerido}): ").lstrip("\ufeff").strip()
        if entrada == "":
            return opciones[sugerido]
        if entrada in opciones:
            return opciones[entrada]
        print(f"    [!] Opción no válida. Elige una de: {', '.join(opciones.keys())}")


def imprimir_catalogo_consola(lista_productos: List[Dict[str, Any]]) -> None:
    """Imprime en consola la Lista 1: Catálogo de 20 productos del mercado local."""
    print(f"\n{LINEA_DOBLE}")
    print(" LISTA 1 DEL MODELO: CATÁLOGO DE 20 PRODUCTOS (PRECIOS MERCADO LOCAL MXN)")
    print(LINEA_DOBLE)
    print(
        f"{'ID':>2} | {'Producto':<25} | {'Porción':<22} | {'Precio':>9} | "
        f"{'Kcal':>5} | {'Prot(g)':>7} | {'Carb(g)':>7} | {'Grasa(g)':>8}"
    )
    print(LINEA_SIMPLE)
    for p in lista_productos:
        print(
            f"{p['id']:>2} | {p['nombre']:<25} | {p['porcion_desc']:<22} | "
            f"${p['precio_mxn']:>7.2f} | {p['kcal']:>5} | {p['proteinas_g']:>7.1f} | "
            f"{p['carbohidratos_g']:>7.1f} | {p['grasas_g']:>8.1f}"
        )
    print(LINEA_SIMPLE)


def capturar_parametros_manuales() -> tuple[ObjetivosNutricionales, ParametrosAGS, PesosPenalizacion]:
    """
    Solicita manualmente al usuario los objetivos nutricionales y los parámetros
    del Algoritmo Genético Simple, dejando fijos automáticamente los pesos de
    penalización y los criterios de parada.
    """
    print(f"\n{LINEA_DOBLE}")
    print(" CONFIGURACIÓN MANUAL DE OBJETIVOS NUTRICIONALES Y PARÁMETROS DEL AGS")
    print(LINEA_DOBLE)

    print("\n[A] Ingresa tus Objetivos Nutricionales Diarios:")
    kcal = pedir_float("Calorías diarias objetivo (kcal)", 800.0, 5000.0, 2000.0)
    prot = pedir_float("Proteínas objetivo (g)", 30.0, 350.0, 150.0)
    carb = pedir_float("Carbohidratos objetivo (g)", 30.0, 600.0, 200.0)
    grasa = pedir_float("Grasas objetivo (g)", 20.0, 200.0, 65.0)

    print("\n[B] Ingresa manualmente los Parámetros del Algoritmo Genético Simple (AGS):")
    tam_pob = pedir_int("Tamaño de la población (λ)", 4, 300, 80)
    if tam_pob % 2 != 0:
        tam_pob += 1
        print(f"    [*] Se ajustó la población a número par (λ = {tam_pob}) para formar {tam_pob // 2} parejas.")

    num_gen = pedir_int("Número máximo de generaciones (T)", 10, 1000, 200)

    print("\n  Métodos de Selección disponibles:")
    print("    1) Ruleta sesgada (Proporcional a la Aptitud)")
    print("    2) Torneo")
    print("    3) Ranking (Proporcional al Rango)")
    metodo_sel = pedir_opcion(
        "Elige el método de selección [1/2/3]",
        {"1": "ruleta", "2": "torneo", "3": "ranking"},
        "1",
    )

    print("\n  Operadores de Cruce (Crossover) disponibles:")
    print("    1) Un punto (Corte fijo a la mitad - PDF Generalidades)")
    print("    2) Un punto (Corte aleatorio - PDF Algoritmos Genéticos)")
    print("    3) Dos puntos")
    print("    4) Uniforme (Máscara aleatoria)")
    tipo_cruce = pedir_opcion(
        "Elige el operador de cruce [1/2/3/4]",
        {
            "1": "un_punto_mitad",
            "2": "un_punto_aleatorio",
            "3": "dos_puntos",
            "4": "uniforme",
        },
        "1",
    )

    prob_cruce = pedir_float("Probabilidad de cruce (pc, de 0.10 a 1.00)", 0.10, 1.00, 0.85)
    tasa_mut_pct = pedir_float("Tasa de mutación en porcentaje (pm %, ej. 5 para 5%)", 0.5, 30.0, 5.0)
    max_porciones = pedir_int("Máximo de porciones permitidas por producto", 2, 10, 6)

    # Pesos de penalización y criterios de parada FIJOS AUTOMÁTICAMENTE (no manuales)
    pesos_fijos = PesosPenalizacion(
        peso_kcal=2.5,
        peso_proteinas=15.0,
        peso_carbohidratos=10.0,
        peso_grasas=12.0,
    )

    parametros = ParametrosAGS(
        tamano_poblacion=tam_pob,
        num_generaciones=num_gen,
        prob_cruce=prob_cruce,
        tasa_mutacion=tasa_mut_pct / 100.0,
        max_porciones_por_producto=max_porciones,
        metodo_seleccion=metodo_sel,
        tipo_cruce=tipo_cruce,
        elitismo=True,
        detener_por_convergencia=True,
        umbral_convergencia_genes=0.95,
        paciencia_convergencia=100,
        semilla=None,
    )

    print(f"\n{LINEA_SIMPLE}")
    print(" [Parámetros automáticos fijos - Pesos de Penalización y Parada]:")
    print(
        f"  • Pesos de penalización: w_kcal={pesos_fijos.peso_kcal}, "
        f"w_prot={pesos_fijos.peso_proteinas}, w_carb={pesos_fijos.peso_carbohidratos}, "
        f"w_grasa={pesos_fijos.peso_grasas}"
    )
    print(
        f"  • Criterios de parada automáticos: Elitismo=Activo | "
        f"Convergencia De Jong >= {parametros.umbral_convergencia_genes*100:.0f}% | "
        f"Estabilidad = {parametros.paciencia_convergencia} generaciones sin cambio"
    )
    print(LINEA_SIMPLE)

    objetivos = ObjetivosNutricionales(
        kcal=kcal,
        proteinas_g=prot,
        carbohidratos_g=carb,
        grasas_g=grasa,
    )
    return objetivos, parametros, pesos_fijos


def ejecutar_proceso_en_consola(
    lista_productos: List[Dict[str, Any]],
    objetivos: ObjetivosNutricionales,
    parametros: ParametrosAGS,
    pesos: PesosPenalizacion,
) -> None:
    """
    Ejecuta el Algoritmo Genético Simple mostrando detalladamente en consola
    cada una de las 6 fases y el uso de las dos listas.
    """
    t_inicio = time.perf_counter()
    rng = random.Random(parametros.semilla)
    tam_pob = parametros.tamano_poblacion

    # ── FASE 1: INICIALIZACIÓN ───────────────────────────────────────────────
    print(f"\n{LINEA_DOBLE}")
    print(" FASE 1: INICIALIZACIÓN (Generación aleatoria de `poblacion_actual` en t = 0)")
    print(LINEA_DOBLE)
    poblacion_actual = inicializar_poblacion(
        tamano_poblacion=tam_pob,
        lista_productos=lista_productos,
        max_porciones=parametros.max_porciones_por_producto,
        rng=rng,
        objetivo_kcal=objetivos.kcal,
    )
    print(f"  Se generó `poblacion_actual` con λ = {tam_pob} individuos (listas de {len(lista_productos)} genes enteros).")
    muestra_init = min(5, tam_pob)
    for i in range(muestra_init):
        print(f"  Individuo #{i+1:>2} (lista_porciones): {poblacion_actual[i]}")
    if tam_pob > muestra_init:
        print(f"  ... ({tam_pob - muestra_init} individuos adicionales en `poblacion_actual`)")

    # ── FASE 2: EVALUACIÓN DE LA POBLACIÓN INICIAL ───────────────────────────
    print(f"\n{LINEA_DOBLE}")
    print(" FASE 2: EVALUACIÓN EN t = 0 (Cálculo de Costo, Fitness Penalizado y Aptitud - Tabla 6.1)")
    print(LINEA_DOBLE)
    poblacion_evaluada = evaluar_poblacion(
        poblacion=poblacion_actual,
        lista_productos=lista_productos,
        objetivos=objetivos,
        pesos=pesos,
    )

    print(
        f"{'Ind':>3} | {'Costo($)':>9} | {'Kcal':>6} | {'Prot(g)':>7} | "
        f"{'Fitness f(x)':>13} | {'Aptitud g(x)':>12} | {'Prob. p_j':>9} | {'Acum. P_j':>9}"
    )
    print(LINEA_SIMPLE)
    for ind in poblacion_evaluada[: min(10, tam_pob)]:
        print(
            f"{ind.id_individuo:>3} | ${ind.costo_total:>8.2f} | {ind.kcal_totales:>6.0f} | "
            f"{ind.proteinas_totales:>7.1f} | {ind.fitness_penalizado:>13.2f} | "
            f"{ind.aptitud:>12.4f} | {ind.prob_seleccion:>9.4f} | {ind.prob_acumulada:>9.4f}"
        )
    if tam_pob > 10:
        print(f" ... (mostrando primeros 10 de {tam_pob} individuos evaluados)")

    suma_apt = sum(x.aptitud for x in poblacion_evaluada)
    media_apt = suma_apt / tam_pob
    mejor_global = copy.deepcopy(min(poblacion_evaluada, key=lambda x: x.fitness_penalizado))
    print(LINEA_SIMPLE)
    print(
        f" Resumen t=0 -> Suma Aptitud: {suma_apt:.4f} | Aptitud Media: {media_apt:.4f} | "
        f"Mejor Fitness Inicial: {mejor_global.fitness_penalizado:.2f} (Costo: ${mejor_global.costo_total:.2f})"
    )

    # ── CICLO EVOLUTIVO (FASES 3, 4, 5 Y 6) ──────────────────────────────────
    generaciones_sin_mejora = 0
    motivo_parada = f"Se alcanzó el límite máximo de {parametros.num_generaciones} generaciones."
    gen_final = 0
    paso_impresion = max(1, parametros.num_generaciones // 10)

    for gen in range(1, parametros.num_generaciones + 1):
        gen_final = gen
        nueva_poblacion: List[List[int]] = []

        if gen == 1:
            print(f"\n{LINEA_DOBLE}")
            print(" FASES 3, 4 Y 5 EN t = 1: DEMOSTRACIÓN DE SELECCIÓN, CRUCE Y MUTACIÓN (Tabla 6.2)")
            print(LINEA_DOBLE)

        for pareja_idx in range(tam_pob // 2):
            # Fase 3: Selección
            padre_1, u1 = seleccionar_padre(
                poblacion_evaluada, parametros.metodo_seleccion, rng, parametros.tamano_torneo
            )
            padre_2, u2 = seleccionar_padre(
                poblacion_evaluada, parametros.metodo_seleccion, rng, parametros.tamano_torneo
            )

            # Fase 4: Cruce (Crossover)
            hijo_1_cruz, hijo_2_cruz, punto_cruce = cruzar_padres(
                padre_1.cromosoma, padre_2.cromosoma, parametros.prob_cruce, parametros.tipo_cruce, rng
            )

            # Fase 5: Mutación
            hijo_1_mut, muts_1 = mutar_cromosoma(
                hijo_1_cruz, parametros.tasa_mutacion, parametros.max_porciones_por_producto, rng
            )
            hijo_2_mut, muts_2 = mutar_cromosoma(
                hijo_2_cruz, parametros.tasa_mutacion, parametros.max_porciones_por_producto, rng
            )

            nueva_poblacion.append(hijo_1_mut)
            nueva_poblacion.append(hijo_2_mut)

            # Mostrar las primeras 3 parejas de la generación 1 como demostración paso a paso
            if gen == 1 and pareja_idx < 3:
                print(f"  [Pareja #{pareja_idx + 1}]")
                print(f"    • Selección ({parametros.metodo_seleccion}): Padre #{padre_1.id_individuo} (u={u1:.3f}) y Padre #{padre_2.id_individuo} (u={u2:.3f})")
                print(f"      Padre A: {padre_1.cromosoma}")
                print(f"      Padre B: {padre_2.cromosoma}")
                if punto_cruce > 0:
                    print(f"    • Cruce ({parametros.tipo_cruce}, corte en pos={punto_cruce}):")
                else:
                    print("    • Cruce: No se aplicó cruce en esta pareja (se duplicaron los padres).")
                print(f"      Hijo 1 cruzado: {hijo_1_cruz}")
                print(f"      Hijo 2 cruzado: {hijo_2_cruz}")
                print(f"    • Mutación (pm={parametros.tasa_mutacion*100:.1f}%):")
                print(f"      Hijo 1 mutado : {hijo_1_mut}  (genes alterados: {muts_1 if muts_1 else 'ninguno'})")
                print(f"      Hijo 2 mutado : {hijo_2_mut}  (genes alterados: {muts_2 if muts_2 else 'ninguno'})\n")

        # Evaluar la nueva población
        nueva_poblacion_eval = evaluar_poblacion(nueva_poblacion, lista_productos, objetivos, pesos)

        # Fase 6: Reemplazo (`nueva_poblacion` reemplaza a `poblacion_actual`)
        poblacion_evaluada = reemplazar_poblacion(
            poblacion_actual_eval=poblacion_evaluada,
            nueva_poblacion_eval=nueva_poblacion_eval,
            elitismo=parametros.elitismo,
        )
        poblacion_actual = [ind.cromosoma for ind in poblacion_evaluada]

        if gen == 1:
            print(f"{LINEA_DOBLE}")
            print(" FASE 6: REEMPLAZO Y EVOLUCIÓN POR GENERACIONES (`nueva_poblacion` -> `poblacion_actual`)")
            print(LINEA_DOBLE)
            print(
                f"{'Gen':>5} | {'Mejor Costo':>11} | {'Kcal':>6} | {'Prot(g)':>7} | "
                f"{'Fitness f*(x)':>13} | {'Aptitud g*(x)':>13} | {'Convergencia':>12} | {'Evento'}"
            )
            print(LINEA_SIMPLE)

        mejor_gen = min(poblacion_evaluada, key=lambda x: x.fitness_penalizado)
        hubo_mejora = False
        if mejor_gen.fitness_penalizado < mejor_global.fitness_penalizado - 1e-6:
            mejor_global = copy.deepcopy(mejor_gen)
            generaciones_sin_mejora = 0
            hubo_mejora = True
        else:
            generaciones_sin_mejora += 1

        conv_genes = calcular_convergencia_genetica(poblacion_actual)

        # Imprimir en consola cuando hay nuevo óptimo, en intervalos regulares o en la primera generación
        if gen == 1 or hubo_mejora or (gen % paso_impresion == 0) or gen == parametros.num_generaciones:
            etiqueta = "★ Nuevo óptimo" if hubo_mejora else "Reemplazo gen."
            print(
                f"{gen:>5} | ${mejor_global.costo_total:>10.2f} | {mejor_global.kcal_totales:>6.0f} | "
                f"{mejor_global.proteinas_totales:>7.1f} | {mejor_global.fitness_penalizado:>13.2f} | "
                f"{mejor_global.aptitud:>13.4f} | {conv_genes*100:>10.1f} % | {etiqueta}"
            )

        # Criterio de parada automático
        if parametros.detener_por_convergencia:
            if conv_genes >= parametros.umbral_convergencia_genes:
                motivo_parada = (
                    f"Convergencia genética alcanzada ({conv_genes*100:.1f}% >= "
                    f"{parametros.umbral_convergencia_genes*100:.0f}%, Criterio de De Jong) en la generación {gen}."
                )
                break
            if generaciones_sin_mejora >= parametros.paciencia_convergencia and gen >= 30:
                motivo_parada = (
                    f"Convergencia por estabilidad ({generaciones_sin_mejora} generaciones consecutivas "
                    f"sin cambio en el óptimo) en la generación {gen}."
                )
                break

    tiempo_total = time.perf_counter() - t_inicio

    # ── RESULTADO FINAL EN CONSOLA ───────────────────────────────────────────
    print(LINEA_SIMPLE)
    print(f" [Parada del Algoritmo]: {motivo_parada}")
    print(f" [Tiempo de cómputo]   : {tiempo_total*1000:.2f} ms ({gen_final} generaciones ejecutadas)")

    print(f"\n{LINEA_DOBLE}")
    print(" SOLUCIÓN ÓPTIMA ENCONTRADA (EMPAREJAMIENTO DE `lista_productos` Y `lista_porciones`)")
    print(LINEA_DOBLE)
    print(f" Cromosoma ganador (Lista 2 - 20 enteros): {mejor_global.cromosoma}\n")

    print(
        f"{'ID':>2} | {'Ingrediente Seleccionado':<25} | {'Porciones':>9} | {'Medida Unitaria':<22} | "
        f"{'Subtotal':>10} | {'Kcal':>6} | {'Prot':>6} | {'Carb':>6} | {'Grasa':>6}"
    )
    print(LINEA_SIMPLE)

    for prod, porc in zip(lista_productos, mejor_global.cromosoma):
        if porc > 0:
            sub = porc * float(prod["precio_mxn"])
            kcal_p = porc * float(prod["kcal"])
            prot_p = porc * float(prod["proteinas_g"])
            carb_p = porc * float(prod["carbohidratos_g"])
            gras_p = porc * float(prod["grasas_g"])
            print(
                f"{prod['id']:>2} | {prod['nombre']:<25} | {porc:>9} | {prod['porcion_desc']:<22} | "
                f"${sub:>9.2f} | {kcal_p:>6.0f} | {prot_p:>6.1f} | {carb_p:>6.1f} | {gras_p:>6.1f}"
            )

    print(LINEA_SIMPLE)
    print("\n RESUMEN NUTRICIONAL Y ECONÓMICO FINAL:")
    print(f"  • COSTO TOTAL MÍNIMO DIARIO : ${mejor_global.costo_total:.2f} MXN")
    print(
        f"  • Calorías                  : {mejor_global.kcal_totales:.0f} kcal "
        f"(Objetivo: {objetivos.kcal:.0f} kcal | Desviación: {mejor_global.desviacion_kcal:.0f} kcal)"
    )
    print(
        f"  • Proteínas                 : {mejor_global.proteinas_totales:.1f} g "
        f"(Objetivo: {objetivos.proteinas_g:.1f} g | Desviación: {mejor_global.desviacion_proteinas:.1f} g)"
    )
    print(
        f"  • Carbohidratos             : {mejor_global.carbohidratos_totales:.1f} g "
        f"(Objetivo: {objetivos.carbohidratos_g:.1f} g | Desviación: {mejor_global.desviacion_carbohidratos:.1f} g)"
    )
    print(
        f"  • Grasas                    : {mejor_global.grasas_totales:.1f} g "
        f"(Objetivo: {objetivos.grasas_g:.1f} g | Desviación: {mejor_global.desviacion_grasas:.1f} g)"
    )
    print(f"  • Fitness Penalizado f(x)   : {mejor_global.fitness_penalizado:.2f} (Costo + Penalizaciones)")
    print(f"  • Función de Aptitud g(x)   : {mejor_global.aptitud:.4f}")
    print(LINEA_DOBLE)


def main() -> None:
    lista_productos = obtener_catalogo_base()
    imprimir_catalogo_consola(lista_productos)
    objetivos, parametros, pesos = capturar_parametros_manuales()
    ejecutar_proceso_en_consola(lista_productos, objetivos, parametros, pesos)


if __name__ == "__main__":
    main()

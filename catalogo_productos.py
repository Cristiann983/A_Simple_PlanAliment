"""
catalogo_productos.py
─────────────────────────────────────────────────────────────────────────────
Catálogo estático de 20 productos alimenticios del mercado local mexicano
(referencia Profeco / Quién es Quién en los Precios y tablas nutricionales SMAE).

Incluye los 3 ingredientes base citados en la sección 2.7.2 del documento
Generalidades del Proyecto (Pechuga de pollo, Arroz blanco, Huevo) más 17
alimentos esenciales de la canasta básica y dieta deportiva en México.
"""

from __future__ import annotations
from typing import List, Dict, Any
import copy


# Lista 1 del modelo: Catálogo de 20 productos con precios en MXN y macronutrientes por porción
CATALOGO_20_PRODUCTOS: List[Dict[str, Any]] = [
    {
        "id": 0,
        "nombre": "Pechuga de pollo",
        "porcion_desc": "100 g (cocida)",
        "precio_mxn": 15.00,
        "kcal": 165,
        "proteinas_g": 31.0,
        "carbohidratos_g": 0.0,
        "grasas_g": 3.0,
        "categoria": "Proteína animal",
        "referencia_mercado": "$130–$150 MXN/kg (Profeco / Mercado local)",
    },
    {
        "id": 1,
        "nombre": "Arroz blanco",
        "porcion_desc": "100 g (cocido)",
        "precio_mxn": 4.00,
        "kcal": 130,
        "proteinas_g": 2.0,
        "carbohidratos_g": 28.0,
        "grasas_g": 0.0,
        "categoria": "Cereales y tubérculos",
        "referencia_mercado": "$28–$36 MXN/kg en grano seco (PACIC Profeco)",
    },
    {
        "id": 2,
        "nombre": "Huevo de gallina",
        "porcion_desc": "1 pieza (50 g)",
        "precio_mxn": 3.00,
        "kcal": 70,
        "proteinas_g": 6.0,
        "carbohidratos_g": 0.0,
        "grasas_g": 5.0,
        "categoria": "Proteína animal",
        "referencia_mercado": "$48–$54 MXN paquete 18 pzas (PACIC Profeco)",
    },
    {
        "id": 3,
        "nombre": "Frijol negro",
        "porcion_desc": "100 g (cocido)",
        "precio_mxn": 5.00,
        "kcal": 132,
        "proteinas_g": 9.0,
        "carbohidratos_g": 24.0,
        "grasas_g": 0.5,
        "categoria": "Leguminosas",
        "referencia_mercado": "$38–$50 MXN/kg en grano seco (PACIC Profeco)",
    },
    {
        "id": 4,
        "nombre": "Avena en hojuelas",
        "porcion_desc": "50 g (1/2 taza)",
        "precio_mxn": 3.50,
        "kcal": 190,
        "proteinas_g": 6.5,
        "carbohidratos_g": 33.5,
        "grasas_g": 3.5,
        "categoria": "Cereales y tubérculos",
        "referencia_mercado": "$25–$32 MXN bolsa 400 g (QQP Profeco)",
    },
    {
        "id": 5,
        "nombre": "Atún en agua",
        "porcion_desc": "1 lata drenada (100 g)",
        "precio_mxn": 18.00,
        "kcal": 116,
        "proteinas_g": 25.5,
        "carbohidratos_g": 0.0,
        "grasas_g": 1.0,
        "categoria": "Proteína animal",
        "referencia_mercado": "$17–$22 MXN lata 140 g (PACIC Profeco)",
    },
    {
        "id": 6,
        "nombre": "Leche entera de vaca",
        "porcion_desc": "1 vaso (250 ml)",
        "precio_mxn": 7.00,
        "kcal": 150,
        "proteinas_g": 8.0,
        "carbohidratos_g": 12.0,
        "grasas_g": 8.0,
        "categoria": "Lácteos",
        "referencia_mercado": "$26–$30 MXN/litro (PACIC Profeco)",
    },
    {
        "id": 7,
        "nombre": "Tortilla de maíz",
        "porcion_desc": "2 piezas (60 g)",
        "precio_mxn": 1.50,
        "kcal": 128,
        "proteinas_g": 3.4,
        "carbohidratos_g": 27.0,
        "grasas_g": 1.5,
        "categoria": "Cereales y tubérculos",
        "referencia_mercado": "$22–$26 MXN/kg en tortillería (QQP Profeco)",
    },
    {
        "id": 8,
        "nombre": "Plátano tabasco",
        "porcion_desc": "1 pieza mediana (120 g)",
        "precio_mxn": 3.50,
        "kcal": 105,
        "proteinas_g": 1.3,
        "carbohidratos_g": 27.0,
        "grasas_g": 0.4,
        "categoria": "Frutas y verduras",
        "referencia_mercado": "$24–$30 MXN/kg (PACIC Profeco)",
    },
    {
        "id": 9,
        "nombre": "Aguacate Hass",
        "porcion_desc": "50 g (1/3 pieza)",
        "precio_mxn": 7.50,
        "kcal": 80,
        "proteinas_g": 1.0,
        "carbohidratos_g": 4.0,
        "grasas_g": 7.5,
        "categoria": "Grasas saludables",
        "referencia_mercado": "$65–$85 MXN/kg (QQP Profeco)",
    },
    {
        "id": 10,
        "nombre": "Bistec de res magro",
        "porcion_desc": "100 g (cocido)",
        "precio_mxn": 22.00,
        "kcal": 185,
        "proteinas_g": 26.0,
        "carbohidratos_g": 0.0,
        "grasas_g": 8.5,
        "categoria": "Proteína animal",
        "referencia_mercado": "$190–$225 MXN/kg (PACIC Profeco)",
    },
    {
        "id": 11,
        "nombre": "Papa cocida",
        "porcion_desc": "100 g",
        "precio_mxn": 4.00,
        "kcal": 87,
        "proteinas_g": 2.0,
        "carbohidratos_g": 20.0,
        "grasas_g": 0.1,
        "categoria": "Cereales y tubérculos",
        "referencia_mercado": "$35–$45 MXN/kg (PACIC Profeco)",
    },
    {
        "id": 12,
        "nombre": "Lentejas cocidas",
        "porcion_desc": "100 g (cocidas)",
        "precio_mxn": 4.50,
        "kcal": 116,
        "proteinas_g": 9.0,
        "carbohidratos_g": 20.0,
        "grasas_g": 0.4,
        "categoria": "Leguminosas",
        "referencia_mercado": "$42–$52 MXN/kg en bolsa (QQP Profeco)",
    },
    {
        "id": 13,
        "nombre": "Queso panela",
        "porcion_desc": "50 g",
        "precio_mxn": 9.00,
        "kcal": 115,
        "proteinas_g": 9.5,
        "carbohidratos_g": 1.5,
        "grasas_g": 8.0,
        "categoria": "Lácteos",
        "referencia_mercado": "$70–$85 MXN paquete 400 g (QQP Profeco)",
    },
    {
        "id": 14,
        "nombre": "Cacahuate natural",
        "porcion_desc": "30 g (1 puñado)",
        "precio_mxn": 4.00,
        "kcal": 170,
        "proteinas_g": 7.5,
        "carbohidratos_g": 5.0,
        "grasas_g": 15.0,
        "categoria": "Grasas saludables",
        "referencia_mercado": "$110–$135 MXN/kg a granel (Mercado local)",
    },
    {
        "id": 15,
        "nombre": "Pan integral de caja",
        "porcion_desc": "2 rebanadas (60 g)",
        "precio_mxn": 5.00,
        "kcal": 150,
        "proteinas_g": 6.0,
        "carbohidratos_g": 26.0,
        "grasas_g": 2.0,
        "categoria": "Cereales y tubérculos",
        "referencia_mercado": "$45–$55 MXN paquete 680 g (PACIC Profeco)",
    },
    {
        "id": 16,
        "nombre": "Manzana",
        "porcion_desc": "1 pieza mediana (150 g)",
        "precio_mxn": 6.50,
        "kcal": 78,
        "proteinas_g": 0.4,
        "carbohidratos_g": 21.0,
        "grasas_g": 0.3,
        "categoria": "Frutas y verduras",
        "referencia_mercado": "$40–$52 MXN/kg (PACIC Profeco)",
    },
    {
        "id": 17,
        "nombre": "Brócoli al vapor",
        "porcion_desc": "100 g",
        "precio_mxn": 4.50,
        "kcal": 35,
        "proteinas_g": 2.5,
        "carbohidratos_g": 7.0,
        "grasas_g": 0.4,
        "categoria": "Frutas y verduras",
        "referencia_mercado": "$38–$48 MXN/kg (Mercado local)",
    },
    {
        "id": 18,
        "nombre": "Aceite vegetal / oliva",
        "porcion_desc": "1 cucharada (10 ml)",
        "precio_mxn": 2.50,
        "kcal": 88,
        "proteinas_g": 0.0,
        "carbohidratos_g": 0.0,
        "grasas_g": 10.0,
        "categoria": "Grasas saludables",
        "referencia_mercado": "$42–$55 MXN botella 900 ml (PACIC Profeco)",
    },
    {
        "id": 19,
        "nombre": "Yogur natural sin azúcar",
        "porcion_desc": "1 taza (150 g)",
        "precio_mxn": 8.50,
        "kcal": 95,
        "proteinas_g": 8.5,
        "carbohidratos_g": 7.0,
        "grasas_g": 3.8,
        "categoria": "Lácteos",
        "referencia_mercado": "$52–$62 MXN bote 900 g (QQP Profeco)",
    },
]


def obtener_catalogo_base() -> List[Dict[str, Any]]:
    """Devuelve una copia limpia de la lista de 20 productos del catálogo base."""
    return copy.deepcopy(CATALOGO_20_PRODUCTOS)


def generar_catalogo_escalado(num_ingredientes: int = 100) -> List[Dict[str, Any]]:
    """
    Amplía el catálogo hasta `num_ingredientes` (ej. 100 ingredientes, sección 2.7.7)
    sin cambiar la estructura ni la lógica base del Algoritmo Genético Simple.
    Si num_ingredientes <= 20, devuelve los primeros `num_ingredientes` del catálogo base.
    """
    base = obtener_catalogo_base()
    if num_ingredientes <= len(base):
        return base[:num_ingredientes]

    variantes_extra = [
        ("Pierna y muslo de pollo", "100 g (cocido)", 9.50, 184, 24.0, 0.0, 9.0, "Proteína animal"),
        ("Pasta / Espagueti cocido", "100 g", 3.50, 158, 5.8, 31.0, 0.9, "Cereales y tubérculos"),
        ("Sardina en tomate", "100 g", 11.00, 185, 21.0, 1.5, 10.5, "Proteína animal"),
        ("Garbanzo cocido", "100 g", 5.50, 164, 8.9, 27.4, 2.6, "Leguminosas"),
        ("Camote cocido", "100 g", 5.00, 86, 1.6, 20.1, 0.1, "Cereales y tubérculos"),
        ("Lomo de cerdo magro", "100 g", 16.00, 143, 26.0, 0.0, 3.5, "Proteína animal"),
        ("Queso Oaxaca", "50 g", 10.50, 145, 11.0, 1.0, 10.5, "Lácteos"),
        ("Requesón fresco", "100 g", 11.00, 136, 11.5, 3.0, 8.5, "Lácteos"),
        ("Claras de huevo", "100 ml (3 claras)", 6.50, 52, 11.0, 0.7, 0.2, "Proteína animal"),
        ("Zanahoria cruda", "100 g", 2.80, 41, 0.9, 9.6, 0.2, "Frutas y verduras"),
        ("Espinaca cocida", "100 g", 4.00, 23, 2.9, 3.6, 0.4, "Frutas y verduras"),
        ("Naranja Valencia", "1 pieza (150 g)", 3.80, 70, 1.4, 17.5, 0.2, "Frutas y verduras"),
        ("Semilla de girasol", "30 g", 4.50, 175, 6.2, 6.0, 15.4, "Grasas saludables"),
        ("Almendras fileteadas", "30 g", 11.00, 173, 6.3, 6.5, 14.8, "Grasas saludables"),
        ("Leche descremada", "1 vaso (250 ml)", 7.20, 86, 8.3, 12.2, 0.4, "Lácteos"),
        ("Frijol pinto cocido", "100 g", 5.20, 143, 9.0, 26.2, 0.7, "Leguminosas"),
        ("Filete de tilapia", "100 g", 17.00, 128, 26.0, 0.0, 2.7, "Proteína animal"),
        ("Jitomate saladet", "100 g", 3.20, 18, 0.9, 3.9, 0.2, "Frutas y verduras"),
        ("Calabacita italiana", "100 g", 3.50, 17, 1.2, 3.1, 0.3, "Frutas y verduras"),
        ("Amaranto natural", "30 g", 3.80, 112, 4.1, 19.6, 2.1, "Cereales y tubérculos"),
    ]

    catalogo = list(base)
    idx = len(catalogo)
    ciclo = 0
    while len(catalogo) < num_ingredientes:
        v = variantes_extra[ciclo % len(variantes_extra)]
        sufijo = "" if ciclo < len(variantes_extra) else f" (Var. {ciclo // len(variantes_extra) + 1})"
        factor = 1.0 + ((ciclo // len(variantes_extra)) * 0.03)
        catalogo.append(
            {
                "id": idx,
                "nombre": f"{v[0]}{sufijo}",
                "porcion_desc": v[1],
                "precio_mxn": round(v[2] * factor, 2),
                "kcal": int(round(v[3])),
                "proteinas_g": round(v[4], 1),
                "carbohidratos_g": round(v[5], 1),
                "grasas_g": round(v[6], 1),
                "categoria": v[7],
                "referencia_mercado": "Catálogo ampliado de escalabilidad (Profeco / SMAE)",
            }
        )
        idx += 1
        ciclo += 1

    return catalogo

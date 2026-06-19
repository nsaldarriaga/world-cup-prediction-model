# World Cup Match Prediction Model

Proyecto de ciencia de datos que predice resultados de los partidos del Mundial de Fútbol 2026 usando Python, modelos de Poisson, simulación Monte Carlo, Elo rating y calibración probabilística.

El objetivo principal es construir un modelo capaz de estimar:

* goles esperados del equipo local;
* goles esperados del equipo visitante;
* probabilidad de victoria local;
* probabilidad de empate;
* probabilidad de victoria visitante;
* predicción probabilística jornada por jornada durante el Mundial.

---

## 1. Objetivo del proyecto

El objetivo del proyecto es desarrollar un pipeline de predicción para partidos internacionales de fútbol, con foco en la Copa del Mundo.

A diferencia de un modelo que solo predice una clase final, este proyecto busca generar probabilidades interpretables:

```text
Home win: 52%
Draw:     25%
Away win: 23%
```

Esto permite evaluar no solo si el modelo acierta o falla, sino también si asigna una probabilidad razonable al resultado que finalmente ocurre.

Por ese motivo, las métricas principales del proyecto son:

* accuracy;
* log_loss;
* tasa de victorias local;
* tasa de victorias visitante;
* tasa de empates predichos;
* MAE de goles esperados.

---

## 2. Dataset

El proyecto utiliza un dataset histórico público de resultados internacionales de selecciones nacionales.

El dataset contiene partidos históricos con columnas como:

* fecha;
* equipo local;
* equipo visitante;
* goles local;
* goles visitante;
* torneo;
* ciudad;
* país;
* indicador de cancha neutral.

También contiene partidos futuros del calendario, que permiten generar predicciones para jornadas pendientes.

Para el entrenamiento principal se utilizaron partidos desde 2010 en adelante, con el objetivo de trabajar con información más representativa del fútbol internacional moderno.

---

## 3. Pipeline general

El pipeline del proyecto sigue esta estructura:

```text
Carga de datos
    ↓
Limpieza y separación de partidos jugados / futuros
    ↓
Construcción de features históricas
    ↓
Cálculo de features de forma reciente
    ↓
Cálculo de Elo rating
    ↓
Modelo Poisson para goles esperados
    ↓
Conversión de goles esperados a probabilidades H/D/A
    ↓
Calibración multinomial de probabilidades
    ↓
Evaluación temporal
    ↓
Predicción de partidos futuros
```

---

## 4. Features utilizadas

El modelo utiliza features construidas a partir del rendimiento histórico de cada selección.

### Forma reciente

Se calcularon medias móviles de corto y mediano plazo:

* goles a favor últimos 5 partidos;
* goles a favor últimos 10 partidos;
* goles en contra últimos 5 partidos;
* goles en contra últimos 10 partidos;
* puntos promedio últimos 5 partidos;
* puntos promedio últimos 10 partidos.

### Diferencias entre equipos

Para representar la comparación directa entre selecciones, se crearon features de diferencia:

* diferencia de ataque;
* diferencia defensiva;
* diferencia de puntos recientes;
* diferencia de Elo.

### Torneo y contexto

También se agregaron variables relacionadas al tipo de partido:

* amistoso;
* Copa del Mundo;
* clasificatorio mundialista;
* copa continental;
* Nations League;
* partido competitivo;
* cancha neutral.

---

## 5. Modelo base: Poisson

El núcleo del proyecto es un modelo Poisson para estimar goles esperados.

Se entrenan dos modelos:

```text
Modelo 1: predice lambda_home
Modelo 2: predice lambda_away
```

Donde:

```text
lambda_home = goles esperados del equipo local
lambda_away = goles esperados del equipo visitante
```

Luego, con esos valores, se construye una matriz de marcadores posibles usando la distribución de Poisson.

A partir de esa matriz se calculan:

```text
P(Home win)
P(Draw)
P(Away win)
```

Este enfoque es interpretable porque permite explicar la predicción desde los goles esperados.

---

## 6. Simulación Monte Carlo

Además del cálculo analítico de probabilidades, el proyecto incluye simulación Monte Carlo.

La simulación permite generar miles de partidos posibles a partir de los goles esperados de cada equipo.

Ejemplo conceptual:

```text
Argentina vs France

Simulación 1: 2-1
Simulación 2: 1-1
Simulación 3: 0-1
...
```

Luego se agregan los resultados simulados para estimar:

* probabilidad de victoria local;
* probabilidad de empate;
* probabilidad de victoria visitante;
* promedio de goles simulados.

---

## 7. Elo rating

El proyecto incorpora Elo rating como medida de fuerza relativa entre selecciones.

El Elo permite capturar información histórica acumulada que no siempre aparece en las medias móviles recientes.

Se construyeron features como:

* Elo del equipo local;
* Elo del equipo visitante;
* diferencia de Elo;
* transformaciones suavizadas de la diferencia de Elo.

Estas variables ayudan al modelo a diferenciar equipos fuertes, equipos débiles y partidos potencialmente equilibrados.

---

## 8. Calibración probabilística

Uno de los principales hallazgos del proyecto fue que el modelo Poisson generaba probabilidades razonables, pero no perfectamente calibradas.

El modelo podía mantener una accuracy cercana al 60%, pero el `log_loss` indicaba que las probabilidades podían mejorar.

Para resolver esto se agregó una capa de calibración:

```text
Poisson probabilities
    ↓
Logistic Regression calibrator
    ↓
Calibrated probabilities
```

El calibrador toma como input las probabilidades generadas por el Poisson y aprende a reajustarlas para que estén mejor alineadas con la frecuencia real de los resultados.

---

## 9. OOF Multinomial Calibrator

El calibrador final se entrena usando una estrategia OOF, es decir, out-of-fold.

La idea es evitar que el calibrador aprenda sobre probabilidades generadas por un modelo entrenado sobre los mismos partidos.

El flujo conceptual es:

```text
Fold 1:
Poisson entrena en una parte del histórico
Predice probabilidades sobre datos no vistos

Fold 2:
Poisson entrena en otra parte del histórico
Predice probabilidades sobre datos no vistos

Luego:
El calibrador aprende usando esas probabilidades fuera de muestra.
```

Esto reduce el riesgo de overfitting y hace que la calibración sea más realista para partidos futuros.

---

## 10. Experimentos realizados

Se compararon diferentes estrategias para mejorar el modelo probabilístico.

### Experimento 1: Poisson + calibrador multinomial OOF

Este fue el baseline final del proyecto.

Resultados principales:

```text
Accuracy:        0.604799
Log loss:        0.863299
Draw real rate:  0.230833
Draw pred rate:  0.022614
```

Este modelo logró mantener una accuracy competitiva y mejorar el log loss frente a versiones anteriores.

### Experimento 2: Calibrador con class_weight="balanced"

Se probó ajustar los pesos de clase para darle más importancia al empate.

Resultados:

```text
Accuracy:        0.564258
Log loss:        0.899676
Draw real rate:  0.230833
Draw pred rate:  0.264755
```

Conclusión:

El modelo empezó a predecir muchos más empates, pero empeoró accuracy y log loss. Esto indica que el problema de los empates no se resuelve simplemente balanceando clases.

### Experimento 3: Calibrador con features contextuales

Se agregaron features adicionales al calibrador, como diferencias de lambdas, Elo, tipo de torneo y flags de contexto.

Resultados:

```text
Accuracy:        0.603696
Log loss:        0.860925
Draw real rate:  0.230833
Draw pred rate:  0.001655
```

Conclusión:

Este experimento obtuvo el menor log loss, pero la mejora fue marginal y redujo casi por completo la predicción de empates.

### Experimento 3B: Calibrador enfocado en empates

Se probó una variante con features enfocadas en partidos cerrados.

Resultados:

```text
Accuracy:        0.603696
Log loss:        0.867355
Draw real rate:  0.230833
Draw pred rate:  0.000276
```

Conclusión:

No mejoró el log loss ni resolvió el problema de empates.

### Experimento 4: Poisson bivariado

Se probó una extensión bivariada del Poisson para capturar dependencia entre los goles de ambos equipos mediante un componente común.

El tuning mostró que el mejor valor era:

```text
common_factor = 0.00
```

A medida que aumentaba el componente común, el log loss empeoraba.

Conclusión:

El Poisson bivariado fue descartado porque no mejoró la calidad probabilística del modelo.

---

## 11. Métricas finales

Resultados del modelo final:

```text 
Accuracy:        0.604799
Log loss:        0.863299
Draw real rate:  0.230833
Draw pred rate:  0.022614
```

La accuracy muestra que el modelo acierta aproximadamente el 60% de los resultados.

El log loss muestra que las probabilidades están razonablemente calibradas para un problema de alta incertidumbre como el fútbol.

---

## 12. Problema de los empates

Uno de los principales desafíos detectados fue la predicción de empates.

Aunque los empates representan aproximadamente el 23% de los partidos, el modelo tiende a asignarles probabilidad sin elegirlos frecuentemente como clase final.

Esto ocurre porque el empate suele quedar como segunda clase más probable, incluso en partidos equilibrados.

Ejemplo:

```text
Home: 39%
Draw: 31%
Away: 30%
```

En este caso, el modelo predice Home, aunque el empate tenga una probabilidad relevante.

Este comportamiento muestra que el modelo puede ser útil probabilísticamente incluso cuando no predice muchos empates como clase final.

---

## 13. Predicción dinámica del torneo

El proyecto está diseñado para operar de forma dinámica durante el torneo.

El flujo esperado es:

```text
1. Se juegan nuevos partidos.
2. Se actualiza el dataset con los resultados reales.
3. Se recalculan features recientes y Elo.
4. Se vuelve a entrenar o actualizar el pipeline.
5. Se predice la siguiente jornada.
```

Esto permite adaptar el modelo a cambios recientes de forma, lesiones indirectamente reflejadas en resultados, rendimiento reciente y evolución del torneo.

---

## 14. Estructura del proyecto

Estructura general esperada:

```text
project/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   └── experiments.ipynb
│
├── outputs/
│   ├── final_experiment_summary.csv
│   ├──next_round_predictions.csv
│   ├──calendar_predictions.csv    
│
├── src/
│   ├── data_loader.py
│   ├── cleaning.py
│   ├── features.py
│   ├── elo.py
│   ├── model.py
│   ├── probabilities.py
│   ├── evaluation.py
│   ├── dynamic.py
│   └── config.py
│
├── main_train.py
├── main_predict_calendar.py
├── main_predict_next_round.py
├── requirements.txt
└── README.md
```

---

## 15. Cómo ejecutar el proyecto

### Entrenar y evaluar el modelo

```bash
python main_train.py
```

Este script:

* carga los datos;
* limpia el dataset;
* separa partidos jugados y futuros;
* construye el dataframe de modelado;
* entrena el modelo Poisson;
* entrena el calibrador;
* evalúa métricas de train y test;
* muestra predicciones de ejemplo.

### Predecir calendario futuro

```bash
python main_predict_calendar.py
```

### Predecir próxima jornada

```bash
python main_predict_next_round.py
```

---

## 16. Experimentos en notebook

Los experimentos de calibración y comparación de modelos se encuentran en la carpeta `notebooks`.

La notebook parte desde el dataframe final del pipeline:

```python
model_df = build_model_dataset(played_matches)
df_exp = model_df[model_df["date"] >= "2010-01-01"].copy()
```

Esto asegura que todos los experimentos usen:

* el mismo dataset;
* el mismo split temporal;
* las mismas features base;
* las mismas métricas.

---

## 17. Métricas utilizadas

### Accuracy

Mide la proporción de partidos donde la clase predicha coincide con el resultado real.

```text
Predicción correcta / total de partidos
```

### Log loss

Mide la calidad probabilística del modelo.

Penaliza más cuando el modelo asigna baja probabilidad al resultado que realmente ocurre.

Por ejemplo, si el resultado real es empate:

```text
Modelo A: Draw = 0.30
Modelo B: Draw = 0.05
```

El Modelo B recibe una penalización mayor.

Por eso el log loss es clave para evaluar modelos probabilísticos.

### Draw pred rate

Mide qué proporción de partidos el modelo predice como empate.

Esta métrica fue importante porque el modelo tendía a subpredecir empates como clase final.

---

## 18. Limitaciones

El proyecto tiene algunas limitaciones importantes:

* no utiliza cuotas de casas de apuestas;
* no incluye alineaciones titulares;
* no incluye lesiones;
* no incluye información táctica;
* no incluye datos de jugadores;
* no incluye xG real por partido;
* los empates siguen siendo difíciles de predecir como clase final;
* el modelo depende de la calidad y actualización del dataset histórico.

---


## 19. Conclusión final

El proyecto logró construir un modelo probabilístico interpretable para predicción de partidos internacionales.

El enfoque final combina:

```text
Poisson para goles esperados
+
Elo y features históricas
+
calibración multinomial OOF
```

La principal mejora técnica fue pasar de un modelo que solo generaba probabilidades raw a un pipeline calibrado, capaz de mantener accuracy competitiva y mejorar el log loss.

El modelo final no pretende eliminar la incertidumbre del fútbol, sino representarla mejor mediante probabilidades más realistas.
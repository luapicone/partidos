# partidos

Motor base para analizar partidos y generar predicciones de futbol con salida lista para contenido corto.

## Alcance del MVP

Esta primera version esta enfocada en `selecciones nacionales`, porque usa un dataset historico abierto de partidos internacionales. El flujo es:

1. descargar resultados historicos
2. calcular ratings tipo Elo
3. ponderar partidos oficiales vs amistosos
4. medir forma reciente ajustada por nivel de rival y decaimiento temporal
5. estimar probabilidades de victoria, empate y marcador probable
6. validar el modelo con backtesting
7. generar una salida corta estilo TikTok

## Instalacion

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Descargar datos

```bash
partidos update-data
```

## Ejemplo de prediccion

```bash
partidos predict \
  --team-a "Argentina" \
  --team-b "Uruguay" \
  --date 2026-06-25 \
  --neutral \
  --tiktok-script
```

## Generar grafico de probabilidades

```bash
partidos predict \
  --team-a "Brazil" \
  --team-b "Haiti" \
  --date 2026-06-19 \
  --neutral \
  --chart
```

Eso genera un `SVG` en `charts/` con barras para:
- victoria equipo A
- empate
- victoria equipo B

## Medir si el modelo realmente sirve

```bash
partidos backtest --matches 200
```

## Que devuelve

- probabilidades de `equipo A / empate / equipo B`
- `marcador mas probable`
- ratings Elo
- forma reciente
- forma reciente ajustada por rival
- goles esperados
- guion corto para narracion
- grafico `SVG` para usar en navegador o edicion de video

## Mejoras agregadas sobre el MVP inicial

- `peso de torneo`: amistosos pesan menos que eliminatorias y mundial
- `decaimiento temporal`: los partidos mas recientes influyen mas
- `forma ajustada por rival`: no vale lo mismo sumar contra una seleccion top que contra una debil
- `backtesting`: accuracy, log loss y brier score para medir calidad del modelo
- `desbalance real entre equipos`: el modelo ahora amplifica mas los cruces muy desparejos
- `goles ajustados por rival`: los goles recientes ya no pesan igual contra cualquier oponente

## Estructura

```text
src/partidos/
  cli.py
  config.py
  data.py
  model.py
  output.py
```

## Siguientes extensiones naturales

- soporte para clubes y ligas
- integracion con fixtures futuros
- export a JSON/CSV
- plantillas visuales para reels y TikTok

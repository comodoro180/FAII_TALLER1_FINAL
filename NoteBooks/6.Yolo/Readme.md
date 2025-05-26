# Documentación del Notebook YOLO con Optimización Optuna

## Tabla de Contenidos
1. [Instalación y Configuración](#1-instalación-y-configuración)
2. [Análisis Exploratorio de Datos (EDA)](#2-análisis-exploratorio-de-datos-eda)
3. [Preparación de los Datos](#3-preparación-de-los-datos)
4. [Configuración YOLO](#4-configuración-yolo)
5. [Optimización con Optuna](#5-optimización-con-optuna)
6. [Entrenamiento con Mejores Hiperparámetros](#6-entrenamiento-con-mejores-hiperparámetros)
7. [Generación de Archivo de Submission](#7-generación-de-archivo-de-submission)
8. [Visualización de Imágenes de Test](#8-visualización-de-imágenes-de-test)

---

## 1. Instalación y Configuración

### Propósito
Esta sección configura el entorno de trabajo instalando todas las dependencias necesarias para el entrenamiento de modelos YOLO y la optimización de hiperparámetros.

### Componentes Principales

#### Instalación de Dependencias
```python
# Librerías principales instaladas:
- torch==2.1.0+cu118  # PyTorch con soporte CUDA
- torchvision==0.16.0+cu118
- ultralytics  # Framework YOLO
- scikit-learn  # Para división de datos
- optuna  # Optimización de hiperparámetros
- pandas, matplotlib, seaborn  # Análisis y visualización
- plotly, bokeh  # Visualizaciones interactivas
```

#### Verificación del Hardware
- Comprueba la disponibilidad de GPU
- Muestra información del dispositivo CUDA
- Configura el entorno para usar GPU si está disponible

### Salidas Esperadas
- Confirmación de la versión de PyTorch
- Estado de disponibilidad de CUDA
- Información del dispositivo GPU

---

## 2. Análisis Exploratorio de Datos (EDA)

### Propósito
Analizar y comprender la estructura del dataset, incluyendo la distribución de clases y la visualización de las anotaciones.

### Componentes Principales

#### Carga de Datos
```python
# Archivos necesarios:
- train.csv: Contiene las anotaciones (filename, class, xmin, ymin, xmax, ymax)
- ./images/: Directorio con las imágenes
```

#### Análisis Estadístico
- **Número total de anotaciones**: Cuenta de todas las bounding boxes
- **Imágenes únicas**: Número de imágenes diferentes en el dataset  
- **Distribución de clases**: Frecuencia de cada clase en el dataset
- **Visualización**: Gráfico de barras mostrando el desbalance de clases

#### Visualización de Muestras
- Muestra 16 imágenes aleatorias con sus bounding boxes
- Cada imagen se dibuja con rectángulos verdes y etiquetas de clase
- Layout de 2x8 para una visualización organizada

#### Mapeo de Clases
- Crea diccionarios bidireccionales:
  - `class_to_id`: Nombres de clase → IDs numéricos
  - `id_to_class`: IDs numéricos → Nombres de clase

### Salidas Esperadas
- Estadísticas del dataset
- Gráfico de distribución de clases
- Grid de imágenes con anotaciones visualizadas
- Mapeo de clases para el entrenamiento

---

## 3. Preparación de los Datos

### Propósito
Convertir los datos del formato CSV a la estructura de directorios requerida por YOLO y transformar las anotaciones al formato YOLO.

### Estructura de Directorios Creada
```
yolo_dataset/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

### Componentes Principales

#### División Train/Validation
- **80/20 split**: 80% entrenamiento, 20% validación
- **Estratificación**: Mantiene la proporción de clases en ambos conjuntos
- **Reproducibilidad**: `random_state=42` para resultados consistentes

#### Conversión de Formato
- **Anotaciones originales**: `[xmin, ymin, xmax, ymax]` (coordenadas absolutas)
- **Formato YOLO**: `[class_id, center_x, center_y, width, height]` (coordenadas normalizadas)

#### Procesamiento de Archivos
- Copia imágenes a los directorios correspondientes
- Genera archivos `.txt` con anotaciones en formato YOLO
- Manejo de errores para imágenes no encontradas o corrompidas
- Validación de coordenadas normalizadas (0.0 ≤ valor ≤ 1.0)

### Validaciones Incluidas
- Verificación de existencia de imágenes
- Validación de coordenadas de bounding boxes
- Manejo de anotaciones inválidas
- Reporte de imágenes problemáticas

---

## 4. Configuración YOLO

### Propósito
Cargar el modelo YOLO preentrenado y configurar el entorno para el entrenamiento.

### Componentes Principales

#### Modelo Base
- **YOLOv11 Nano** (`yolo11n.pt`): Modelo más ligero para experimentación rápida
- Descarga automática de pesos preentrenados
- Adaptación automática del número de clases durante el entrenamiento

#### Verificación del Sistema
- Confirmación de versiones de PyTorch y TorchVision
- Verificación de disponibilidad y configuración de CUDA
- Preparación del entorno para entrenamiento con GPU

### Alternativas Disponibles
- Modelos más grandes: `yolo11s.pt`, `yolo11m.pt`, `yolo11l.pt`, `yolo11x.pt`
- Entrenamiento desde cero usando archivos YAML de configuración

---

## 5. Optimización con Optuna

### Propósito
Encontrar automáticamente los mejores hiperparámetros para el modelo YOLO usando optimización bayesiana.

### Hiperparámetros Optimizados

#### Parámetros de Entrenamiento
- **Learning Rate**: `1e-5` a `1e-2` (escala logarítmica)
- **Batch Size**: `[16, 32, 64]` (valores categóricos)
- **Optimizer**: `["SGD", "Adam", "AdamW"]` (algoritmos de optimización)
- **Epochs**: `200` a `300` (número de iteraciones)
- **Image Size**: `320` a `736` píxeles (incrementos de 32)

#### Función Objetivo
- **Métrica objetivo**: mAP50-95 (mean Average Precision)
- **Dirección**: Maximización
- **Manejo de errores**: Retorna 0.0 si falla el entrenamiento

### Proceso de Optimización
1. **Generación de hiperparámetros**: Optuna sugiere combinaciones
2. **Entrenamiento**: Cada trial entrena un modelo completo
3. **Evaluación**: Se registra el mAP50-95 del conjunto de validación
4. **Optimización**: Algoritmo bayesiano mejora las sugerencias

### Configuración del Estudio
- **Número de trials**: 300 pruebas
- **Almacenamiento**: Resultados en `optuna_runs/trial_X/`
- **Reproducibilidad**: Cada trial se guarda independientemente

---

## 6. Entrenamiento con Mejores Hiperparámetros

### Propósito
Realizar el entrenamiento final usando los hiperparámetros óptimos encontrados por Optuna.

### Configuración del Entrenamiento

#### Parámetros Utilizados
```python
epochs = study.best_params['epochs']
image_size = study.best_params['image_size'] 
batch_size = study.best_params['batch_size']
optimizer = study.best_params['optimizer']
learning_rate = study.best_params['learning_rate']
```

#### Configuración de Salida
- **Directorio de proyecto**: `custom_yolo_training`
- **Nombre de ejecución**: `run_1`
- **Device**: GPU (device=0) si está disponible
- **Verbose**: Salida detallada habilitada

### Proceso de Entrenamiento
1. **Carga del modelo base**: YOLO preentrenado
2. **Configuración**: Aplicación de hiperparámetros óptimos
3. **Entrenamiento**: Proceso completo con validación automática
4. **Guardado**: Pesos del mejor modelo (`best.pt`) y último (`last.pt`)

### Evaluación Post-Entrenamiento
- **Validación automática**: Evaluación en conjunto de validación
- **Métricas reportadas**: mAP50-95, mAP50, mAP75
- **Visualizaciones**: Curvas de entrenamiento, matriz de confusión
- **Inferencia de prueba**: Ejemplo en imagen específica

---

## 7. Generación de Archivo de Submission

### Propósito
Crear las predicciones finales para el conjunto de test y generar el archivo de submission en el formato requerido.

### Proceso de Predicción

#### Configuración
- **Modelo usado**: Mejores pesos guardados (`best.pt`)
- **Umbral de confianza**: 0.1 (configurable)
- **Estrategia**: Una predicción por imagen (la de mayor confianza)

#### Manejo de Casos Especiales
- **Sin detecciones**: Valores por defecto (`class="none"`, coordenadas `[0,0,1,1]`)
- **Imágenes faltantes**: Fila con valores por defecto
- **Errores de procesamiento**: Logging y continuación con valores por defecto

#### Formato de Salida
```csv
filename,class,xmin,ymin,xmax,ymax
image_001.jpg,person,100,50,200,150
image_002.jpg,none,0,0,1,1
```

### Validaciones de Calidad
- Verificación de una fila por imagen de test
- Validación de formato de columnas
- Comparación con archivo de muestra
- Reporte de estadísticas finales

---

## 8. Visualización de Imágenes de Test

### Propósito
Visualizar las predicciones del modelo en el conjunto de test para evaluación cualitativa.

### Configuración de Visualización

#### Layout
- **Grid**: 6 columnas × 11 filas (66 subplots)
- **Tamaño**: 32×22 pulgadas para resolución adecuada
- **Manejo**: Subplots vacíos se ocultan automáticamente

#### Elementos Visuales
- **Bounding boxes**: Rectángulos verdes
- **Etiquetas**: Clase + confianza
- **Títulos**: Nombre del archivo
- **Indicadores de error**: Texto rojo para problemas

### Información Mostrada
- **Detecciones exitosas**: Boxes con clase y confianza
- **Sin detecciones**: Título en rojo
- **Errores**: Mensaje de error en el subplot
- **Estadísticas**: Logging de casos problemáticos

### Casos Manejados
- Imágenes no encontradas
- Errores de lectura
- Fallos en la inferencia
- Problemas de mapeo de clases

---

## Consideraciones Técnicas

### Requisitos del Sistema
- **GPU**: NVIDIA con soporte CUDA (recomendado)
- **RAM**: Mínimo 8GB, recomendado 16GB+
- **Almacenamiento**: ~10GB para datasets y modelos

### Optimizaciones de Rendimiento
- **Batch processing**: Procesamiento por lotes eficiente
- **Caching**: Almacenamiento de resultados intermedios
- **Paralelización**: Uso de múltiples workers cuando es posible

### Mejores Prácticas
- **Logging detallado**: Para debugging y monitoreo
- **Checkpoints**: Guardado regular de progreso
- **Validación cruzada**: Múltiples métricas de evaluación
- **Reproducibilidad**: Seeds fijos y versionado de código

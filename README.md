# Taller 1 - Fundamentos de Analítica II

## Autores
- **Alexandra López Cuenca**  
- **Carlos Arturo Hoyos Rincón**

## Descripción General

La solución al taller está organizada en una propuesta distribuida en **6 notebooks**, cada uno con un objetivo específico en el flujo de trabajo de procesamiento de imágenes y modelado para tareas de clasificación, regresión y detección. A continuación se presenta el detalle de cada uno:

---

### 1. `Balanceo_img.ipynb` – *Balanceo de Imágenes*

Se generaron imágenes adicionales aplicando técnicas de *augmentation* con el objetivo de **balancear la cantidad de muestras por clase**.  
**Nota importante:** No se aplicaron transformaciones de rotación ni cambio de tamaño.

---

### 2. `Modelo_personalizado.ipynb` – *Modelo Personalizado*

Se diseñó un modelo desde cero usando las imágenes balanceadas del punto anterior, con el **objetivo principal de realizar clasificación**.  
Este modelo será reutilizado posteriormente como **backbone personalizado**.

---

### 3. `Transfer_densenet201.ipynb` – *Transfer Learning con DenseNet201*

Se implementa el modelo **DenseNet201** como **backbone** para realizar tanto la **clasificación como la regresión** sobre el conjunto de datos.

---

### 4. `Transfer_ResNet.ipynb` – *Transfer Learning con ResNet50*

Se emplea la arquitectura **ResNet50** como **backbone** para abordar las tareas de **clasificación y regresión**.

---

### 5. `Transfer_Personalizado.ipynb` – *Transfer con Modelo Personalizado*

En este notebook se reutiliza el **modelo personalizado** previamente entrenado como **backbone** para realizar nuevas tareas, optimizando el desempeño en clasificación y regresión.

---

### 6. `Yolo.ipynb` – *Detección con YOLOv11*

Implementamos el modelo **YOLOv11** para resolver la tarea de **detección de objetos**, con el objetivo de generar **submissions** para la competencia en Kaggle.

---

## Recomendaciones

- Asegúrese de tener las dependencias necesarias instaladas para ejecutar cada notebook (PyTorch, Albumentations, OpenCV, etc.) utilice el archivo requirements.txt.
- El uso de GPU es altamente recomendado para entrenar los modelos en un tiempo razonable.
- Verifique que las rutas de las imágenes y los archivos `.csv` estén correctamente definidos antes de ejecutar cada notebook.

---

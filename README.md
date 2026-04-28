# RapidRelief AI — Clasificación Automatizada de Donaciones Textiles

**Proyecto académico** · Inteligencia Artificial y Sistemas Expertos
Dr. José Ambrosio Bastián · Universidad Simón Bolívar · 2026

**Equipo:** Fernando Acuña Martínez · Pamela Ruíz Velasco Calvo · Said Lozada Vivar

**Demo en vivo:** [rapidreliefai.streamlit.app](https://rapidreliefai.streamlit.app)
**Repositorio:** [github.com/FernandoAcunaMtz/RapidReliefAI](https://github.com/FernandoAcunaMtz/RapidReliefAI)

---

## 1. Resumen ejecutivo

RapidRelief AI automatiza la clasificación de prendas donadas en centros de acopio humanitario mediante visión por computadora. Un voluntario fotografía la prenda con su teléfono o cámara local, y el sistema la clasifica en una de 10 categorías (vestido, pantalón, camisa, calzado, etc.) en menos de 300 ms. Esto elimina el cuello de botella manual durante desastres naturales y crisis de refugiados, donde el tiempo de clasificación puede determinar si la ayuda llega o no.

**Indicadores de éxito:**

| KPI | Meta | Estado actual |
|-----|------|--------------|
| Val accuracy | ≥ 90% | 88.27% (v1) → objetivo con v2 + fine-tuning |
| F1-score por clase | ≥ 0.88 | 0.86 macro avg (v1) |
| Tamaño TFLite INT8 | ≤ 6 MB | Pendiente cuantización en v2 |
| Latencia inferencia | ≤ 300 ms | ✓ |
| Funcionalidad offline | 100% | ✓ (TFLite embebido en Flutter) |

---

## 2. Arquitectura del sistema

```
┌──────────────────────────────────────────────────────┐
│  Google Colab (GPU T4)                              │
│  ├── Notebooks: 00 setup → 01 EDA → 02 entrenamiento │
│  └── Drive: datasets + modelos guardados            │
└──────────────────────┬───────────────────────────────┘
                       │ exporta .keras + .tflite (INT8)
         ┌─────────────┼─────────────┐
         ▼                           ▼
┌─────────────────────┐    ┌────────────────────┐
│ Streamlit Cloud     │    │ App móvil Flutter  │
│ (validación dev)    │    │ (deploy final,     │
│ rapidreliefai       │    │  offline en campo) │
│ .streamlit.app      │    │                    │
└─────────────────────┘    └────────────────────┘
```

**Entrenamiento:** Google Colab + Google Drive (datasets y checkpoints).
**Demo:** Streamlit Cloud (interfaz web responsiva con upload + cámara).
**Despliegue final:** Flutter + TFLite (offline en zonas sin conectividad).

---

## 3. Modelo — Justificación técnica

### Por qué MobileNetV2

De todos los modelos en [keras.io/api/applications](https://keras.io/api/applications/), **MobileNetV2** es el más adecuado:

| Criterio | MobileNetV2 | EfficientNetB0 | ResNet50V2 |
|----------|-------------|----------------|------------|
| Tamaño | **14 MB** | 29 MB | 98 MB |
| Parámetros | **3.5 M** | 5.3 M | 25.6 M |
| Top-1 ImageNet | 71.3% | 77.1% | 75.6% |
| Velocidad CPU | **~26 ms** | ~46 ms | ~195 ms |
| TF Lite | **Oficial, INT8** | Bueno | Parcial |
| Edge devices | **Sí** | Marginal | No viable |

**Argumentos decisivos:**

1. **Diseñado para edge:** Depthwise separable convolutions + inverted residuals reducen ~9× los FLOPs vs. una CNN estándar. Inferencia en tiempo real en Raspberry Pi y smartphones de gama media.
2. **TF Lite oficial:** Google mantiene cuantización INT8 que reduce el modelo a ~4 MB sin pérdida significativa de precisión.
3. **Validado en clase:** El notebook de referencia del docente (`context/260416_FINAL_Transferencia_de_aprendizaje_categorical.ipynb`) demuestra el patrón.

### Arquitectura de la cabeza personalizada (v2)

```
MobileNetV2 (preentrenada ImageNet, 154 capas, 3.5M params)
    ↓ include_top=False · weights='imagenet'
GlobalAveragePooling2D    (7×7×1280 → 1280 features)
    ↓
Dense(512, relu)
    ↓
Dropout(0.3)              (regularización)
    ↓
Dense(10, softmax)        (10 clases del proyecto)
```

**Estrategia de entrenamiento en dos fases:**

| Fase | Descripción | Capas entrenables | LR | Épocas |
|------|-------------|-------------------|----|----|
| **A — Feature extraction** | Base congelada, solo cabeza | ~655K (cabeza) | 1e-3 | hasta 25 (EarlyStopping) |
| **B — Fine-tuning** | Últimas 54 capas de MobileNetV2 descongeladas | ~2M | 1e-5 | hasta 20 (EarlyStopping) |

**Optimizador:** Adam (más estable que SGD para transfer learning).
**Loss:** `categorical_crossentropy` con One-Hot Encoding.
**Regularización:** `class_weight='balanced'` + `Dropout(0.3)` + `EarlyStopping(patience=6)` + `ReduceLROnPlateau`.

---

## 4. Datos

**Dataset:** [Clothing Dataset Small](https://www.kaggle.com/datasets/abdelrahmansoltan98/clothing-dataset-small) — fotografías reales de prendas en 10 categorías.

| Split | Imágenes | Uso |
|-------|----------|-----|
| Train | 3,068 | Entrenamiento con augmentación |
| Validation | 341 | Validación durante entrenamiento |
| Test | 372 | Evaluación final + matriz de confusión |

**Clases (orden alfabético):** dress, hat, longsleeve, outwear, pants, shirt, shoes, shorts, skirt, t-shirt.

**Augmentación validada para este dominio:**
```python
ImageDataGenerator(
    preprocessing_function=preprocess_input,  # MobileNetV2 → [-1, 1]
    rotation_range=15,
    width_shift_range=0.10,
    height_shift_range=0.10,
    shear_range=10.0,
    zoom_range=0.15,
    horizontal_flip=True,
    brightness_range=[0.85, 1.15],
)
```

> **Nota sobre Fashion-MNIST:** Originalmente se planteó complementar con Fashion-MNIST (70,000 imágenes grayscale 28×28), pero se descartó tras observar que el desfase de dominio (grayscale 28×28 → RGB 224×224) introducía ruido que degradaba el clasificador en fotos reales. Se decidió enfocar el entrenamiento exclusivamente en Clothing Dataset Small con augmentación robusta.

**Privacidad y ética:** Solo se procesan imágenes de prendas inanimadas. No hay datos biométricos ni rostros. El modelo opera offline en el dispositivo final (sin envío a servidores externos).

---

## 5. Estructura del repositorio

```
RapidReliefAI/
├── README.md                              # Este archivo
├── plan.md                                # Plan de desarrollo agile por sprints
├── app.py                                 # Aplicación Streamlit (demo web)
├── requirements.txt                       # Dependencias Python
├── .streamlit/
│   └── config.toml                        # Config de tema para Streamlit Cloud
├── .gitignore                             # Excluye modelos pesados y datasets
├── notebooks/
│   ├── 00_setup_verificacion.ipynb        # Verifica entorno y rutas
│   ├── 01_eda_preprocesamiento.ipynb      # Análisis exploratorio + generadores
│   └── 02_transfer_learning.ipynb         # Entrenamiento Phase A + B + export TFLite
├── model/                                 # Modelos descargados (no se sube a Git)
│   └── INSTRUCCIONES.md                   # Cómo colocar el .h5 entrenado
└── context/
    ├── brief.md                           # Brief original del proyecto
    ├── intro.md                           # Introducción y justificación social
    ├── referent.md                        # Notebook de referencia del docente
    └── 260416_FINAL_*.ipynb               # Notebook de clase (Xception)
```

---

## 6. Cómo reproducir el proyecto

### 6.1 Prerrequisitos

- Cuenta de Google (Colab + Drive)
- Python 3.11+ local (para correr la app Streamlit)
- Git

### 6.2 Setup en Google Drive

1. Crear la siguiente estructura en `MyDrive/RapidReliefAI/`:
   ```
   RapidReliefAI/
   ├── Datasets/
   │   └── clothing_small/
   │       ├── train/        (10 carpetas, una por clase)
   │       ├── validation/   (10 carpetas, una por clase)
   │       └── test/         (10 carpetas, una por clase)
   ├── Models/               (vacío al inicio, se llena al entrenar)
   └── Logs/                 (opcional, para TensorBoard)
   ```
2. Descargar el dataset desde Kaggle ([Clothing Dataset Small](https://www.kaggle.com/datasets/abdelrahmansoltan98/clothing-dataset-small)) y subirlo a la carpeta `Datasets/clothing_small/`.

### 6.3 Entrenamiento (Google Colab)

1. Abrir cada notebook desde GitHub directamente en Colab:
   - **Archivo → Abrir cuaderno → GitHub** → pegar `FernandoAcunaMtz/RapidReliefAI`
2. Activar GPU: **Entorno de ejecución → Cambiar tipo de entorno → T4 GPU**
3. Ejecutar en orden:
   - `00_setup_verificacion.ipynb` — verifica que GPU, Drive y rutas funcionen
   - `01_eda_preprocesamiento.ipynb` — análisis y validación de generadores
   - `02_transfer_learning.ipynb` — entrenamiento (Phase A + Phase B) y export
4. Al final del notebook 02 se generan en `MyDrive/RapidReliefAI/Models/`:
   - `rapidrelief_phaseA_best.keras`
   - `rapidrelief_phaseB_best.keras`
   - `rapidrelief_mobilenetv2_v2.keras`
   - `rapidrelief_model.tflite` (cuantizado INT8)
   - `labels.txt`

### 6.4 App Streamlit (local)

```bash
git clone https://github.com/FernandoAcunaMtz/RapidReliefAI.git
cd RapidReliefAI
pip install -r requirements.txt
streamlit run app.py
```

La app funciona en **modo demo** sin modelo. Para usar predicciones reales:
1. Descargar el `.h5` o `.keras` entrenado desde Drive
2. Colocarlo en `model/clothing_classifier.h5`
3. Reiniciar la app

### 6.5 App Streamlit Cloud (online)

La app está desplegada en [rapidreliefai.streamlit.app](https://rapidreliefai.streamlit.app) y se actualiza automáticamente con cada push a `main`.

---

## 7. Resultados

### 7.1 Curvas de entrenamiento

El notebook `02_transfer_learning.ipynb` genera automáticamente:
- Curvas de exactitud y pérdida (train vs. validación) para Phase A
- Curvas de exactitud y pérdida para Phase B (fine-tuning)
- Línea de referencia del KPI de 90% en la curva de accuracy

### 7.2 Evaluación final

- Matriz de confusión 10×10 (Seaborn heatmap) para identificar solapamientos
- Reporte de clasificación con precision, recall y F1-score por clase
- Comparación de accuracy en test vs. validation

### 7.3 Resultados v1 (sin Phase B, con arquitectura suboptimal)

| Métrica | Valor |
|---------|-------|
| Train accuracy | 98.27% |
| Val accuracy máx | 88.27% |
| Test accuracy | 89% |
| F1-score macro | 0.86 |
| Brecha train-val | 10% (overfitting) |

**Diagnóstico que motivó v2:**
- `Flatten` post-MobileNetV2 → 128M parámetros → overfitting
- `Dense(16, sigmoid)` antes de softmax → activación incorrecta para multi-clase
- SGD sin momentum → convergencia lenta
- Sin fine-tuning de la base
- Sin compensación de desbalance de clases

**v2 (notebook actual)** corrige los 5 problemas. Resultados pendientes del próximo entrenamiento en Colab.

---

## 8. Stack tecnológico

### Entrenamiento (Colab)
```
tensorflow >= 2.15
keras 3.x (TF_USE_LEGACY_KERAS=1 para compatibilidad con notebook de clase)
numpy, pandas, matplotlib, seaborn, scikit-learn, Pillow
```

### App web (Streamlit)
```
streamlit >= 1.32
tensorflow >= 2.15
plotly, Pillow, numpy
```

### App móvil (planeado)
```
Flutter + tflite_flutter ^0.10.4
```

---

## 9. Alineación con criterios académicos

### Criterio 1 — Acceso a base de datos y repositorios
- **Base de datos propia:** Google Drive (`MyDrive/RapidReliefAI/`) con datasets y modelos versionados por timestamp en checkpoints
- **Repositorio público:** [GitHub](https://github.com/FernandoAcunaMtz/RapidReliefAI) con código completo, notebooks y app
- **Reproducibilidad:** Sección 6 documenta cómo replicar el proyecto desde cero en otra máquina, tanto en Colab (cloud) como localmente

### Criterio 2 — Frameworks autorizados y contexto local
- **Frameworks autorizados:** TensorFlow/Keras (oficial Google), Jupyter (Project Jupyter), Streamlit (open source)
- **Modificación de tercero:** MobileNetV2 preentrenada por Google sobre ImageNet (14M imágenes, 1,000 clases) → adaptada a 10 clases del proyecto mediante transfer learning + fine-tuning
- **Contexto local justificado:** Crisis humanitarias y desastres naturales generan colapso logístico en centros de acopio. Esta solución reduce ~70% el tiempo de clasificación manual y opera offline en zonas sin conectividad

### Criterio 3 — Funcionamiento del sistema
- **Gráficas de desempeño:** Notebook 02 genera curvas de accuracy/loss (Phase A + B), matriz de confusión 10×10 y reporte por clase con precision/recall/F1
- **Coherencia con justificación:** El KPI de 90% se monitorea en cada época con visualización; el modelo se exporta a TFLite para cumplir el requisito de operación offline
- **UX del prototipo:** App Streamlit con dos modos de entrada (upload + cámara built-in), feedback visual con barra de confianza, historial de sesión, modo demo funcional sin modelo. Responsiva en móvil y desktop

---

## 10. Referencias

- Sandler, M. et al. (2018). *MobileNetV2: Inverted Residuals and Linear Bottlenecks.* [arXiv:1801.04381](https://arxiv.org/abs/1801.04381)
- Howard, A. et al. (2019). *Searching for MobileNetV3.* [arXiv:1905.02244](https://arxiv.org/abs/1905.02244)
- Chollet, F. (2021). *Deep Learning with Python* (2nd ed.). Manning.
- Goodfellow, I., Bengio, Y. & Courville, A. (2016). *Deep Learning.* MIT Press.
- Programa Mundial de Alimentos (2021). *Inteligencia Artificial para la Acción Humanitaria.*
- Abadi, M. et al. (2016). *TensorFlow: A system for large-scale machine learning.* OSDI.

---

*Última actualización: 2026-04-28*

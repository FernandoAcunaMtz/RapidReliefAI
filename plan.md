# RapidRelief AI — Plan de Desarrollo
**Clasificación Automatizada de Donaciones Textiles con CNN + Transfer Learning**

> Proyecto: Fernando Acuña Martínez, Pamela Ruíz Velasco Calvo, Said Lozada Vivar
> Docente: Dr. José Ambrosio Bastián | IA y Sistemas Expertos | Marzo–Mayo 2026

---

## 1. Arquitectura de Solución

```
┌─────────────────────────────────────────────────────┐
│  GOOGLE COLAB (GPU T4 gratuita)                     │
│  ├── Notebooks de entrenamiento (.ipynb)            │
│  ├── Google Drive → datasets + modelos guardados   │
│  └── Exportación → model.tflite + model.keras      │
└──────────────────────┬──────────────────────────────┘
                       │ TF Lite export
         ┌─────────────▼─────────────┐
         │   App Móvil (Flutter)     │
         │   + inferencia local      │
         │   + sin conexión          │
         └───────────────────────────┘
```

**Entorno de desarrollo primario:** Google Colab (GPU gratuita, integración Drive)
**Control de versiones:** Este repositorio Git (notebooks + app + documentación)
**Datasets:** descargados desde Kaggle, subidos a Google Drive

---

## 2. Selección del Modelo — Justificación Técnica

### Por qué MobileNetV2

De todos los modelos en [keras.io/api/applications](https://keras.io/api/applications/), **MobileNetV2** es el más adecuado para este proyecto:

| Criterio       | MobileNetV2 | EfficientNetB0 | ResNet50V2 |
|----------------|----------------------|----------------|------------|
| Tamaño         | **14 MB**               | 29 MB | 98 MB |
| Parámetros     | **3.5 M**               | 5.3 M | 25.6 M |
| Top-1 ImageNet | 71.3%                   | 77.1% | 75.6% |
| Velocidad CPU  | **~26 ms**              | ~46 ms | ~195 ms |
| TF Lite support| **Oficial, optimizado** | Bueno | Parcial |
| Raspberry Pi   | **Sí** | Marginal | No viable |

**Argumentos decisivos:**

1. **Diseñado para edge/móvil:** MobileNetV2 usa _depthwise separable convolutions_ + _inverted residuals_, reduciendo ~9× los FLOPs vs una CNN estándar equivalente. Esto garantiza inferencia en tiempo real en Raspberry Pi y smartphones de gama media.

2. **Validado en el notebook de referencia:** `260416_FINAL_Transferencia_de_aprendizaje_categorical.ipynb` usa exactamente MobileNetV2 con `include_top=False`, logrando 100% val_accuracy en 10 épocas sobre un dominio de imágenes reales. El patrón es directamente replicable.

3. **TF Lite oficial:** Google mantiene una versión optimizada de MobileNetV2 específicamente para TF Lite con cuantización INT8, reduciendo el modelo a ~4 MB sin pérdida significativa de precisión.

4. **`preprocess_input` escala a [-1, 1]** (no [0,1] como ResNet), lo que es más estable en inferencia móvil.

5. **La brecha de accuracy es recuperable:** La diferencia de ~6% en ImageNet entre MobileNetV2 y EfficientNetB0 se cierra durante el fine-tuning en dominio específico (moda/textiles). Estudios de TL en Fashion-MNIST muestran que MobileNetV2 fine-tuneado supera >90% en este dominio.

> **Alternativa documentada:** Si durante el Sprint 2 MobileNetV2 no supera 88% en validación, se migra a **EfficientNetB0** (mismo proceso, cambio de 2 líneas de código).

---

## 3. Estrategia de Datos

### Dataset único: Clothing Dataset Small

| Dataset | Imágenes | Clases | Tipo | Resolución |
|---------|----------|--------|------|------------|
| [Clothing Dataset Small](https://www.kaggle.com/datasets/abdelrahmansoltan98/clothing-dataset-small) | ~3,800 | 10 | RGB, fotos reales | Variable |

**Decisión sobre Fashion-MNIST:** Originalmente el plan combinaba Clothing Dataset Small con Fashion-MNIST (70,000 imágenes grayscale 28×28). Tras experimentar con la combinación se observó que la conversión grayscale → RGB de FMNIST introducía un desfase de dominio que degradaba la precisión sobre fotos reales. **Se descartó FMNIST** y se enfocó el entrenamiento exclusivamente en Clothing Dataset Small con augmentación robusta. La compensación se logra con `class_weight='balanced'` + augmentación aumentada (rotación, brillo, shifts).

### Distribución de clases (Clothing Small)

| Split | Imágenes |
|-------|----------|
| Train | 3,068 |
| Validation | 341 |
| Test | 372 |
| **Total** | **3,781** |

Clases (orden alfabético, asignación de `flow_from_directory`):
`0=dress, 1=hat, 2=longsleeve, 3=outwear, 4=pants, 5=shirt, 6=shoes, 7=shorts, 8=skirt, 9=t-shirt`

### Pipeline de preprocesamiento

```
Clothing Dataset Small (RGB variable)
    ↓
ImageDataGenerator + flow_from_directory
    ↓
Resize → 224×224
    ↓
preprocess_input MobileNetV2 (escala a [-1, 1])
    ↓
Augmentation (train) → batches (32, 224, 224, 3) + labels OHE (32, 10)
```

### Augmentation strategy

```python
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=15,
    width_shift_range=0.10,
    height_shift_range=0.10,
    shear_range=10.0,
    zoom_range=0.15,
    horizontal_flip=True,
    brightness_range=[0.85, 1.15],
    fill_mode='nearest'
)
```

Aumentos adicionales (rotación, brillo, shifts) añadidos en v2 para compensar la falta de FMNIST y reducir el sobreajuste observado en v1.

---

## 4. Arquitectura CNN (Transfer Learning + Fine-Tuning)

### Fase A — Feature Extraction (base congelada)

Arquitectura implementada en `notebooks/02_transfer_learning.ipynb`:

```python
base = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base.trainable = False

x = GlobalAveragePooling2D()(base.output)   # 7×7×1280 → 1280 (vs Flatten=62,720)
x = Dense(512, activation='relu')(x)
x = Dropout(0.3)(x)                         # regularización
output = Dense(10, activation='softmax')(x) # 10 clases

model = Model(inputs=base.input, outputs=output)
model.compile(
    optimizer=Adam(1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

**Justificación de cambios respecto a v1:**
- `GlobalAveragePooling2D` en lugar de `Flatten`: reduce 128M → 655K parámetros en la cabeza
- `Dense(16, sigmoid)` eliminada: `sigmoid` es para multi-etiqueta, no multi-clase
- `Dropout(0.3)` añadido: combate el overfitting observado en v1 (gap train-val ~10%)
- `Adam` en lugar de `SGD`: convergencia más rápida y estable para transfer learning

### Fase B — Fine-Tuning (descongelar últimas 54 capas)

```python
base.trainable = True
for layer in base.layers[:-54]:
    layer.trainable = False

model.compile(optimizer=Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
```

LR 100× menor (1e-5) para refinar las representaciones sin destruir los pesos preentrenados de ImageNet.

### Callbacks

```python
callbacks = [
    ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_accuracy'),
    EarlyStopping(patience=6, restore_best_weights=True, monitor='val_accuracy'),
    ReduceLROnPlateau(factor=0.5, patience=3, monitor='val_loss', min_lr=1e-6),
]
```

### Compensación del desbalance de clases

El test set tiene 73 zapatos vs. 12 faldas. Se aplica `class_weight='balanced'` en `model.fit()` para compensar:

```python
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight('balanced', classes=np.arange(10), y=train_data.classes)
model.fit(..., class_weight=dict(enumerate(class_weights)))
```

---

## 5. Plan Agile por Sprints

### Sprint 0 — Setup & Entorno (3 días)
**Objetivo:** Infraestructura lista, datasets descargados y accesibles

- [ ] Crear estructura de carpetas en Google Drive:
  ```
  MyDrive/RapidReliefAI/
  ├── datasets/
  │   ├── fashion_mnist/
  │   └── clothing_small/train|val|test/
  ├── models/
  └── logs/
  ```
- [ ] Descargar ambos datasets desde Kaggle (API o manual) y subirlos a Drive
- [ ] Crear notebook `00_setup_verification.ipynb` que valide:
  - GPU disponible en Colab (`tf.config.list_physical_devices('GPU')`)
  - Conteo de imágenes por clase en ambos datasets
  - Verificación de clases solapantes
- [ ] Crear repositorio Git con estructura de proyecto
- [ ] Verificar versiones: TF 2.x, Keras 3.x, Python 3.10+

**Entregable:** Drive configurado + notebook de verificación verde

---

### Sprint 1 — EDA y Preprocesamiento (5 días)
**Objetivo:** Pipeline de datos robusto y unificado

**Notebook:** `01_eda_and_preprocessing.ipynb`

- [ ] **EDA Fashion-MNIST**
  - Visualizar muestras por clase (grid 10×10)
  - Distribución de clases (confirmar balance 6,000/clase)
  - Estadísticas de píxeles (media, desviación)
- [ ] **EDA Clothing Dataset Small**
  - Conteo real de imágenes por clase
  - Visualizar muestras, detectar clases ruidosas o con pocas muestras
  - Análisis de resoluciones (histograma de tamaños)
- [ ] **Función de conversión Fashion-MNIST → RGB 224×224**
  - Implementar y verificar que `preprocess_input` funciona post-conversión
  - Validar que los valores caen en [-1, 1]
- [ ] **Generadores unificados** con `ImageDataGenerator` + `flow_from_directory`
- [ ] **Visualización de augmentation** (grid antes/después de transforms)
- [ ] **Split 85/15 estratificado** — confirmar distribución con seaborn countplot

**KPI de aceptación:** Generadores producen batches de shape `(32, 224, 224, 3)` con etiquetas OHE `(32, 10)`. Distribución de clases en train/test ≤ 2% de desviación.

---

### Sprint 2 — Transfer Learning: Feature Extraction (5 días)
**Objetivo:** Modelo base entrenado con accuracy ≥ 85% en validación

**Notebook:** `02_transfer_learning_feature_extraction.ipynb`

- [ ] Cargar MobileNetV2 con `include_top=False, weights='imagenet'`
- [ ] Congelar base completa (`base.trainable = False`)
- [ ] Agregar cabeza de clasificación (GlobalAvgPool → Dense 512 → Dropout → Dense 128 → Dropout → Softmax 10)
- [ ] Compilar con `Adam(lr=1e-3)` + `categorical_crossentropy`
- [ ] Entrenar 15–20 épocas con callbacks
- [ ] **Visualizaciones obligatorias:**
  - Curvas de accuracy y loss (train vs val)
  - Matriz de confusión (seaborn heatmap, 10×10)
  - Top-5 clases con más errores y sus imágenes de confusión
- [ ] Guardar modelo: `model.save('models/mobilenetv2_fe_v1.keras')`

**KPI de aceptación:** `val_accuracy ≥ 0.85` antes de fine-tuning.

> Si val_accuracy < 0.82 después de 20 épocas → migrar a EfficientNetB0 (documentar decisión).

---

### Sprint 3 — Fine-Tuning (4 días)
**Objetivo:** Modelo final con accuracy ≥ 90% — cumplir KPI técnico del brief

**Notebook:** `03_fine_tuning.ipynb`

- [ ] Cargar modelo guardado del Sprint 2
- [ ] Descongelar base desde capa 100 en adelante
- [ ] Recompilar con `Adam(lr=1e-5)` (LR 100× menor para no destruir pesos pre-entrenados)
- [ ] Entrenar 10–15 épocas adicionales con callbacks (EarlyStopping patience=5)
- [ ] Comparar métricas antes/después de fine-tuning en tabla
- [ ] **Análisis de errores:** visualizar las 20 predicciones más incorrectas con probabilidad de confianza
- [ ] Calcular métricas completas: precision, recall, F1-score por clase (sklearn classification_report)
- [ ] Guardar modelo final: `models/mobilenetv2_finetuned_final.keras`

**KPI de aceptación:** `val_accuracy ≥ 0.90`, F1-score ≥ 0.88 en todas las clases.

---

### Sprint 4 — Exportación TF Lite (3 días)
**Objetivo:** Modelo optimizado listo para deployment móvil/edge

**Notebook:** `04_export_tflite.ipynb`

- [ ] Convertir a TF Lite estándar (float32)
  ```python
  converter = tf.lite.TFLiteConverter.from_keras_model(model)
  tflite_model = converter.convert()
  ```
- [ ] Convertir con cuantización INT8 (reduce tamaño ~4×)
  ```python
  converter.optimizations = [tf.lite.Optimize.DEFAULT]
  converter.representative_dataset = representative_dataset_gen
  tflite_model_quant = converter.convert()
  ```
- [ ] Benchmark comparativo: tamaño de archivo + latencia de inferencia en Colab CPU
- [ ] Validar que la accuracy del modelo TF Lite cuantizado no cae > 1% vs float32
- [ ] Guardar: `models/rapidrelief_model.tflite` y `models/rapidrelief_model_quant.tflite`
- [ ] Generar `labels.txt` con las 10 clases en orden

**KPI de aceptación:** `model_quant.tflite` ≤ 6 MB, accuracy drop ≤ 1%.

---

### Sprint 5 — App Móvil (7 días)
**Objetivo:** App funcional que clasifique prendas en tiempo real (offline)

**Stack:** Flutter + tflite_flutter plugin

**Estructura:**
```
app/
├── lib/
│   ├── main.dart
│   ├── screens/
│   │   ├── home_screen.dart        # Cámara + resultado
│   │   └── history_screen.dart     # Últimas clasificaciones
│   ├── services/
│   │   └── classifier.dart         # Wrapper TF Lite
│   └── widgets/
│       ├── result_card.dart
│       └── confidence_bar.dart
├── assets/
│   ├── rapidrelief_model_quant.tflite
│   └── labels.txt
└── pubspec.yaml
```

- [ ] Setup Flutter project con `tflite_flutter: ^0.10.4`
- [ ] Implementar `ClassifierService` — carga modelo, preprocesa imagen, inferencia, decodifica resultado
- [ ] Pantalla principal: captura de cámara en tiempo real + overlay de resultado + barra de confianza
- [ ] Feedback visual: color según categoría (ropa de abrigo = azul, calzado = verde, etc.)
- [ ] Feedback sonoro: beep corto al clasificar (accesibilidad para operadores)
- [ ] Modo offline garantizado: modelo embebido en `assets/`
- [ ] Historial de últimas 20 clasificaciones (SharedPreferences)
- [ ] Testing en Android (emulador o dispositivo físico)

**KPI de aceptación:** Clasificación en < 300ms en dispositivo mid-range. Funciona sin internet.

---

### Sprint 6 — Demo, Documentación y Entrega (3 días)
**Objetivo:** Entregable académico completo y presentable

- [ ] Generar PDF de reporte técnico desde notebooks (nbconvert o Colab → PDF)
- [ ] Preparar presentación con:
  - Comparativa before/after (clasificación manual vs AI, tiempo medido)
  - Capturas de la app con ejemplos reales
  - Gráficas de entrenamiento y matriz de confusión final
  - Demo en vivo (o video grabado)
- [ ] README.md con instrucciones de replicación
- [ ] Notebook unificado `FINAL_RapidReliefAI.ipynb` (limpio, sin outputs de error, ejecutable de inicio a fin)
- [ ] Subir modelo final a Google Drive con link público para la presentación

---

## 6. Estructura del Repositorio

```
RapidReliefAI/
├── context/                          # Brief y referencias (ya existe)
│   ├── brief.md
│   ├── intro.md
│   └── 260416_FINAL_Transferencia_de_aprendizaje_categorical.ipynb
├── notebooks/                        # Desarrollo en Colab
│   ├── 00_setup_verification.ipynb
│   ├── 01_eda_and_preprocessing.ipynb
│   ├── 02_transfer_learning_feature_extraction.ipynb
│   ├── 03_fine_tuning.ipynb
│   ├── 04_export_tflite.ipynb
│   └── FINAL_RapidReliefAI.ipynb    # Versión entrega limpia
├── app/                              # Flutter app
│   ├── lib/
│   ├── assets/
│   └── pubspec.yaml
├── models/                           # Modelos exportados (gitignored si >100MB)
│   ├── .gitkeep
│   └── README.md                    # Links a Drive con los modelos
├── plan.md                           # Este archivo
└── .gitignore
```

---

## 7. Dependencias y Stack Tecnológico

### Google Colab (entrenamiento)
```
tensorflow==2.19.x
keras==3.13.x
numpy
matplotlib
seaborn
scikit-learn    # classification_report, confusion_matrix
Pillow          # manipulación de imágenes
```

### Flutter (app móvil)
```yaml
dependencies:
  tflite_flutter: ^0.10.4
  camera: ^0.10.5
  image: ^4.1.3
  shared_preferences: ^2.2.2
  flutter_tts: ^3.8.5    # feedback sonoro
```

---

## 8. KPIs y Criterios de Éxito

| KPI | Meta | Medición |
|-----|------|----------|
| Accuracy técnica | ≥ 90% | val_accuracy en notebook Sprint 3 |
| F1-score mínimo por clase | ≥ 0.88 | sklearn classification_report |
| Tamaño modelo TF Lite | ≤ 6 MB | ls -lh modelo.tflite |
| Latencia de inferencia | ≤ 300 ms | Benchmark en dispositivo real |
| Reducción tiempo clasificación | ≥ 70% | Demo comparativa cronometrada |
| Funcionalidad offline | 100% | Test con modo avión activado |

---

## 9. Riesgos y Mitigaciones

| Riesgo | Probabilidad | Mitigación |
|--------|-------------|------------|
| Clases desbalanceadas al combinar datasets | Media | Class weights en `model.fit()` |
| Overfitting por dataset pequeño (Clothing DS) | Alta | Augmentation agresivo + Dropout 0.4 |
| Accuracy < 90% con MobileNetV2 | Media | Migrar a EfficientNetB0 (2 líneas) |
| Fashion-MNIST grayscale degrada accuracy | Media | Probar también entrenamiento solo con Clothing DS |
| Sesgo entre clases similares (Shirt vs T-shirt) | Alta | Analizar matriz de confusión y ajustar thresholds |
| Límite de tiempo GPU en Colab gratuito | Media | Guardar checkpoints cada época en Drive |

---

## 10. Flujo de Trabajo Colaborativo

```
main branch
├── fernando/sprint-1-eda          ← Fernando: notebooks de datos
├── pamela/sprint-2-training       ← Pamela: entrenamiento y métricas
└── said/sprint-5-app              ← Said: app Flutter
```

- **Daily sync:** Comentarios en celdas de notebooks con fecha y autor
- **Definition of Done por sprint:** Notebook ejecutable de inicio a fin sin errores + KPI cumplido
- **Revisión de sprint:** Demo del notebook corriendo en Colab ante el equipo antes de mergear

---

*Plan generado: 2026-04-21 | Versión 1.0*

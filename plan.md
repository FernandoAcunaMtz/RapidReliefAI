# RapidRelief AI — Plan de Desarrollo
**Clasificación Automatizada de Donaciones Textiles con CNN + Transfer Learning**

> Proyecto: Fernando Acuña Martínez, Pamela Ruíz Velasco Calvo, Dijo Lozada Vivar
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

## 3. Estrategia de Datos (Dual Dataset)

### Datasets a combinar

| Dataset | Imágenes | Clases | Tipo | Resolución |
|---------|----------|--------|------|------------|
| [Fashion-MNIST](https://www.kaggle.com/datasets/zalando-research/fashionmnist) | 70,000 | 10 | Grayscale 28×28 | Baja |
| [Clothing Dataset Small](https://www.kaggle.com/datasets/abdelrahmansoltan98/clothing-dataset-small) | ~5,000 | ~10 | RGB, fotos reales | Alta |

### Mapeo de clases unificado

```
Clase Final         Fashion-MNIST           Clothing Dataset Small
─────────────────────────────────────────────────────────────────
0  T-shirt          T-shirt/top (0)         T-Shirt
1  Pantalón         Trouser (1)             Pants
2  Suéter           Pullover (2)            Longsleeve
3  Vestido          Dress (3)               Dress
4  Abrigo           Coat (4)                Outwear
5  Sandalia         Sandal (5)              Shoes (parcial)
6  Camisa           Shirt (6)               Shirt
7  Tenis            Sneaker (7)             Shoes (parcial)
8  Bolsa            Bag (8)                 — (solo FMNIST)
9  Bota             Ankle boot (9)          — (solo FMNIST)
```

> Clases 8 y 9 se nutren principalmente de Fashion-MNIST; el modelo aprende la textura/forma base de esas clases desde imágenes sintéticas y se generaliza con augmentation.

### Pipeline de preprocesamiento

```
Fashion-MNIST (grayscale 28×28)          Clothing DS (RGB variable)
        │                                         │
  Cargar con tf.keras.datasets            Cargar con ImageDataGenerator
        │                                         │
  Grayscale → RGB (stack canal ×3)        Resize → 224×224
        │                                         │
  Resize 28×28 → 224×224 (bicubic)        preprocess_input MobileNetV2
        │                                         │
  preprocess_input MobileNetV2            Augmentation (flip, rot, zoom)
        │                                         │
        └─────────────┬───────────────────────────┘
                      │
              Dataset combinado
              División: 85% train / 15% test
              (split estratificado por clase)
```

### Augmentation strategy

Basado en `context/referent.md` (Xception sobre Clothing Small): augmentation minimalista supera al agresivo en este dataset. Augmentation excesivo introduce ruido sin mejora de generalización.

```python
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    shear_range=10.0,       # validado en referent: mejor que rotation_range
    zoom_range=0.1,         # conservador, evita distorsión de prendas
    horizontal_flip=True,   # única transformación geométrica fuerte
)
# NO usar: rotation_range alto, brightness_range, width/height_shift — degradan accuracy en este dominio
```

---

## 4. Arquitectura CNN (Transfer Learning + Fine-Tuning)

### Fase A — Feature Extraction (base congelada)

Arquitectura validada en `context/referent.md` y adaptada a MobileNetV2:

```python
base = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base.trainable = False

inputs = keras.Input(shape=(224, 224, 3))
x = base(inputs, training=False)      # training=False mantiene BatchNorm congelado
x = GlobalAveragePooling2D()(x)       # mejor que Flatten: reduce parámetros y overfitting
x = Dense(100, activation='relu')(x)  # referent validó Dense(100) como óptimo, no 512
x = Dropout(0.2)(x)                   # referent: dropout=0.2 superó 0.0, 0.5 en val_accuracy
outputs = Dense(10)(x)                # SIN softmax — usar from_logits=True en la loss

model = Model(inputs=inputs, outputs=outputs)
model.compile(
    optimizer=Adam(1e-3),
    loss=CategoricalCrossentropy(from_logits=True),   # más estable numéricamente
    metrics=['accuracy']
)
```

### Fase B — Fine-Tuning (descongelar últimas capas)

```python
# Descongelar desde la capa 100 en adelante (últimos ~54 de 154 layers)
base.trainable = True
for layer in base.layers[:100]:
    layer.trainable = False

model.compile(optimizer=Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
```

### Callbacks obligatorios

```python
callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True, monitor='val_accuracy'),
    ReduceLROnPlateau(factor=0.5, patience=3, monitor='val_loss'),
    ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_accuracy'),
    TensorBoard(log_dir='./logs')
]
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
└── dijo/sprint-5-app              ← Dijo: app Flutter
```

- **Daily sync:** Comentarios en celdas de notebooks con fecha y autor
- **Definition of Done por sprint:** Notebook ejecutable de inicio a fin sin errores + KPI cumplido
- **Revisión de sprint:** Demo del notebook corriendo en Colab ante el equipo antes de mergear

---

*Plan generado: 2026-04-21 | Versión 1.0*

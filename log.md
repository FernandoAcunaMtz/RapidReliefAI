# RapidRelief AI — Bitácora de Conflictos y Resoluciones

**Proyecto:** Clasificación Automatizada de Donaciones Textiles  
**Equipo:** Fernando Acuña · Pamela Ruíz · Said Lozada  
**Instructor:** Dr. José Ambrosio Bastián · USB 2026

---

## 1. Plateau de MobileNetV2 — Val accuracy estancada en 0.8886

**Problema:** Después de cuatro versiones (v2–v4) con MobileNetV2, la precisión de validación no superaba 0.8886 a pesar de ajustes en class weights, Focal Loss (γ=2), augmentación y arquitectura de la cabeza.

**Intentos fallidos:** aumentar épocas, cambiar dropout, agregar Focal Loss, ajustar pesos por clase manualmente.

**Resolución:** Cambio de backbone a **EfficientNetB0**. En Phase A (base congelada, solo cabeza entrenada) se alcanzó **0.9062** en la primera ejecución, superando el KPI de ≥ 90%.

---

## 2. Phase B destruyó el modelo (0.9062 → 0.8739)

**Problema:** Al descongelar las últimas 60 capas de EfficientNetB0 para fine-tuning (Phase B, lr=1e-5), el modelo degradó de 0.9062 a 0.8739 en validación — peor que MobileNetV2.

**Causa:** EfficientNetB0 es más sensible al fine-tuning que MobileNetV2. Las capas BatchNormalization internas se desestabilizaron con el learning rate utilizado y el tamaño reducido del dataset (3,068 imágenes).

**Resolución:** Phase B eliminada por completo del notebook. El modelo final es el mejor checkpoint de Phase A.

---

## 3. KPI mostraba ✗ a pesar de haber alcanzado 0.9062

**Problema:** La celda de resumen del notebook mostraba el KPI como no cumplido (✗) aunque el entrenamiento había logrado 0.9062.

**Causa:** El resumen usaba `val_acc_B` (resultado degradado de Phase B, 0.8739) en lugar de `val_acc_A` para verificar el KPI.

**Resolución:** Se corrigió la celda de resumen para usar `max(history_A.history['val_accuracy'])`.

---

## 4. Bug crítico de normalización en app.py

**Problema:** La app Streamlit aplicaba la normalización de MobileNetV2 (`(x / 127.5) - 1.0`) después de cambiar el backbone a EfficientNetB0. Las predicciones eran silenciosamente incorrectas — sin error visible.

**Causa:** El código de `preprocess()` no fue actualizado al cambiar de backbone.

**Resolución:** Se reemplazó por la normalización torch de EfficientNetB0:
```python
x = (x / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
```

---

## 5. Incoherencia entre los tres notebooks

**Problema:** Al revisar los tres notebooks juntos se encontraron referencias mezcladas:
- `00_setup` y `01_eda` seguían importando MobileNetV2
- Parámetros de augmentación distintos entre `01_eda` y `02_transfer_learning`
- Visualización de imágenes augmentadas con reversión de normalización incorrecta

**Resolución:** Se alinearon los tres notebooks con EfficientNetB0: imports, preprocessing, augmentación (`rotation=20`, `shift=0.15`, `shear=15`, `zoom=0.20`, `brightness=[0.80,1.20]`) y reversión de normalización torch para visualización.

---

## 6. Error `Could not locate class 'Functional'` en Streamlit Cloud

**Problema:** Al cargar el modelo en Streamlit Cloud el sistema lanzaba un error fatal indicando que no podía encontrar la clase `Functional`.

**Causa:** El modelo fue guardado en Colab con `TF_USE_LEGACY_KERAS=1` (usa `tf_keras`), pero Streamlit Cloud cargaba con Keras 3 (`tf.keras`), que usa un formato de serialización incompatible.

**Resolución:** Se agregó `tf_keras` a `requirements.txt` y se cambió la carga del modelo a `tf_keras.models.load_model()`.

---

## 7. Doble normalización — todas las predicciones daban "Falda"

**Problema:** Después de corregir el error de `Functional`, el modelo cargaba correctamente pero todas las predicciones (zapatos, camisa, camiseta) devolvían "Falda" con confianza media.

**Causa:** EfficientNetB0 en tf_keras incluye capas internas de `Rescaling` y `Normalization` — el modelo ya normaliza los píxeles internamente. Al aplicar también la normalización torch en `preprocess()`, los valores de entrada eran completamente incorrectos (doble normalización).

**Resolución:** Se simplificó `preprocess()` para pasar píxeles crudos `[0–255]` sin ninguna normalización manual:
```python
x = np.array(img, dtype=np.float32)  # el modelo normaliza internamente
```

---

## 8. Incompatibilidad de Python 3.13 con el stack

**Problema:** Al intentar instalar dependencias localmente con Python 3.13, `pip` intentó compilar NumPy desde fuente (sin wheel disponible para 3.13) y falló por un carácter `ñ` en la ruta del usuario.

**Causa:** NumPy `<2.0.0` no tiene wheels precompilados para Python 3.13; TensorFlow tampoco soporta Python 3.13.

**Resolución:** Para ejecución local se requiere Python 3.11. Para el deploy en Streamlit Cloud no hay problema — la plataforma usa Python 3.11 por defecto al instalar `tensorflow>=2.15`.

---

## 9. Fashion-MNIST descartado como dataset complementario

**Problema:** Se evaluó agregar Fashion-MNIST para aumentar datos de entrenamiento.

**Causa del descarte:** Desfase de dominio severo — Fashion-MNIST son imágenes en escala de grises de 28×28 px con fondo negro, mientras el Clothing Dataset Small son fotografías reales en color de 224×224 px. Incluirlo introducía ruido que degradaba el clasificador en fotos reales.

**Resolución:** Dataset único: Clothing Dataset Small (3,068 train · 341 val · 372 test).

---

*Última actualización: 2026-04-29*

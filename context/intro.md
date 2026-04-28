RapidRelief AI – Clasificación Automatizada
de Donaciones Textiles
Optimización de la Logística Humanitaria mediante Visión por
Computadora (CNN + Transfer Learning)
Docente: Dr. José Ambrosio Bastián
Inteligencia Artificial y Sistemas Expertos
Alumnos: Fernando Acuña Martínez, Pamela Ruíz Velasco Calvo,
Said Lozada Vivar
Universidad Simón Bolívar · 2026

Introducción y Propósito
En situaciones de desastre natural o crisis de refugiados, la gestión de donaciones
textiles se convierte en un reto crítico. RapidRelief AI surge como una solución
tecnológica de alto impacto para eliminar los cuellos de botella logísticos. Utilizando
una Red Neuronal Convolucional preentrenada (MobileNetV2) con transferencia de
aprendizaje, el proyecto automatiza la clasificación de ropa, permitiendo que la
ayuda vital llegue a las víctimas de forma digna, organizada y en un tiempo récord.

Análisis de Impacto Social (Modelo Estratégico)
- Problema Social: Resolución del colapso logístico en centros de acopio. La
  clasificación manual es lenta y propensa a errores bajo estrés, retrasando la
  entrega de artículos de primera necesidad.
- Beneficiarios: Víctimas que reciben ayuda inmediata y voluntarios de ONGs
  que optimizan su esfuerzo físico y emocional.
- Justificación de IA: Una CNN procesa imágenes en milisegundos sin fatiga,
  garantizando un estándar de clasificación 24/7 inalcanzable para humanos
  en condiciones críticas.
- Privacidad y Ética: Uso exclusivo del Clothing Dataset Small (fotografías
  reales de prendas inanimadas en catálogo). Sin datos biométricos ni rostros,
  garantizando privacidad absoluta. El modelo final opera offline en el
  dispositivo, sin envío a servidores externos.

Estrategia de Datos:
- Preprocesamiento: preprocess_input de MobileNetV2 escala el rango [0, 255]
  a [-1, 1] para estabilidad numérica del gradiente.
- Compensación de desbalance: class_weight='balanced' en model.fit() ajusta
  la pérdida por la frecuencia de cada clase. Las clases minoritarias (hat,
  skirt) reciben mayor peso que las mayoritarias (t-shirt, longsleeve).
- Augmentación: rotación 15°, shear 10°, zoom 15%, flip horizontal, brillo
  ±15% y shifts del 10% incrementan la diversidad efectiva del conjunto de
  entrenamiento.
- Transformación: One-Hot Encoding (OHE) automático mediante
  flow_from_directory(class_mode='categorical').

- Interacción Humano-Sistema: Interfaz dual (App web Streamlit / App móvil
  Flutter) con captura de cámara integrada y feedback visual de confianza.
- Accesibilidad Técnica: Exportación a TensorFlow Lite con cuantización
  INT8. Ejecución local en hardware de bajo costo (Raspberry Pi) y
  smartphones, funcionando sin conexión en zonas de desastre.
- Evolución del Sistema: Implementación futura de Aprendizaje Activo para
  mejorar el modelo con casos atípicos revisados por expertos.
- Sostenibilidad: Modelo Open Source con financiamiento basado en
  subvenciones de "AI for Social Good".

Indicadores de Éxito (KPIs):
- Técnico: val_accuracy ≥ 90%, F1-score ≥ 0.88 por clase.
- Eficiencia: Modelo TFLite ≤ 6 MB, latencia de inferencia ≤ 300 ms.
- Social: Reducción ≥ 70% en el tiempo de clasificación logística manual.

Justificación Estadística de la División de Datos
El dataset (Clothing Dataset Small) viene preparticionado por su autor en Kaggle:
- Train:      3,068 imágenes (~81%)
- Validation:   341 imágenes (~9%)
- Test:         372 imágenes (~10%)
Total: 3,781 imágenes en 10 categorías (dress, hat, longsleeve, outwear, pants,
shirt, shoes, shorts, skirt, t-shirt).

Justificación de la división:
- Train/Val/Test estratificada: La partición original mantiene la distribución
  proporcional de cada clase entre splits.
- Validation independiente: Se evalúa cada época sin contaminar el test
  final, permitiendo usar EarlyStopping con restore_best_weights=True sin
  overfitting al conjunto de prueba.
- Compensación: El desbalance entre clases (hat/skirt: 12 imgs vs t-shirt:
  81 en validation) se compensa con class_weight='balanced' en lugar de
  balanceo manual del dataset, preservando la distribución natural del
  problema.

Decisión sobre Fashion-MNIST:
Originalmente el plan combinaba Clothing Small con Fashion-MNIST (70,000
imágenes grayscale 28×28). Tras experimentar con la combinación se observó
que la conversión grayscale → RGB de FMNIST introducía un desfase de
dominio que degradaba la precisión sobre fotografías reales (predicciones
incoherentes como "pants → shoes 36%"). Se descartó FMNIST y se enfocó el
entrenamiento exclusivamente en Clothing Dataset Small con augmentación
robusta y class weights, decisión documentada en el notebook
02_transfer_learning.ipynb.

Implementación Técnica (Transfer Learning)
- Arquitectura: MobileNetV2 preentrenada en ImageNet (154 capas, 3.5M
  parámetros) + GlobalAveragePooling2D + Dense(512, relu) + Dropout(0.3) +
  Dense(10, softmax).
- Estrategia en dos fases:
  Fase A (Feature Extraction): base congelada, Adam(lr=1e-3),
                               hasta 25 épocas con EarlyStopping.
  Fase B (Fine-Tuning): últimas 54 capas descongeladas,
                        Adam(lr=1e-5), hasta 20 épocas.
- Callbacks: EarlyStopping (patience=6, restore_best_weights=True),
  ReduceLROnPlateau (factor=0.5, patience=3, min_lr=1e-6),
  ModelCheckpoint (save_best_only=True, monitor='val_accuracy').
- Herramientas: TensorFlow/Keras, NumPy, Pandas (EDA), Matplotlib, Seaborn,
  scikit-learn (class_weight, classification_report, confusion_matrix), Pillow.
- Visualización: Curvas de accuracy/loss por fase, Matriz de Confusión 10×10
  (Seaborn heatmap), reporte por clase con precision/recall/F1 (sklearn).
- Exportación: TensorFlow Lite con cuantización INT8 mediante
  representative_dataset (objetivo ≤ 6 MB).

Fuentes de Justificación
Sandler, M., Howard, A., Zhu, M., Zhmoginov, A. y Chen, L.-C. (2018).
MobileNetV2: Inverted Residuals and Linear Bottlenecks. arXiv:1801.04381
(Justificación de la arquitectura optimizada para edge devices).

Goodfellow, I., Bengio, Y. y Courville, A. (2016). Deep Learning. MIT Press.
(Fundamento teórico sobre el uso de CNN para clasificación de patrones
visuales y técnicas de regularización).

Chollet, F. (2021). Deep Learning with Python (2ª ed.). Manning Publications.
(Justificación técnica de la arquitectura Keras, transfer learning y la
importancia de la normalización de datos).

Programa Mundial de Alimentos (PMA). (2021). Inteligencia Artificial para
la Acción Humanitaria. División de Tecnología del PMA. (Justificación del
impacto social de la IA en la optimización de cadenas de suministro
humanitario).

Abadi, M., et al. (2016). TensorFlow: A system for large-scale machine
learning. OSDI. (Justificación del uso de TensorFlow Lite para accesibilidad
en dispositivos móviles sin conexión).

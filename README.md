# RapidRelief AI — Clasificación Automatizada de Donaciones Textiles

**Proyecto académico** · Inteligencia Artificial y Sistemas Expertos  
Dr. José Ambrosio Bastián · Fernando Acuña Martínez, Pamela Ruíz Velasco Calvo, Dijo Lozada Vivar  

---

## ¿De qué trata el proyecto?

RapidRelief AI nace de un problema real y urgente: cuando ocurre un desastre natural o una crisis de refugiados, los centros de acopio reciben cantidades masivas de ropa donada que deben clasificarse manualmente. Ese proceso es lento, agotador y propenso a errores, especialmente cuando los voluntarios trabajan bajo estrés y en condiciones adversas. El resultado es un cuello de botella logístico que retrasa la entrega de artículos de primera necesidad a quienes más los necesitan.

La propuesta del proyecto es automatizar completamente esa clasificación mediante un modelo de visión por computadora. Un voluntario simplemente fotografía una prenda con su teléfono o con una cámara conectada a una estación local, y el sistema devuelve en milisegundos la categoría correspondiente: vestido, pantalón, camisa, calzado, abrigo, entre otras. Esto libera a los voluntarios de la tarea repetitiva de ordenar ropa y les permite concentrarse en actividades de mayor impacto humano.

El proyecto se enmarca éticamente en el uso de datos seguros: se trabaja exclusivamente con imágenes de productos inanimados de catálogo, sin ningún dato biométrico ni fotografías de personas, lo que elimina cualquier riesgo de privacidad. El modelo también está diseñado para funcionar sin conexión a internet, lo cual es esencial en zonas de desastre donde la conectividad no está garantizada.

---

## ¿Cómo se da solución mediante el modelo?

La solución técnica se construye sobre el principio de **transferencia de aprendizaje** (*transfer learning*). En lugar de entrenar una red neuronal desde cero —lo que requeriría millones de imágenes y semanas de cómputo— se parte de **MobileNetV2**, un modelo preentrenado por Google sobre ImageNet (más de 14 millones de imágenes y 1,000 clases). Ese modelo ya sabe detectar texturas, bordes, formas y patrones visuales complejos. Lo que se hace en este proyecto es reutilizar todo ese conocimiento y añadir una nueva cabeza de clasificación específica para las 10 categorías de prendas que interesan al proyecto.

El proceso ocurre en dos fases bien diferenciadas. En la primera, llamada *feature extraction*, el modelo base de MobileNetV2 se congela por completo (sus pesos no se modifican) y solo se entrena la nueva cabeza de clasificación, que aprende a mapear las representaciones visuales ya aprendidas hacia las 10 clases del proyecto. En la segunda fase, *fine-tuning*, se descongela la parte superior del modelo base para que ajuste finamente sus pesos al dominio específico de la moda y los textiles, usando una tasa de aprendizaje cien veces menor para no destruir el conocimiento previo.

Se eligió MobileNetV2 por encima de otras arquitecturas disponibles en Keras (EfficientNet, ResNet, Xception) por tres razones fundamentales. Primero, su tamaño es de apenas 14 MB y puede ejecutarse en tiempo real en dispositivos de bajo costo como Raspberry Pi o smartphones de gama media, lo cual es indispensable para el despliegue en zonas de desastre. Segundo, Google mantiene una versión oficial optimizada para TensorFlow Lite con cuantización INT8, que reduce el modelo a aproximadamente 4 MB sin pérdida significativa de precisión. Tercero, el notebook de referencia entregado por el docente demuestra que este patrón —MobileNetV2 con `include_top=False` más una cabeza personalizada— alcanza resultados sólidos sobre imágenes reales de prendas en pocas épocas de entrenamiento.

Los datos provienen de dos fuentes combinadas: el **Clothing Dataset Small** de Kaggle, que contiene alrededor de 5,000 fotografías reales de prendas en 10 categorías, y **Fashion-MNIST** de Zalando Research, que aporta 70,000 imágenes en escala de grises de artículos de moda a baja resolución. La combinación de ambos permite al modelo aprender tanto de imágenes estilizadas y controladas como de fotografías del mundo real con variaciones de iluminación, fondo y perspectiva.

El producto final del modelo se exporta a formato TensorFlow Lite y se embebe directamente en una aplicación móvil desarrollada en Flutter, que puede clasificar prendas en tiempo real, sin conexión a internet, en menos de 300 milisegundos.

---

## Los tres notebooks del proyecto

### `00_setup_verificacion.ipynb` — Verificación del Entorno

Este notebook es el punto de partida del proyecto y su único propósito es garantizar que toda la infraestructura esté correctamente configurada antes de comenzar cualquier trabajo de modelado. Comienza montando Google Drive (el entorno de almacenamiento centralizado del proyecto) y verificando que la sesión de Google Colab cuenta con una GPU disponible, lo cual es necesario para que el entrenamiento sea viable en tiempos razonables.

A continuación, define todas las rutas del proyecto (datasets, carpeta de modelos guardados) y verifica que cada una existe físicamente en Drive, reportando cualquier ruta faltante antes de que cause un error silencioso más adelante. Luego realiza un conteo detallado de imágenes por clase para el Clothing Dataset Small en sus tres particiones (train, validation, test), lo que permite detectar tempranamente si alguna clase tiene muy pocas muestras o si la distribución está desbalanceada.

El notebook también carga los CSVs de Fashion-MNIST y muestra la distribución de sus 70,000 ejemplos, indicando el mapeo que se aplicará para convertir las etiquetas originales a las clases del proyecto y señalando qué clase (bolsa, clase 8) será descartada por no tener equivalente en el dataset de prendas reales. Finalmente, hace una carga de prueba de MobileNetV2 para confirmar que la versión de TensorFlow instalada es compatible, e imprime un resumen visual con una muestra de cada clase de ambos datasets. Si todas las celdas ejecutan sin error, el entorno está listo para continuar.

---

### `01_eda_preprocesamiento.ipynb` — Análisis Exploratorio y Preprocesamiento

Este notebook construye el puente entre los datos crudos y los generadores de datos que alimentarán el modelo. Su trabajo es doble: entender los datos mediante análisis exploratorio y transformarlos en un formato compatible con MobileNetV2.

En la sección de análisis exploratorio, genera una tabla comparativa de la distribución de clases en las tres particiones del Clothing Dataset Small, acompañada de una gráfica de barras agrupadas que permite identificar visualmente clases desbalanceadas. También visualiza una imagen representativa de cada clase para que el equipo pueda verificar manualmente si los datos tienen ruido o si alguna categoría es ambigua. Para Fashion-MNIST, muestra las 9 clases utilizadas en escala de grises y explica el mapeo hacia las categorías del proyecto.

La parte de preprocesamiento resuelve el problema técnico más importante del proyecto: Fashion-MNIST son imágenes en escala de grises de 28×28 píxeles, mientras que MobileNetV2 requiere imágenes RGB de 224×224. El notebook implementa y valida una función de conversión que apila el canal de grises tres veces para crear una imagen RGB, luego redimensiona a 224×224 mediante interpolación bicúbica, y finalmente aplica la función `preprocess_input` de MobileNetV2 que escala los valores al rango [-1, 1]. Se incluye una verificación numérica que confirma que los valores resultantes caen exactamente en ese rango.

Finalmente, configura los tres generadores de datos (`ImageDataGenerator` con `flow_from_directory`) para entrenamiento, validación y test. El generador de entrenamiento aplica la estrategia de *data augmentation* validada en el notebook de referencia del docente: shear de 10 grados, zoom conservador de 10% y volteo horizontal, evitando rotaciones agresivas ni cambios de brillo que empíricamente degradan la precisión en este dominio. El notebook verifica que los batches producidos tienen la forma correcta `(32, 224, 224, 3)` con etiquetas en formato One-Hot Encoding `(32, 10)`, y muestra visualmente un batch de entrenamiento con los augmentos aplicados.

---

### `02_transfer_learning.ipynb` — Transferencia de Aprendizaje y Entrenamiento

Este es el notebook central del proyecto, donde se construye, entrena, evalúa y exporta el modelo. Sigue fielmente la metodología presentada en clase, adaptándola a las necesidades específicas del problema.

El notebook comienza recreando los generadores de datos del sprint anterior (para ser autocontenido y ejecutable de forma independiente en Colab) y luego carga MobileNetV2 con `include_top=False` y `weights='imagenet'`, congelando todos sus pesos con `trainable = False`. Sobre la salida del modelo base, construye una cabeza de clasificación con la API Funcional de Keras: una capa `Flatten`, seguida de capas `Dense` con 2,048 y 512 neuronas con activación ReLU, una capa intermedia con activación sigmoide, y finalmente una capa de salida `Dense` con 10 neuronas y activación `softmax` para la clasificación multiclase. El modelo se compila con pérdida `categorical_crossentropy` y optimizador SGD.

El entrenamiento se realiza por 10 épocas con un callback de `ModelCheckpoint` que guarda únicamente el mejor modelo (según `val_accuracy`) en Google Drive, siguiendo el mismo patrón del notebook de referencia. Durante el entrenamiento se monitorizan simultáneamente las métricas de entrenamiento y validación.

Una vez entrenado, el notebook genera las curvas de exactitud y pérdida (train vs. validación) para diagnosticar overfitting, y produce la **matriz de confusión** mediante Seaborn: una visualización de 10×10 que muestra, clase por clase, cuántas predicciones fueron correctas y cuáles se confundieron con categorías similares (el solapamiento más esperable en este dataset es entre "shirt" y "t-shirt"). Complementa esto con un reporte completo de clasificación de sklearn que incluye precision, recall y F1-score por cada una de las 10 clases.

El notebook incluye también una función de inferencia individual que permite probar el modelo con cualquier imagen nueva: carga la imagen, la preprocesa, obtiene el vector de probabilidades y muestra la predicción con su nivel de confianza. Se prueba tanto con imágenes del Clothing Dataset como con imágenes convertidas desde Fashion-MNIST, validando que el modelo generaliza entre las dos fuentes de datos.

Para cerrar el ciclo completo, el notebook exporta el modelo entrenado a **TensorFlow Lite** (formato `.tflite`) directamente desde Keras, mide el tamaño del archivo resultante y genera el archivo `labels.txt` con las 10 etiquetas en orden, que la aplicación móvil Flutter necesita para decodificar las predicciones. El resumen final imprime todas las métricas clave y verifica si el KPI técnico del proyecto (val_accuracy ≥ 90%) fue alcanzado.

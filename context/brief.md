RapidRelief AI – Clasificación Automatizada
de Donaciones Textiles
Optimización de la Logística Humanitaria mediante Visión por
Computadora (CNN)
Docente: Dr. José Ambrosio Bastián
Inteligencia Artificial y Sistemas Expertos
Alumnos: Fernando Acuña Martínez, Pamela Ruíz Velasco Calvo,
Said Lozada Vivar
Marzo 2026
Introducción y Propósito
En situaciones de desastre natural o crisis de refugiados, la gestión de donaciones textiles
se convierte en un reto crítico. RapidRelief AI surge como una solución tecnológica de alto
impacto para eliminar los cuellos de botella logística. Utilizando Redes Neuronales
Convolucionales (CNN), el proyecto automatiza la clasificación de ropa, permitiendo que la
ayuda vital llegue a las víctimas de forma digna, organizada y en un tiempo récord.

Análisis de Impacto Social (Modelo Estratégico)
Problema Social: Resolución del colapso logístico en centros de acopio. La
clasificación manual es lenta y propensa a errores bajo estrés, lo que retrasa la
entrega de artículos de primera necesidad.
Beneficiarios: Víctimas que reciben ayuda inmediata y voluntarios de ONGs que
optimizan su esfuerzo físico y emocional.
Justificación de IA: Una CNN procesa imágenes en milisegundos sin fatiga,
garantizando un estándar de clasificación 24/7 inalcanzable para humanos en
condiciones críticas.
Privacidad y Ética: Uso del conjunto de datos Fashion-MNIST (Zalando Research). Al
contener solo productos inanimados de catálogo, se garantiza la ausencia de datos
biométricos o rostros, respetando la privacidad absoluta.
Estrategia de Datos:
○ Limpieza: Normalización $[0, 1]$ para estabilidad del gradiente.
○ Balanceo: Conjunto de datos equilibrado (6.000 muestras/clase) para evitar sesgos
algorítmicos.
○ Transformación: Implementación de One-Hot Encoding (OHE) para una
clasificación multiclase precisa.
Interacción Humano-Sistema: Interfaz dual (App móvil/Estación local) con señales
visuales y sonoras para guiar al voluntario en el depósito de prendas.
Accesibilidad Técnica: Exportación a TensorFlow Lite . Ejecución local en
hardware de bajo costo (Raspberry Pi) y teléfonos inteligentes, funcionando sin conexión en zonas
de desastre.
Evolución del Sistema: Implementación de Aprendizaje Activo para mejorar el
modelo con casos atípicos revisados ​​por expertos.
Sostenibilidad: Modelo Open Source con financiamiento basado en subvenciones
de "AI for Social Good".
Indicadores de Éxito (KPIs):
○ Técnico: Precisión superior al 90%.
○ Social: Reducción del 70% en el tiempo de clasificación logística.
Justificación Estadística de la División de Datos (85/15)
2
El entrenamiento se diseñó bajo la proporción 85% Entrenamiento ($60,000$ imágenes) y
15% Prueba ($10,000$ imágenes) , fundamentado en:
● Ley de los Grandes Números: Un volumen de 60,000 ejemplos permite que la red
aprenda variaciones sutiles y texturas, mejorando la generalización ante fotos reales
no perfectas.
● Representatividad: Un conjunto de prueba de 10,000 imágenes es estadísticamente
robusto, asegurando que el error medido sea una representación fiel del rendimiento
en el mundo real.
● Eficiencia de Keras: Esta proporción evita el "hambre de datos" ( data starvation ) en
el entrenamiento, asegurando que el modelo aprenda patrones complejos antes de
ser validado.

Implementación Técnica (CNN)
● Arquitectura: Red Neuronal Convolucional con dos bloques de extracción de
características (Conv2D + MaxPooling2D) y una densa final de 128 neuronas.
● Herramientas: Keras, NumPy, Matplotlib y Seaborn (sin uso de Pandas para
optimizar memoria).
● Visualización: Matriz de Confusión mediante Seaborn para identificar y corregir
solapamientos entre categorías similares (ej. Camisa vs Camiseta).
Fuentes de Justificación
Xiao, H., Rasul, K. y Vollgraf, R. (2017). Fashion-MNIST: un nuevo conjunto de datos de imágenes
para realizar evaluaciones comparativas de algoritmos de aprendizaje automático . Investigación Zalando. Recuperado de
https://arxiv.org/abs/1708.07747 (Justificación del dataset como estándar de visión
computacional).
Goodfellow, I., Bengio, Y. y Courville, A. (2016). Aprendizaje profundo . Prensa del MIT.
(Fundamento teórico sobre el uso de CNN para clasificación de patrones visuales).
Chollet, F. (2021). Aprendizaje profundo con Python (2ª ed.). Publicaciones de Manning.
(Justificación técnica de la arquitectura Keras y la importancia de la normalización de
datos).
Programa Mundial de Alimentos (PMA). (2021). Inteligencia Artificial para
la Acción Humanitaria . División de Tecnología del PMA. (Justificación del impacto social de la IA en la
optimización de cadenas de suministro humanitario).
Abadi, M., et al. (2016). TensorFlow: un sistema para el aprendizaje automático a gran escala .
OSDI. (Justificación del uso de TensorFlow Lite para accesibilidad en dispositivos
móviles sin conexión).
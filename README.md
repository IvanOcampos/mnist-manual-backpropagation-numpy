# Red neuronal MNIST desde cero con NumPy

Implementación de un perceptrón multicapa para clasificar MNIST. Las capas densas, ReLU, Softmax, entropía cruzada y backpropagation están implementadas manualmente con NumPy. Matplotlib se usa para visualizar los datos, las activaciones y las métricas. TensorFlow/Keras se importa únicamente para descargar el conjunto MNIST.

## ¿Por qué funciona backpropagation?

La predicción de la red se construye aplicando operaciones una tras otra: cada capa combina sus entradas con pesos y bias, y las activaciones transforman el resultado. La pérdida mide cuánto se alejó la predicción de la etiqueta correcta. Como cada operación depende de la salida de la anterior, la regla de la cadena permite repartir el cambio de esa pérdida hacia atrás, desde la salida hasta cada peso.

En una capa densa, ese cálculo produce cuánto cambiaría la pérdida si cambiáramos cada peso o bias. El entrenamiento resta una fracción de esos gradientes a los parámetros. Así, para un paso pequeño, la pérdida tiende a bajar; repetir el proceso con muchos ejemplos ajusta las capas para que sus predicciones se acerquen a las etiquetas. En la salida Softmax junto con entropía cruzada, el gradiente respecto a los logits se simplifica a `(P - Y) / N`: indica en qué dirección y cuánto corregir las probabilidades, promediado sobre el lote.

Backpropagation no busca mágicamente los pesos correctos ni garantiza que cualquier entrenamiento converja: calcula de forma eficiente cómo contribuye cada parámetro al error actual. La inicialización, la tasa de aprendizaje, los datos y las iteraciones determinan si esas correcciones producen un buen modelo.

## Resultados registrados

La ejecución guardada en el notebook alcanza **97,48 % de accuracy en test** tras 15 epochs. Su matriz de confusión identifica **4 → 9** como el par de clases más confundido (25 casos) y al **8** como la clase con más errores totales (44 de 974). El notebook muestra imágenes de esos casos y discute una posible explicación visual. Los valores pueden cambiar al volver a entrenar.

# Metodología de entrenamiento

## Objetivo

El flujo implementado entrena clasificadores de audio para asignar cada fragmento a una de tres categorías: `no_violence`, `physical_violence` o `verbal_violence`. Además de esa clasificación multiclase, se define una decisión binaria de alerta que agrupa las dos categorías de violencia frente a `no_violence`.

El procedimiento separa el entrenamiento, la selección de configuraciones mediante validación y la evaluación final. Las decisiones sobre modelos, épocas y umbral se toman sin utilizar las predicciones de test.

## Preparación y control del corpus

El corpus se organiza en tres particiones —`train`, `val` y `test`— y en una carpeta por clase dentro de cada partición. Un archivo `metadata.csv` registra, como mínimo, el nombre del audio, su etiqueta, su partición, su origen y el grupo de fuente al que pertenece. Para identificar audios aumentados también se conserva una referencia a su archivo padre.

Antes de entrenar, el programa contrasta los metadatos con los archivos WAV y verifica las etiquetas y particiones permitidas, la unicidad de los nombres, la correspondencia entre registros y archivos, y la separación de grupos de fuente entre particiones. Los audios aumentados se admiten únicamente en entrenamiento; cada uno debe corresponder a un audio real de entrenamiento de la misma clase y fuente. Validación y test se reservan para audios con origen `real_clean`. La auditoría calcula hashes y una huella del corpus para detectar cambios entre fases.

Esta verificación depende de que los metadatos describan correctamente la procedencia de los audios. La adquisición del corpus, los criterios de anotación y la generación de las versiones aumentadas no se realizan íntegramente en este proyecto; deben documentarse por separado y no se infieren a partir del código de entrenamiento.

## Normalización y representación del audio

Cada archivo se carga como señal de punto flotante. Se rechazan los archivos vacíos o con valores no finitos. El preprocesamiento convierte la señal a mono, remuestrea a 16 kHz cuando es necesario, conserva como máximo los primeros cinco segundos y normaliza por el valor máximo absoluto. Las señales más cortas se rellenan con ceros hasta completar esa duración. El recorte se toma del inicio del archivo.

La representación posterior depende de la arquitectura:

- **MobileNetV3, ResNet50 y DenseNet121:** se calcula un espectrograma Mel de 128 bandas, con FFT de 1024 y salto de 512 muestras, y se convierte a decibelios. Los valores se normalizan por espectrograma; la representación se replica en tres canales y se normaliza con las medias y desviaciones estándar de ImageNet para adaptarla a los modelos visuales preentrenados.
- **Wav2Vec2:** se utiliza la forma de onda. Las muestras reales se estandarizan a media cero y varianza unitaria antes del relleno; una máscara distingue esas muestras de los ceros añadidos. La agregación temporal de las representaciones considera los frames correspondientes a la señal real.
- **Whisper:** el audio se transforma con `WhisperFeatureExtractor`, utilizando el checkpoint `openai/whisper-tiny`. El modelo se emplea para clasificación de audio, no para transcripción.

## Modelos y transferencia de aprendizaje

Se pueden entrenar cinco alternativas: MobileNetV3 Large, ResNet50 y DenseNet121 de Torchvision, junto con Whisper `openai/whisper-tiny` y Wav2Vec2 `facebook/wav2vec2-base`. En todos los casos, la capa o cabeza de clasificación se adapta a las tres etiquetas del proyecto.

Los modelos se inicializan con pesos preentrenados. En Wav2Vec2 se congela el extractor convolucional de características y se ajustan el encoder y la cabeza de clasificación. La cabeza combina de forma aprendible representaciones de las capas del modelo, las proyecta y realiza un promedio temporal con máscara antes de clasificar. SpecAugment temporal está activado por defecto en Wav2Vec2 y puede desactivarse desde la línea de comandos. No se congela explícitamente ninguna capa de las CNN ni de Whisper en esta implementación.

## Procedimiento de entrenamiento

La ejecución recibe como argumentos la ubicación del corpus y del directorio de salida, los modelos que se desean entrenar, el número de épocas, el tamaño del lote, la semilla y el número de procesos de carga. Si no se especifica un subconjunto, se procesan los modelos definidos por el proyecto. El programa selecciona CUDA cuando está disponible y, en caso contrario, ejecuta en CPU.

El entrenamiento es supervisado y utiliza entropía cruzada multiclase. Wav2Vec2 aplica suavizado de etiquetas de 0,05 durante entrenamiento; para validación se usa la pérdida sin suavizado. Se utiliza AdamW: las CNN tienen una tasa de aprendizaje de `1e-3`, Whisper de `5e-5`, y Wav2Vec2 separa el encoder (`2e-5`) de la cabeza y los pesos de combinación (`2e-4`). El decaimiento de pesos configurado es `1e-2`; para Wav2Vec2 se excluyen de este decaimiento los parámetros de una dimensión o menos.

Whisper y Wav2Vec2 utilizan un planificador lineal con calentamiento durante el 10 % de los pasos de entrenamiento y recorte de la norma del gradiente a 1,0. Las CNN no utilizan un planificador explícito en este flujo. El entrenamiento completa el número de épocas indicado: la implementación no detiene el proceso anticipadamente, aunque guarda el checkpoint que considera mejor según validación.

## Validación y selección del checkpoint

Al final de cada época, el modelo se evalúa sobre `val`. La selección del checkpoint sigue un orden lexicográfico: primero se prioriza el F1 macro; en caso de empate, el F1 ponderado; después, la exactitud; y, si persiste el empate, la menor pérdida. El entrenamiento continúa hasta consumir todas las épocas configuradas, independientemente de cuál haya sido la mejor época.

Una vez seleccionado el checkpoint, se vuelve a aplicar sobre validación para calcular las probabilidades y seleccionar el umbral de alerta. La puntuación de alerta es la suma de las probabilidades de `physical_violence` y `verbal_violence`, equivalente a uno menos la probabilidad de `no_violence`. Se elige entre los umbrales candidatos observados en validación y los extremos cero y uno, maximizando el F1 binario; los desempates favorecen mayor recall, menor tasa de falsos positivos y, finalmente, el umbral más alto. No se realiza una calibración probabilística explícita.

La validación también sirve para comparar modelos o variantes experimentales. El script `compare_wav2vec.py` permite contrastar Wav2Vec2 con SpecAugment activado y desactivado usando las mismas semillas y el mismo corpus. Esta comparación se basa en validación y no reemplaza la evaluación final.

## Evaluación final y separación de test

La fase de test se ejecuta por separado, una vez cerradas las decisiones experimentales. Utiliza el checkpoint y el umbral seleccionados durante validación; test no interviene en la elección de arquitectura, época, hiperparámetros ni umbral. Antes de evaluar, el programa comprueba que coincidan la huella del corpus, el orden de las etiquetas y el hash de los pesos; para Wav2Vec2 también verifica que la configuración registrada corresponda a la del modelo.

La auditoría inicial inspecciona también la estructura y los metadatos de test para comprobar la integridad del corpus, pero no calcula predicciones de esa partición durante la fase de entrenamiento. Una evaluación de test ya completada queda bloqueada por defecto para evitar reutilizarla como conjunto de decisión.

## Reproducibilidad y trazabilidad

Se fijan semillas para Python, NumPy y PyTorch. Los cargadores de datos reciben un generador con semilla, y PyTorch se configura para favorecer algoritmos deterministas. Estas medidas reducen fuentes de variación, pero no garantizan reproducibilidad bit a bit entre equipos, versiones de bibliotecas o dispositivos.

Para identificar cada ejecución, se conserva la configuración efectiva junto con la huella del corpus, el hash del checkpoint y las versiones principales del entorno. Para reconstruir el experimento también deben conservarse el comando utilizado, la versión exacta del código, las dependencias, la identificación del hardware y la procedencia de los pesos preentrenados. Los checkpoints guardados contienen los pesos del modelo, no el estado del optimizador ni del planificador, por lo que no constituyen por sí solos un mecanismo de reanudación completa.

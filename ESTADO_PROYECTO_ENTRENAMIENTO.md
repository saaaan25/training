# Estado técnico y metodología del proyecto de entrenamiento

> Documento de apoyo para redactar el informe con ayuda de un agente de IA. Describe la implementación disponible al 21 de octubre de 2026. Los resultados experimentales, sus valores y su interpretación quedan pendientes de incorporación.

## 1. Propósito y alcance

El proyecto implementa un flujo de entrenamiento y evaluación de modelos de aprendizaje profundo para clasificar violencia a partir de audio. La entrada es un archivo de audio y la salida corresponde a una de tres categorías:

| Etiqueta del sistema | Interpretación |
| --- | --- |
| `no_violence` | Audio etiquetado como ausencia de violencia. |
| `physical_violence` | Audio etiquetado como violencia física. |
| `verbal_violence` | Audio etiquetado como violencia verbal. |

Estas categorías representan las etiquetas del corpus. El código no define por sí mismo los criterios semánticos de anotación: el informe deberá explicar cómo se asignaron y revisaron las etiquetas.

Además de la clasificación multiclase, se implementa una decisión binaria de alerta que agrupa las dos categorías de violencia. Esto permite documentar dos tareas relacionadas: identificar el tipo de audio y decidir si debe emitirse una alerta.

El alcance de esta carpeta comprende la auditoría del corpus, la preparación de las entradas, la construcción de modelos, el entrenamiento, la selección mediante validación y la evaluación final separada. La adquisición del corpus, la anotación y la generación previa de audios aumentados requieren documentación adicional; no se implementan íntegramente aquí.

## 2. Estado funcional de la implementación

| Componente | Estado descrito en este documento |
| --- | --- |
| Auditoría de archivos y metadatos | Implementada en el código. |
| Preparación de audio | Implementada con representaciones específicas por familia de modelos. |
| Construcción de clasificadores | Implementada para las arquitecturas descritas más adelante. |
| Entrenamiento y selección de checkpoint | Implementados. |
| Selección del umbral de alerta | Implementada sobre validación. |
| Comparación de SpecAugment | Script de comparación implementado. |
| Evaluación final y exportación de reportes | Implementadas como fase independiente. |
| Resultados, tablas numéricas y conclusiones experimentales | Pendientes de documentar. |

“Implementado” describe una capacidad observada en el código; no constituye una afirmación sobre su rendimiento, su validación exhaustiva o la finalización de una ejecución particular.

## 3. Organización del código

| Archivo | Responsabilidad |
| --- | --- |
| [trainmodels.py](trainmodels.py) | Interfaz de línea de comandos, lectura de argumentos, elección del dispositivo e inicio de la fase solicitada. |
| [common.py](common.py) | Etiquetas, constantes, semillas, utilidades JSON y cálculo de hashes. |
| [audit.py](audit.py) | Verificación de metadatos, correspondencia con archivos, separación de fuentes y huella del corpus. |
| [data.py](data.py) | Lectura del audio, transformaciones y construcción de los cargadores de datos. |
| [models.py](models.py) | Arquitecturas, adaptación de cabezas clasificadoras, configuración y optimizadores. |
| [pipeline.py](pipeline.py) | Ciclo de entrenamiento, validación, selección, evaluación final y exportación de artefactos. |
| [metrics.py](metrics.py) | Métricas multiclase, métricas binarias, selección del umbral e intervalos bootstrap. |
| [compare_wav2vec.py](compare_wav2vec.py) | Comparación de Wav2Vec2 con y sin SpecAugment mediante semillas emparejadas. |
| [requirements.txt](requirements.txt) | Dependencias del proyecto, actualmente sin versiones fijadas. |
| [README.md](README.md) | Instrucciones operativas de instalación, entrenamiento y evaluación. |

La separación permite explicar en el informe qué módulo asegura la integridad del corpus, cuál transforma los datos y cuál controla el protocolo experimental.

## 4. Diseño del flujo experimental

El flujo implementado sigue esta secuencia:

1. Recibir la configuración del experimento por línea de comandos.
2. Seleccionar GPU CUDA si está disponible o CPU en caso contrario.
3. Auditar los metadatos y los archivos del corpus.
4. Construir el modelo y los cargadores de entrenamiento y validación.
5. Entrenar durante el presupuesto de épocas configurado.
6. Seleccionar el mejor checkpoint exclusivamente con validación.
7. Recuperar ese checkpoint y seleccionar el umbral binario con validación.
8. Guardar los artefactos necesarios para identificar y revisar la ejecución.
9. Ejecutar test en una fase independiente, después de cerrar las decisiones experimentales.

La fase de entrenamiento no calcula métricas predictivas de test. Sin embargo, la auditoría sí inspecciona los archivos y metadatos de todas las particiones, incluyendo test, para verificar integridad y construir la huella del corpus. Esta diferencia debe expresarse con precisión en el informe.

## 5. Corpus y particiones

### 5.1. Estructura requerida

Se espera un directorio con `metadata.csv` y tres particiones: `train`, `val` y `test`. Cada partición contiene una carpeta por clase.

```text
dataset/
  metadata.csv
  train/
    no_violence/
    physical_violence/
    verbal_violence/
  val/
    no_violence/
    physical_violence/
    verbal_violence/
  test/
    no_violence/
    physical_violence/
    verbal_violence/
```

El código reconoce perfiles de conteo para `semi_dataset` y `final_dataset`. También admite conteos esperados definidos por argumentos para otros nombres de corpus. Los tamaños observados, la distribución por clase y la composición por origen quedan pendientes para el apartado de datos del informe.

### 5.2. Metadatos y trazabilidad

Los campos requeridos incluyen nombre de archivo, etiqueta, partición, origen y grupo de fuente. Se acepta `source_parent` como alternativa a `source_group`. Para los registros aumentados se requiere una referencia al archivo padre, mediante `parent_file` o su alternativa `clean_parent`.

La auditoría comprueba:

- Presencia de columnas obligatorias y validez de etiquetas, particiones y orígenes.
- Unicidad de nombres de archivo y rutas registradas.
- Correspondencia entre los WAV esperados y los presentes en las carpetas inspeccionadas.
- Existencia de las carpetas de cada clase en cada partición.
- Ausencia de grupos de fuente compartidos entre particiones, según los metadatos.
- Restricción de los audios aumentados al entrenamiento.
- Uso exclusivo de `real_clean` en validación y test.
- Existencia de un padre real de entrenamiento para cada audio aumentado, con la misma clase y fuente.
- Cumplimiento de los conteos esperados cuando corresponde.

Se calcula un SHA-256 del CSV y una huella que incorpora metadatos, rutas, tamaños y hashes de los archivos. Esa huella permite detectar cambios del corpus entre entrenamiento y evaluación.

### 5.3. Límites de la auditoría

La separación de fuentes depende de que los identificadores del CSV representen correctamente la procedencia real. La auditoría no sustituye la revisión del proceso de particionado o de la calidad de las etiquetas.

Aunque calcula hashes de los WAV, el código actual no realiza una comprobación explícita que rechace contenido idéntico guardado con nombres distintos a partir de la igualdad de esos hashes. Tampoco detecta automáticamente similitud acústica, versiones recodificadas o fragmentos solapados. No debe describirse como una garantía completa contra toda forma de fuga de información.

## 6. Preparación del audio

### 6.1. Procesamiento común

El audio se lee con SoundFile en punto flotante. Se rechazan archivos vacíos o con valores no finitos. Cuando es necesario, se remuestrea a 16 kHz, se convierte a mono promediando canales y se conserva como máximo el inicio de cinco segundos. La señal se normaliza por su máximo absoluto cuando este es distinto de cero.

Las entradas más cortas se completan con ceros hasta alcanzar 80 000 muestras. El recorte es determinista y toma el principio del audio; no hay selección aleatoria de ventanas ni agregación de segmentos de un archivo largo.

Estos valores son parámetros de la implementación, no resultados experimentales. Su justificación metodológica deberá relacionarse con las características del corpus y los requisitos del sistema.

### 6.2. Representación para las CNN

Para MobileNetV3, ResNet50 y DenseNet121 se calcula un espectrograma Mel con:

| Parámetro | Configuración del código |
| --- | --- |
| Frecuencia de muestreo | 16 kHz |
| Tamaño de FFT | 1024 |
| Salto temporal | 512 muestras |
| Bandas Mel | 128 |
| Escala | Conversión a decibelios |
| Ajuste de valores | Normalización mínimo-máximo por espectrograma |
| Canales | Repetición de la misma representación en tres canales |
| Normalización final | Media y desviación estándar de ImageNet |

Los tres canales son copias del mismo espectrograma. Esta adaptación permite reutilizar modelos visuales preentrenados. El código no incluye un redimensionamiento explícito a una imagen cuadrada.

### 6.3. Representación para Wav2Vec2

Wav2Vec2 recibe la forma de onda. Antes del relleno, el fragmento real se estandariza a media cero y varianza unitaria, con un término de estabilidad numérica. El relleno permanece en cero.

La entrada incluye una máscara que distingue muestras reales de relleno. Después de obtener las representaciones del encoder y proyectarlas, el promedio temporal considera los frames correspondientes a muestras reales.

La máscara enviada al encoder depende de su configuración de normalización. Por ello, el promedio final con máscara no debe describirse como una eliminación universal de la influencia del relleno en todas las operaciones internas.

Existe un comentario en `data.py` que indica ausencia de estandarización por audio, pero las instrucciones ejecutables sí la aplican. Para documentar el comportamiento debe tomarse como referencia la operación implementada.

### 6.4. Representación para Whisper

Se utiliza `WhisperFeatureExtractor`, asociado al checkpoint configurado, para producir las características de entrada requeridas por Whisper. Este extractor aplica sus propias convenciones de representación y relleno; no se debe asumir que su tensor final tiene la misma duración representada que la entrada cruda de Wav2Vec2.

El flujo usa Whisper para clasificación de audio, sin transcripción ni análisis de texto transcrito.

## 7. Arquitecturas y transferencia de aprendizaje

| Nombre en el programa | Arquitectura o checkpoint del código actual | Adaptación |
| --- | --- | --- |
| `mobilenet_v3` | MobileNetV3 Large de Torchvision | Sustitución de la última capa clasificadora. |
| `resnet50` | ResNet50 de Torchvision | Sustitución de la capa `fc`. |
| `densenet121` | DenseNet121 de Torchvision | Sustitución de la capa clasificadora. |
| `whisper` | `openai/whisper-tiny` | Modelo de clasificación de audio con etiquetas del proyecto. |
| `wav2vec` | `facebook/wav2vec2-base` | Clasificación con proyección, agregación temporal y cabeza adaptada. |

Las CNN utilizan pesos preentrenados de Torchvision. Los modelos de audio cargan pesos mediante Transformers. Las capas de clasificación se adaptan al número de clases del proyecto.

En Wav2Vec2 se congela el extractor convolucional de características. El encoder Transformer y los parámetros entrenables de la cabeza se ajustan durante el entrenamiento. Para las CNN y Whisper no se establece una congelación explícita de capas en esta implementación.

Un comentario de `models.py` menciona XLSR, pero la constante usada para construir el modelo señala `facebook/wav2vec2-base`. El informe debe identificar el checkpoint efectivo, evitando trasladar ese comentario como descripción de la arquitectura.

## 8. Configuración de entrenamiento

### 8.1. Pérdida y optimización

Se usa entropía cruzada multiclase. Wav2Vec2 incorpora suavizado de etiquetas de 0,05 durante entrenamiento; la validación y el test usan entropía cruzada sin suavizado. No se aplican pesos de clase en la función de pérdida.

El optimizador es AdamW. La configuración actual establece:

| Modelo o grupo | Tasa de aprendizaje inicial |
| --- | --- |
| CNN | `1e-3` |
| Whisper | `5e-5` |
| Encoder entrenable de Wav2Vec2 | `1e-5` |
| Cabeza y pesos de agregación de Wav2Vec2 | `1e-4` |

Para Wav2Vec2 se separan grupos por tasa de aprendizaje y regularización. Los parámetros de una dimensión o menos se excluyen del decaimiento de pesos; los restantes usan `1e-2`. En los otros modelos se configura AdamW con decaimiento de pesos de `1e-2`.

Los modelos Transformer emplean un planificador lineal con calentamiento durante el 10 % de los pasos configurados y descenso posterior de la tasa. También se recorta la norma del gradiente a 1,0. Las CNN no tienen un planificador explícito en este flujo.

El número de épocas, tamaño del lote, semilla y número de procesos de carga se reciben por argumentos. Los valores por defecto de la interfaz no deben presentarse como los valores utilizados por una ejecución sin revisar su registro.

### 8.2. Selección del checkpoint

Al terminar cada época se evalúa validación. El checkpoint se selecciona mediante una comparación ordenada de:

1. Mayor F1 macro de validación.
2. Mayor F1 ponderado si persiste un empate.
3. Mayor exactitud si continúa el empate.
4. Menor pérdida si las métricas anteriores coinciden.

Se guarda el estado de pesos del mejor checkpoint. El entrenamiento continúa hasta completar el presupuesto configurado; no hay parada temprana implementada. Seleccionar una época anterior como mejor checkpoint no equivale a haber detenido el entrenamiento en ella.

La pérdida de entrenamiento de Wav2Vec2 no debe compararse directamente con la de validación como si fueran idénticos objetivos, porque solo la primera incluye suavizado de etiquetas. Además, las métricas de entrenamiento reúnen predicciones calculadas mientras el modelo cambia entre lotes.

## 9. Particularidades de Wav2Vec2

La configuración actual activa una combinación ponderada de las representaciones de sus capas. Los pesos de esa combinación se inicializan explícitamente y se normalizan con softmax durante la inferencia. Después se aplican proyección, promedio temporal con máscara y clasificación.

SpecAugment puede activarse o desactivarse desde la interfaz. Cuando está habilitado se usan máscaras temporales con probabilidad de 0,05 y longitud de cinco posiciones de la representación interna; no corresponden a cinco muestras de audio ni a cinco segundos. El enmascaramiento de características y LayerDrop están desactivados.

Se adapta el enmascaramiento temporal mediante `torch.where`. Esta implementación depende de métodos y utilidades internos de Transformers, lo que hace especialmente importante conservar las versiones de las dependencias.

La configuración de Wav2Vec2, su checkpoint, tasas de aprendizaje y una identificación del preprocesamiento se guardan en los registros nuevos de validación. La fase de test comprueba la compatibilidad de esos campos con el código disponible.

## 10. Reproducibilidad y conservación de evidencia

Se fijan semillas para Python, NumPy y PyTorch. Los cargadores incorporan un generador con semilla y una función de inicialización para sus procesos de trabajo. Se desactiva el benchmark de cuDNN y se solicitan algoritmos deterministas.

La llamada de PyTorch usa `warn_only=True`: las operaciones sin alternativa determinista pueden generar advertencias en lugar de detener la ejecución. Además, el contexto `quiet()` silencia advertencias y salidas durante las épocas. Por ello no corresponde prometer reproducibilidad exacta entre equipos o versiones.

Los registros incluyen versiones de paquetes principales, parámetros de entrenamiento, huella del corpus y hash del checkpoint. Para fortalecer la trazabilidad del informe se deberá conservar también:

- Una copia o revisión exacta del código utilizado, incluyendo cambios locales.
- Un archivo de dependencias con versiones fijadas.
- Identificación de GPU, controlador y configuración del dispositivo utilizado.
- Comando completo, fecha y directorio de salida de cada ejecución.
- Procedencia y revisión de los pesos preentrenados.
- Descripción del origen del corpus, particionado y aumentos.

`requirements.txt` no fija versiones y los pesos guardados no incluyen el estado del optimizador o del planificador. El flujo no implementa una reanudación completa de entrenamiento desde esos checkpoints.

## 11. Protocolo de alerta binaria

La puntuación de alerta se define como:

```text
p_alerta = 1 - p(no_violence)
alerta = p_alerta >= umbral
```

Con softmax, esta puntuación equivale a sumar las probabilidades de las categorías de violencia. La decisión binaria se obtiene de esa suma y puede diferir de agrupar directamente la clase ganadora del clasificador multiclase.

Se prueban los valores únicos observados en validación y los extremos cero y uno como umbrales candidatos. Se elige el que maximiza F1 binario; los desempates favorecen mayor recall, menor tasa de falsos positivos y, finalmente, mayor umbral.

Una vez elegido, el umbral se conserva para test. Su valor y sus métricas quedan pendientes de incorporación al informe. No debe interpretarse automáticamente como una probabilidad calibrada de violencia en condiciones reales, ya que no hay una fase explícita de calibración.

## 12. Comparación planificada de SpecAugment

`compare_wav2vec.py` organiza ejecuciones con SpecAugment activado y desactivado para cada semilla. Requiere al menos dos semillas distintas y controla que los registros coincidan con los parámetros esperados y con una misma huella de corpus.

Al completar todas las ejecuciones previstas, el script prepara:

- `runs.csv`: métricas de validación de cada combinación de variante y semilla.
- `comparison.json`: media y desviación estándar del F1 macro por variante, diferencias emparejadas y diferencia media.

La comparación se basa en validación. Sus resúmenes son descriptivos y no implementan una prueba de significación estadística. La documentación deberá distinguir el número de ejecuciones planificadas del número efectivamente completado, sin interpretar resultados parciales como una comparación definitiva.

La ejecución del script es secuencial y no incluye una recuperación automática de experimentos interrumpidos. Una carpeta con un checkpoint, sin su registro final, no basta para declarar completa una ejecución.

## 13. Evaluación final prevista

La evaluación de test se invoca como fase separada. Antes de predecir se verifica la huella del corpus, el orden de etiquetas y el hash del checkpoint. Para Wav2Vec2 también se comprueba su configuración registrada.

El flujo contempla métricas multiclase y binarias, reportes por clase, matrices de confusión y predicciones individuales. Incluye intervalos de confianza mediante bootstrap estratificado por clase, calculados con percentiles y el número de iteraciones solicitado.

Este bootstrap remuestrea archivos dentro de cada clase; no representa por sí solo la variabilidad entre entrenamientos, semillas o corpus. Si varios archivos de test proceden de una misma fuente, deberá valorarse si conviene una estimación que respete esa dependencia.

Un marcador `completed.json` impide repetir por defecto un test ya terminado. La opción de repetición se reserva para recuperación técnica. El protocolo del informe deberá cerrar arquitectura, configuración y umbral antes de consultar test.

Los registros antiguos que no contengan los campos exigidos por el código actual pueden requerir recuperar el código de su ejecución para evaluar correctamente. No se deben completar esos campos suponiendo que una configuración histórica coincide con la actual.

## 14. Cómo se documentarán los resultados — pendiente

Este apartado define la estructura de presentación futura. No incorpora valores, rankings ni conclusiones de los experimentos.

### 14.1. Ficha por experimento

Cada experimento deberá tener una ficha con identificador, objetivo, dataset y huella, particionado, arquitectura, checkpoint preentrenado, preprocesamiento, hiperparámetros, semilla, versiones, equipo, revisión del código y rutas de artefactos. Se indicará si terminó correctamente y si puede reconstruirse su configuración.

### 14.2. Tablas previstas

| Tabla del informe | Contenido que se incorporará | Fuente de evidencia |
| --- | --- | --- |
| Caracterización del corpus | Distribución por clase, partición, origen y grupo de fuente. | Metadatos y auditoría. |
| Configuración experimental | Parámetros efectivos y diferencias controladas entre ejecuciones. | Registro de validación, comando y código archivado. |
| Comparación en validación | Arquitectura, época seleccionada, pérdida, exactitud y F1. | `validation/*.json` y `validation_leaderboard.csv`. |
| Comparación de SpecAugment | Resultados por semilla y estadísticos emparejados. | `runs.csv` y `comparison.json`, una vez completos. |
| Clasificación final | Métricas globales y por clase en test. | `test/<modelo>/final_metrics.json` y `classification_report.csv`. |
| Alerta binaria | Umbral seleccionado, precisión, recall, F1, FPR, FNR y conteos de error. | Registro de validación y métricas finales de test. |
| Incertidumbre | Intervalos de confianza y método utilizado. | Campo `bootstrap_ci_95` de las métricas finales. |

Todas las tablas de resultados permanecen pendientes. Cuando se incorporen, cada una deberá identificar claramente la partición utilizada y el experimento de origen.

### 14.3. Figuras previstas

Las historias de entrenamiento permitirán generar curvas de pérdida y F1 por época, señalando el checkpoint seleccionado. Las matrices de confusión permitirán mostrar los errores entre clases. Las predicciones y la búsqueda de umbral permitirán estudiar el compromiso entre sensibilidad y falsas alertas.

Cada figura deberá incluir título, ejes, leyenda, partición, identificador de ejecución y una explicación de su interpretación. Se deberá distinguir una figura producida con validación de otra producida con test.

### 14.4. Análisis de errores

La revisión posterior podrá organizar los errores por clase, tipo de confusión, fuente y características acústicas verificables. Las interpretaciones deberán apoyarse en la revisión de los archivos correspondientes, sin inferir las causas de un error únicamente a partir de una probabilidad.

Se documentará cómo se seleccionaron los casos revisados para evitar presentar ejemplos aislados como representativos de todo el corpus. Las modificaciones sugeridas después de observar test deberán plantearse como trabajo futuro o evaluarse con un nuevo conjunto independiente.

### 14.5. Reglas de interpretación

- Separar los resultados de desarrollo de la evaluación final.
- Comparar ejecuciones controlando corpus, preprocesamiento, configuración y entorno.
- Informar variabilidad entre semillas cuando se disponga de repeticiones comparables.
- Considerar el tamaño de validación antes de interpretar diferencias pequeñas.
- Distinguir desempeño multiclase y desempeño de alerta binaria.
- Evitar atribuir una mejora a un único cambio si se modificaron varios factores.
- No generalizar el rendimiento a audio continuo o a condiciones reales sin una evaluación específica.

## 15. Artefactos y su función documental

La estructura de salida prevista es:

```text
directorio_experimento/
  dataset_audit_train.json
  checkpoints/best_<modelo>.pt
  histories/<modelo>_history.csv
  validation/<modelo>.json
  validation/<modelo>_predictions.csv
  validation/<modelo>_threshold_search.csv
  validation_leaderboard.csv
  dataset_audit_test.json
  test/<modelo>/confusion_matrix.csv
  test/<modelo>/classification_report.csv
  test/<modelo>/predictions.csv
  test/<modelo>/final_metrics.json
  test/<modelo>/completed.json
  test_summary.csv
```

La estructura representa lo que puede producir el código actual; no implica que todos los archivos estén presentes en todas las ejecuciones. Los registros y predicciones respaldarán las tablas futuras, mientras que los checkpoints y hashes permitirán identificar los pesos evaluados.

Las rutas de checkpoints se guardan como rutas absolutas. Si se trasladan los experimentos a otro equipo, deberá conservarse la identidad de los pesos y documentarse la actualización de sus ubicaciones.

## 16. Pendientes para completar el informe

1. Describir la procedencia del corpus, criterios de inclusión y procedimiento de anotación.
2. Explicar cómo se definieron los grupos de fuente y cómo se generaron las particiones.
3. Documentar las técnicas de aumento, sus parámetros y la relación con los archivos originales.
4. Justificar las representaciones, duración de entrada y arquitecturas elegidas.
5. Consolidar fichas reproducibles de las ejecuciones que se incluirán.
6. Definir el criterio de comparación y cerrar las decisiones antes del test.
7. Incorporar posteriormente las tablas y figuras experimentales con sus evidencias.
8. Redactar el análisis de errores, limitaciones, discusión y conclusiones a partir de la evaluación correspondiente.

Quedan fuera de las conclusiones actuales la superioridad de una arquitectura, el efecto de SpecAugment, la generalización del clasificador y su idoneidad operativa. Esas afirmaciones requieren evidencia que se documentará después.

## 17. Texto base para el apartado de metodología

> Se implementó un flujo modular para la clasificación de audio en las categorías de ausencia de violencia, violencia física y violencia verbal. El procedimiento incluye auditoría de metadatos y archivos, preparación de representaciones específicas por arquitectura, adaptación de modelos preentrenados y entrenamiento supervisado. La selección del checkpoint utiliza las métricas de validación, priorizando F1 macro. A partir del checkpoint seleccionado se determina también un umbral de alerta binaria sobre validación. La evaluación final se plantea como una fase separada, con pesos y umbral previamente fijados, y contempla métricas globales, reportes por clase, matrices de confusión e intervalos de confianza. Los resultados experimentales y su discusión se incorporarán una vez consolidado el protocolo y la evidencia correspondiente.

Este texto deberá complementarse con la descripción del corpus, la justificación de las decisiones y las referencias bibliográficas verificadas. Este documento se basa en archivos locales del proyecto y no constituye una revisión de literatura.

## 18. Instrucciones para el agente de IA que apoye la redacción

```text
Usa este documento como descripción técnica del proyecto de entrenamiento.

Redacta la metodología distinguiendo lo implementado de lo planificado y de lo
pendiente de verificar. Conserva el protocolo de separación entre entrenamiento,
validación y test. Describe los parámetros del código como configuración técnica,
sin presentarlos automáticamente como los de todas las ejecuciones históricas.

No incorpores todavía resultados, valores de métricas, rankings, comparaciones
empíricas ni conclusiones de rendimiento. Mantén esos apartados como pendientes
y explica qué tablas, figuras y artefactos se usarán para completarlos después.

No inventes fuentes del corpus, técnicas de aumento, criterios de anotación,
características del equipo, tiempos de entrenamiento ni referencias. Identifica
la información faltante y solicita evidencia cuando sea necesaria. No atribuyas
una mejora a un cambio sin un experimento que permita aislar su efecto.

Cuando se autorice incorporar resultados, vincula cada afirmación con su archivo
de evidencia, configuración y partición. Evita usar test para nuevas decisiones
de arquitectura, hiperparámetros o umbral.
```

# Evaluación con RAGAS
En esta rama presentamos una serie de scripts para llevar a cabo evaluaciones de datasets mediante [Ragas](https://docs.ragas.io/en/stable/). En nuestro caso, empleamos modelos de embeddings y modelos generativos [locales](https://docs.ragas.io/en/stable/howtos/integrations/langchain/) a través de [LangChain y HuggingFace](https://python.langchain.com/api_reference/huggingface/llms/langchain_huggingface.llms.huggingface_pipeline.HuggingFacePipeline.html#langchain_huggingface.llms.huggingface_pipeline.HuggingFacePipeline) para obtener los resultados de las métricas, además de aplicar [prompts adaptados](https://docs.ragas.io/en/stable/howtos/customizations/metrics/_metrics_language_adaptation/) al español.

## Jerarquía de archivos
* ``orig_prompts/`` → carpeta de prompts originales aplicados por Ragas. Son archivos JSON generados a partir de aplicar una función ``save_prompts()`` en las métricas de puntuación.

* ``prompts/`` → carpeta de prompts traducidos por Ragas. Comparten la misma estructura que los prompts originales. En realidad, lo que está realmente traducido son los ejemplos que usa Ragas en las instrucciones de métricas.

* ``results/`` → carpeta de datasets evaluados. Son archivos JSONL a modo de ejemplo de resultados de evaluación de Ragas, que en realidad es el mismo dataset que se proporciona de entrada (``dataset.jsonl``) con una columna añadida por cada métrica empleada, cuyo valor es su correspondiente puntuación.

* ``config.yaml`` → archivo de configuración para los scripts. Se especifican todos los parámetros que intervienen en el código compartido. En el propio fichero se indican los que se deben modificar.

* ``dataset.jsonl`` → ejemplo de dataset de entrada para aplicar evaluación con Ragas, incluyendo los atributos necesarios a incorporar.

* ``launcher.sh`` → script de lanzamiento para nodos de computación. Se debe incluir la ruta del archivo de configuración.

* ``main.py`` → script de ejecución principal. Inicia y configura los parámetros principales para realizar evaluaciones mediante la clase ``RAGEvaluator``.

* ``rag_evaluator.py`` → script de evaluación de métricas con Ragas. Entre ellas se pueden escoger:
    * [``LLMContextPrecisionWithoutReference``](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_precision/#context-precision-without-reference)
    * [``LLMContextPrecisionWithReference`` ](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_precision/#context-precision-with-reference)
    * [``LLMContextRecall``](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_recall/)
    * [``Faithfulness``](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/#faithfulness)
    * [``ResponseRelevancy``](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/answer_relevance/#response-relevancy)
    * [``ContextRelevance``](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/nvidia_metrics/#context-relevance)

## Flujo de ejecución
El fichero ``launcher.sh`` ejecuta el código ``main.py`` con la ruta del archivo de configuración ``config.yaml``. En este archivo de configuración **se deben modificar** rutas como la ubicación de los modelos, la ubicación del proyecto, la ubicación del dataset y la ubicación de salida de resultados. El resto de parámetros se han mantenido por defecto con la configuración que nos ofreció mejores resultados durante nuestros experimentos.

```
sbatch launcher.sh
```

### Carga del modelo local
En ``main.py`` se inicia una instancia de la clase ``RAGEvaluator`` (en ``rag_evaluator.py``) indicando el modelo de embeddings y el modelo generativo que se va a utilizar de manera local. Para ello, cargamos nuestros modelos localmente con HuggingFace mediante ``HuggingFacePipeline`` (consultar función ``RAGEvaluator._get_llm()``) dado el primer caso, o bien mediante ``HuggingFaceEmbeddings`` dado el segundo (consultar función ``RAGEvaluator._get_embeddings()``). Una vez cargados, se adecúan a Ragas con [LangChain](https://docs.ragas.io/en/stable/howtos/integrations/langchain/) mediante ``LangchainLLMWrapper`` y ``LangchainEmbeddingsWrapper`` respectivamente.

### Selección y/o traducción de prompts
 A continuación se establecen los prompts que van a intervenir en la puntuación de las métricas mediante la función ``RAGEvaluator.set_prompts()``. Su función es buscar dentro de la carpeta de prompts (especificada en ``config.yaml``) la existencia de los archivos predefinidos por Ragas en el mismo idioma indicado en la configuración. Si existen dichos archivos se cargan, y si no, se generan a través de la función ``adapt_prompts()`` de Ragas asignada a cada métrica particularmente.

> ⚠️ **ADVERTENCIA**
> La carpeta ``prompts/`` contiene ejemplos de prompts traducidos al español con los modelos de la configuración actual siguiendo el método establecido por Ragas. Dichas traducciones se realizaron en la versión ``ragas v0.2.15``, pero en la más reciente ``ragas v0.3.6``, aplicar el mismo flujo de ejecución nos da errores. Entendemos que se debe a la idoneidad de los modelos, puesto que Ragas espera principalmente LLMs como GPT.

> 💡 **CONSEJO** 
> Dada la configuración actual, la ejecución funcionará porque en el código se utilizarán los archivos de prompts que descargamos previamente. Para una experiencia personalizada en otro idioma que sea lo más similar posible a una ejecución real, recomendamos traducir manualmente los ``"examples"`` de los archivos de prompts de Ragas al idioma deseado y modificar los nombres de archivo o parámetros de configuración pertinentes, siempre respetando el orden y estructura actual que compartimos en este proyecto.

### Evaluación de datasets
Si todas las configuraciones son correctas, el archivo ``dataset.jsonl`` mantiene la misma estructura y la ejecución hasta este punto ha sucedido con normalidad, empezarán las evaluaciones del conjunto de datos con Ragas. Aunque existen métricas en Ragas que no requieren el uso de LLMs en su puntuación, en nuestro caso aplicamos aquellas que en las que sí intervienen (como se mencionó previamente). Tras esperar a que finalice la ejecución, los resultados deberían aparecer en ``results/`` o aquella carpeta de salida definida en el archivo de configuración.

## Explicación de métricas
De entre todas las [métricas disponibles](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/) en Ragas, a continuación mostramos aquellas que consideramos más significativas para la evaluación de RAGs. Normalmente, estas métricas solicitan la mayoría o la totalidad de los siguientes atributos implicados en un RAG:
* ``user_input (str)``: pregunta o consulta de entrada a un RAG.
* ``response (str)``: respuesta de salida del RAG.
* ``retrieved_contexts (list)``: lista de contextos recuperados por el RAG para generar la respuesta.
* ``reference (str)``: respuesta de referencia deseable.

### Context Precision
Esta métrica evalúa no solo cuántos fragmentos relevantes aparecen entre los primeros $K$ resultados, sino también en qué posiciones aparecen. 
Da más peso a los fragmentos relevantes que aparecen en posiciones más altas, reflejando la capacidad del sistema para priorizar la información útil.

$$\text{Context Precision@K} = \frac{\sum_{k=1}^{K} \left( \text{Precision@k} \times v_k \right)}{\text{Total number of relevant items in the top } K \text{ results}}$$

* $K$: número total de fragmentos considerados (por ejemplo, los primeros 5).
* $vk$: indicador de relevancia en la posición $k = 1$ si el fragmento en la posición $k$ es relevante, 0 si no lo es.
* $\text{Precision@k}$: la precisión calculada hasta la posición $k$.
* $\text{Denominador}$: número total de fragmentos relevantes presentes entre los primeros $K$ resultados

Si todos los fragmentos relevantes aparecen en los primeros puestos, el valor será alto. Si los fragmentos relevantes están dispersos o en posiciones bajas, el valor será menor. Esta métrica premia la aparición temprana de fragmentos relevantes y penaliza cuando estos aparecen tarde o no aparecen. En la versión [``LLMContextPrecisionWithoutReference``](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_precision/#context-precision-without-reference) se compara ``retrieved_contexts`` con ``response``, mientras que en la versión [``LLMContextPrecisionWithReference`` ](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_precision/#context-precision-with-reference) se compara ``retrieved_contexts`` con ``reference``.

$$\text{Precision@k} = {\text{true positives@k} \over  (\text{true positives@k} + \text{false positives@k})}$$
* $\text{true positives@k}$: número de fragmentos relevantes entre los primeros $k$.
* $\text{false positives@k}$: número de fragmentos no relevantes entre los primeros $k$.

$\text{Precision@k}$ mide la proporción de elementos relevantes entre los primeros $k$ resultados recuperados. Un valor alto indica que la mayoría de los fragmentos recuperados en los primeros puestos son relevantes para la consulta.
Un valor bajo indica que hay mucho ruido o fragmentos irrelevantes entre los primeros resultados.

#### Para una consulta se obtienen 3 fragmentos cuyas relevancias son:
* Fragmento 1: Irrelevante.
* Fragmento 2: Relevante.
* Fragmento 3: Relevante.

#### Los valores de Precision@k serán:
* $\text{Precision@1} = 0 / 1 = 0$ (el primero no es relevante).
* $\text{Precision@2} = 1 / 2 = 0,5$ (uno de los dos primeros es relevante).
* $\text{Precision@3} = 2 / 3 = 0,67$ (dos de los tres primeros son relevantes).

#### Para calcular Context Precision@K con k=3:
$$\text{Context Precision@3} = \frac{(0 \times v_1 = 0) + (0,5 \times v_2 = 1) + (0,67 \times v_3 = 1)}{\text{Total number of relevant items = 2}} = 0,585$$

### Context Recall
[``LLMContextRecall``](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_recall/) evalúa la capacidad de recuperación completa del sistema, asegurándose de que no se omita información importante que debería estar presente para responder correctamente a la pregunta.
Mide cuántos de los fragmentos o documentos relevantes de una consulta fueron efectivamente recuperados por el sistema.

Su valor varía entre 0 y 1, donde 1 significa que se recuperó toda la información relevante y 0 que no se recuperó nada relevante

$$\text{Context Recall} = \frac{\text{Number of claims in the reference supported by the retrieved context}}{\text{Total number of claims in the reference}}$$

1. El LLM divide ``reference`` en afirmaciones ($\text{claims}$).
2. Se verifica para cada $\text{claim}$ si puede ser inferida directamente de ``retrieved_contexts``.
3. Se calcula la métrica mediante la fórmula.

#### Ejemplo

La referencia se divide en 4 claims, de las cuales tres pueden ser inferidas de los contextos y una no. En este caso:
$$\text{Context Recall} = \frac{\text{Number of claims in the reference supported by the retrieved context} = 3}{\text{Total number of claims in the reference} = 4} = 0,75$$

### Faithfulness
[``Faithfulness``](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/#faithfulness) evalúa si todas las afirmaciones de la respuesta pueden ser justificadas únicamente con la información presente en los fragmentos recuperados, evitando así las alucinaciones o invenciones del modelo.

Mide cuán consistente y fiel es la respuesta generada por el sistema respecto al contexto recuperado.
El resultado es un valor entre 0 y 1, donde 1 indica que todas las afirmaciones están justificadas por el contexto y 0 que ninguna lo está.

$$\text{Faithfulness} = \frac{\text{Number of claims in the response supported by the retrieved context}}{\text{Total number of claims in the response}}$$

1. El LLM divide ``response`` en afirmaciones ($\text{claims}$).
2. Se verifica para cada $\text{claim}$ si puede ser inferida directamente de ``retrieved_contexts``.
3. Se calcula la métrica mediante la fórmula.

Un valor alto de $\text{Faithfulness}$ indica que la respuesta es totalmente consistente con el contexto recuperado. Sin embargo, un valor bajo indica que la respuesta contiene información no respaldada por el contexto, lo que puede sugerir la existencia de alucinaciones o errores factuales.

#### Ejemplo
* ``user_input``: ¿Dónde y cuándo nació Einstein? 
* ``retrieved_context``: Albert Einstein (nacido el 14 de marzo de 1879) fue un físico teórico alemán. 
* ``response``: Einstein nació en Alemania el 20 de marzo de 1879.

De esta respuesta, un modelo extrae las siguientes $\text{claims}$:

1. $\text{Einstein nació en Alemania.}$ → Esta afirmación sí se puede inferir del contexto.
2. $\text{Einstein nació el 20 de marzo de 1879.}$ → Esta afirmación no se puede inferir del contexto.

Con esto, podemos calcular la puntuación de $\text{Faithfulness}$.

$$\text{Faithfulness} = \frac{\text{Number of claims in the response supported by the retrieved context}=1}{\text{Total number of claims in the response}=2} = 0,5$$

### Response Relevancy
[``ResponseRelevancy``](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/answer_relevance/#response-relevancy) evalúa si la respuesta aborda directamente la intención de la pregunta, penalizando respuestas incompletas o que contienen información redundante o innecesaria.

Mide qué tan relevante es la respuesta generada respecto a la pregunta original del usuario.
Esta métrica no evalúa la veracidad de la respuesta, solo su pertinencia respecto a la pregunta.

$$\text{Answer Relevancy} = \frac{1}{N} \sum_{i=1}^{N} \text{cosine similarity}(E_{g_i}, E_o)$$

1. El LLM genera varias preguntas (por defecto 3) a partir de la respuesta.
2.  Se calcula la similitud de coseno entre el embedding de la pregunta original $E_0$ y el embedding de cada pregunta generada $E_{gi}$.
3. Se calcula la media de las similitudes de coseno mediante la fórmula anterior.

Una puntuación alta de la métrica indica que la respuesta es muy relevante y responde directamente a la pregunta. Por el contrario, una puntuación baja indica que la respuesta es incompleta, irrelevante o contiene información innecesaria.

#### Ejemplo
* ``user_input``: ¿Dónde está Francia y cuál es su capital?
    * Si la respuesta obtenida es ``response_1 = “Francia está en Europa occidental.”`` debería devolver una puntuación más baja.
    * Si la respuesta obtenida es ``response_2 = “Francia está en Europa occidental y su capital es París.”`` debería devolver una puntuación más alta.

### Context Relevance
[``ContextRelevance``](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/nvidia_metrics/#context-relevance) evalúa, mediante el razonamiento de un LLM, hasta qué punto los fragmentos de contexto recuperados (``retrieved_contexts``) son relevantes para responder a una pregunta de usuario (``user_input``). Dos prompts predeterminados están implicados en el cálculo de la métrica.

1. ``template_relevance1`` → "### Instructions\n\n"
"You are a world class expert designed to evaluate the relevance score of a Context"
" in order to answer the Question.\n"
"Your task is to determine if the Context contains proper information to answer the Question.\n"
"Do not rely on your previous knowledge about the Question.\n"
"Use only what is written in the Context and in the Question.\n"
"Follow the instructions below:\n"
"0. If the context does not contains any relevant information to answer the question, say 0.\n"
"1. If the context partially contains relevant information to answer the question, say 1.\n"
"2. If the context contains any relevant information to answer the question, say 2.\n"
"You must provide the relevance score of 0, 1, or 2, nothing else.\nDo not explain.\n"
"### Question: {query}\n\n"
"### Context: {context}\n\n"
"Do not try to explain.\n"
"Analyzing Context and Question, the Relevance score is "

2. ``template_relevance2`` → "As a specially designed expert to assess the relevance score of a given Context in relation to a Question, "
"my task is to determine the extent to which the Context provides information necessary to answer the Question. "
"I will rely solely on the information provided in the Context and Question, and not on any prior knowledge.\n\n"
"Here are the instructions I will follow:\n"
"* If the Context does not contain any relevant information to answer the Question, I will respond with a relevance score of 0.\n"
"* If the Context partially contains relevant information to answer the Question, I will respond with a relevance score of 1.\n"
"* If the Context contains any relevant information to answer the Question, I will respond with a relevance score of 2.\n\n"
"### Question: {query}\n\n"
"### Context: {context}\n\n"
"Do not try to explain.\n"
"Based on the provided Question and Context, the Relevance score is  ["

El LLM recibe el prompt ``template_relevance1`` que contiene ``user_input`` y ``retrieved_contexts`` incrustados, junto con instrucciones claras para puntuar la relevancia del contexto con tres opciones:

* 0 → el contexto no es relevante para la pregunta.
* 1 → parcialmente relevante.
* 2 → completamente relevante.

La puntuación final se normaliza en el rango $[0, 1.0]$ y se promedia sobre dos ejecuciones del LLM para mayor robustez. Si alguna ejecución falla o da una salida inválida, se reintenta hasta 5 veces por defecto. En casos límite (como cuando el contexto está vacío o es idéntico a la pregunta), la puntuación se fuerza a 0 directamente.

#### Ejemplo
* ``user_input``: "When and Where Albert Einstein was born?"
* ``retrieved_contexts``: ["Albert Einstein was born March 14, 1879.", "Albert Einstein was born at Ulm, in Württemberg, Germany."]

Al LLM se le solicitan dos plantillas distintas (``template_relevance1`` y ``template_relevance2``) para evaluar la relevancia de los contextos recuperados en relación con la consulta del usuario. Cada pregunta devuelve una valoración de relevancia de 0, 1 o 2.

Cada valoración se normaliza a una escala $[0, 1.0]$ dividiéndola por 2. Si ambas valoraciones son válidas, la puntuación final es la media de estos valores normalizados; si sólo una es válida, se utiliza esa puntuación.

En este ejemplo, los dos contextos recuperados responden plenamente a la consulta del usuario, ya que proporcionan tanto la fecha de nacimiento como el lugar de nacimiento de Albert Einstein. Por lo tanto, ambas preguntas puntuarían los contextos combinados con 2 (totalmente relevantes). Al normalizar cada puntuación se obtiene $2 / 2 = 1.0$, y al promediar los dos resultados se mantiene la puntuación final en $1.0$.

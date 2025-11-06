# 📚 RAG ILENIA

Este proyecto es un **chatbot basado en RAG** (Retrieval-Augmented Generation) para el proyecto **ILENIA** que responde preguntas utilizando información de documentos indexados en **Elasticsearch**.  


## 🚀 Ejecución

#### 📦 Instalación de librerías

Antes de ejecutar el sistema, asegúrate de instalar las librerías necesarias.  
1. Crea (opcional pero recomendado) y activa un entorno virtual:
```bash
python -m venv venv
source venv/bin/activate     # En Linux/Mac
venv\Scripts\activate        # En Windows
```

2. Instala las dependencias indicadas en ``requirements.txt``:
```bash
pip install -r requirements.txt
```

#### 🚀 Lanzar sistema
Para lanzar el sistema, despues de configurar correctamente `config_base.json` ejecuta el script (se asume que ElasticSearch ya está corriendo en modo servidor):

```bash
./launch_rag.sh
```
A continuación se muestra una captura del sistema en funcionamiento:
![Captura de la interfaz](screenshot.png)

> ⚡ **Nota:** El **Reranker** se ejecuta en **GPU** cuando está disponible, ya que el procesamiento se realiza en local.  
> Esto acelera significativamente el paso de *reranking* de los documentos antes de la generación de respuestas.



## 📂 Estructura del proyecto
- `launch_rag.sh` → Script de inicio que ejecuta el sistema completo.
- `rag_interface.py` → Contiene la **interfaz gráfica** desarrollado con *Gradio*. 
  - Es el punto de entrada principal del sistema.
  - Requiere como entrada los archivos de configuración:
    - `config_base.json` → Configuración general del sistema. **Es importante configurar los endpoints de los modelos y de ElasticSearch antes de lanzar el sistema, ya que de lo contrario NO VA A FUNCIONAR**
    - `es_indices.json` → Configuración de los índices de Elasticsearch.
- `rag_base.py` → Implementa la **lógica del RAG**. Clase RAG con los métodos principales:
    - *generate_response* → Pipeline de respuesta a partir de la query del usuario:

            1. Recupera documentos de Elasticsearch.  
            2. Reordena los documentos con el Reranker.  
            3. Selecciona el Top-k y los pasa al modelo generativo como contexto.  
            4. Obtiene la respuesta final.  
    - *search_documents* → Búsqueda de documentos relevantes en Elasticsearch.
- `reranker.py` → Define la clase **Reranker** y el método de reranking de documentos para priorizar los más relevantes.
- `style.css` → Archivo de estilos para personalizar el diseño de la interfaz.
- `requirements.txt` → Lista de librerías necesarias para ejecutar el proyecto.
- `utils/` → Directorio con funciones auxiliares y utilidades.


## Fichero de índices de ElasticSearch (`es_config.json`)
El fichero de configuración para ElasticSearch es un JSON donde la clave es el identificador o nombre genérico para el indice y el valor es a su vez un JSON con distintos campos de configuración. Por cada índice tendrá que tener una entrada como la que se muestra a continuación:

      {
        "nombre": {
            "bm25_index": "indice de elasticsearch indexado por texto",
            "embedding_index": "indice de elasticsearch indexado con embeddings",
			"embedding_field": "campo donde se almacena el embedding",
			"embedding_text_field": "texto asociado al embedding anterior (es decir, el chunk del que se ha hecho el embedding",
            "highlight_field": "(texto) campo para extraer el fragmento con mejor matching",
            "search_fields": [
                "campos para realizar la busqueda textual"
            ],
            "filter": "",
            "metadata_fields":{ 
	            // campos para incluir en el metadata
                "title": "titulo",
                "date": "fechaPublicacion",
                "url": "url",
            },  
            "formatted_text": "Plantilla fstring del texto que se le pasa al LLM como contexto",
            "formatted_text_fields": [
                "campos para insertar en el formatted_text"
            ] 
        }
    }

### Explicación de los campos


Los primeros campos son para indicar dónde y como hacer la búsqueda:
 - bm25_index y embedding_index son los nombres de los indices de ElasticSearch donde se van a consultar los datos, indexados textualmente o con embeddings respectivamente.
 - highlight_field es el campo que se tendrá en cuenta para tomar un highlight en caso de hacer búsqueda textual
 - embedding_field  es el campo donde se han indexado los embeddings en el índice de ElasticSearch
 - embedding_text_field es el campo donde se almacena el texto del que se ha hecho el embedding del campo anterior (es decir, el chunk de texto del que se hace el embedding) -- el objetivo de este campo es para mostrar el texto después en la interfaz
 - search_fields son los campos que se van a querer utilizar para realizar la búsqueda en caso de hacer búsqueda textual. Se pueden indicar varios campos (titulo, texto, etc) y todos ellos se tendrán en cuenta a la hora de buscar
 - filter es para indicar los filtros de búsqueda (no está implementado aún)

Los siguientes campos son para configurar la información que se le pasará al modelo y la información que se va a mostrar por la interfaz:
 - metadata_fields son los campos que se van a pasar como metadatos. Los tres campos incluidos en el ejemplo (title, date, url) son los que se utilizan para visualizar en las fuentes, asi que habrá que indicar de qué campos de nuestros JSONs queremos cogerlos.
 ![metadata_fields visualization](https://i.ibb.co/fVbTy2Fh/example.png)
 - formatted_text y formatted_text_fields es para indicar como se le va a pasar la información como contexto al LLM. Por ejemplo, en formated text se podría indicar algo como `f" ({oraganismo}) **{titulo}**: {text} (Fecha de publicacion: {fechaPublicacion})"` por lo que el modelo recibiría cada texto incluyendo el organismo que lo publica, titulo y su fecha de publicación, y en formatted_text_fields habría que poner el nombre de los campos para poblar el fstring (organismo, titulo y fechaPublicacion) EXCEPTO `text` , el cual es el ÚNICO CAMPO OBLIGATORIO y se completa con el texto obtenido en el retrieval.

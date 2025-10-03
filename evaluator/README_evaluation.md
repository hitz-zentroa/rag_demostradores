# Evaluación RAGs ILENIA

En estos scripts se incluyen diversos scripts para realizar la evaluación de un sistema RAG. Actualmente soporta dos enfoques:
- Métricas tradicionales de IR basadas en juicios de relevancia: precision, recall, mrr.
- Métricas basadas en llm-as-a-judge: [Context Precision](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_precision/) y [Context Recall](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_recall/)

## Ejecución
```bash
sh launch_evaluation.sh
```

Este script ejecuta la evaluación de todas las métricas, admitiendo en cada caso evaluar un solo fichero de resultados o un directorio con varias ejecuciones. Las métricas se almacenan en tres ficheros jsonl (uno para las métricas tradicionales y uno por cada métrica de juez).

## Estructura

La carpeta `evaluator/` contiene la siguiente organización de archivos y subcarpetas para la evaluación de sistemas RAG:

```
evaluator/
├── launch_evaluation.sh                # Script principal para lanzar todas las evaluaciones
├── README_evaluation.md                # Documentación y guía de uso
├── traditional-ir/                     # Evaluación tradicional de IR
│   ├── evaluator_traditional.py        # Script de evaluación tradicional
│   └── ir_metrics.py                   # Definición de las métricas tradicionales
└── llm-as-a-judge/                     # Evaluación basada en LLM-as-a-judge
    ├── evaluator_judge.py              # Script de evaluación con LLM como juez
    ├── prompts.py                      # Prompts utilizados por el juez LLM
    ├── judge_metrics.py                # Implementación de métricas de juez
    └── Judge.py                        # Clases para los modelos de juez
```

- **`launch_evaluation.sh`**: Script que automatiza la ejecución de todas las métricas sobre los ficheros de resultados.
- **`traditional-ir/`**: Carpeta con scripts y utilidades para evaluar con métricas tradicionales de IR (precisión, recall, MRR).
- **`llm-as-a-judge/`**: Carpeta con scripts y utilidades para evaluar con métricas basadas en LLM como juez (context precision, context recall).

Cada subcarpeta contiene los scripts principales de evaluación y los módulos auxiliares necesarios para calcular las métricas correspondientes.

## Formato

Los scripts están pensados para trabajar con ficheros de resultados en formato `json` con la siguiente información:
```javascript
{
    consulta: "",
    respuesta_referencia: "",
    contexto_referencia: "",
    contextos_recuperados:{
        contexto1: {

        },
        contexto2: {

        }
    }

}
```
Los scripts que controlan el flujo en cada evaluación (`llm-as-a-judge/evaluator_judge.py`y `traditional-ir/evaluator_traditional.py`) tienen al principio una función llamada `extract_eval_fields()`, la cual se puede modificar para adaptar cómo se cogen los datos de los ficheros de resultados. La idea sería que, mientras no tengamos un marco de evaluación común, cada grupo la edite para que se ajuste al formato que emplea. Por ejemplo, en Nós, como estamos evaluando a nivel de párrafo a mayores de documento, el fichero de entrada tiene esta pinta:

```javascript
[
    {
        "id": 0,
        "user_input": "Cal é o propósito do sistema ES-Alert implementado polo Ministerio do Interior?",
        "reference_answer": "O sistema ES-Alert ten como obxectivo enviar alertas á poboación que se atope en zonas afectadas..."
        "reference_source_id": "codigo_cero01",
        "reference_context": "O Ministerio do Interior, a través da Dirección Xeral de Protección Civil e Emerxencias, está a...",
        "reference_context_paragraphs": [0,1],
        "retrieved_contexts": [
        {
            "context": "O Ministerio do Interior, a través da Dirección Xeral de Protección Civil e Emerxencias, está a realizar...",
            "context_metadata": {
                "id": 49328,
                "source_id": "codigo_cero01",
                "title": "O envío de alertas móbiles de Protección Civil probarase en Galicia o vindeiro xoves",
                "paragraph_position": 0
            }
        },
        {
            "context": "O sistema ES-Alert foi despregado o pasado 21 de xuño, e é froito da colaboración ...",
            "context_metadata": {
                "id": 49328,
                "source_id": "codigo_cero01",
                "title": "O envío de alertas móbiles de Protección Civil probarase en Galicia o vindeiro xoves",
                "paragraph_position": 2
            }
        },
        {
            "context": "O maquinista que o 24 de xullo de 2013 descarrilou en Angrois provocando a morte de 80 persoas e...",
            "context_metadata": {
                "id": "2070",
                "source_id": "Praza-2022-11-12 09:41:00",
                "title": "Os catro (ou cinco) elementos do sistema de seguridade desconectado nos Alvia que podían evitar o despiste do maquinista",
                "paragraph_position": 0
            }
        }
        ]
    },
    {
        "id": 1,
        "user_input": "Quen está coordinando as probas do sistema ES-Alert?",
    }
  ]
# 🧠 RAG Backend API

API construida con **FastAPI**.  
Proporciona endpoints para configuración de modos, detección de idioma/dominio e inferencia de los demostradores RAG.

---

## 🚀 Requisitos e instalación

Es necesario instalar fastapi y uvicorn:


```bash
pip install fastapi uvicorn
```


## 🧩 Ejecución del servidor

Desde la carpeta raíz del proyecto:

```bash
uvicorn app.main:app --reload --port 8000
```


---

## 📖 Documentación de la API

Una vez en marcha:

- **Swagger UI:** http://127.0.0.1:8000/docs  
- **ReDoc:** http://127.0.0.1:8000/redoc


----

## ⚙️ Endpoints disponibles

### GET /get_config
Obtiene la lista de combinaciones `(language, domain)` soportadas.

**Response model:**
```json
{
  "modes": [
    { "language": "eu", "domain": "legal" },
    { "language": "es", "domain": "general" }
  ]
}
```

---

### POST /configure
Determina el idioma y dominio más apropiado según un texto de entrada.

**Request body:**
```json
{
  "prompt": "Kaixo, zein dira nire eskubideak lan-kontratu batean?",
  "language": null,
  "domain": null
}
```

**Response model:**
```json
{
  "language": "eu",
  "domain": "legal"
}
```

---

### POST /predict
Ejecuta el pipeline de generación aumentada con recuperación (**RAG**).

**Request body:**
```json
{
  "history": [
    { "role": "user", "content": "Explica as funcións do Parlamento de Galicia." }
  ],
  "prompt": "E quen elixe ao presidente?",
  "domain": "politics",
  "language": "gl"
}
```

**Response model:**
```json
{
  "response": "O presidente é elixido polo Parlamento...",
  "contexts": [
    {
      "id": "ctx_001",
      "title": "Constitución de Galicia",
      "passage": "O presidente da Xunta é elixido polo Parlamento entre os seus membros.",
      "timestamp": "2023-02-10T12:30:00Z",
      "url": "https://parlamento.gal/documento/constitucion",
      "metadata": {}
    }
  ],
}
```

---

## 🧱 Estructura del proyecto
```
app/
├── main.py           # Punto de entrada principal
├── models.py         # Modelos Pydantic (requests/responses)
└── routes/
    ├── config.py     # Endpoint GET /get_config
    ├── configure.py  # Endpoint POST /configure
    └── predict.py    # Endpoint POST /predict
```

---

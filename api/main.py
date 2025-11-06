from fastapi import FastAPI
from routes import config, configure, predict

app = FastAPI(
    title="RAG Backend API",
    description="API para configurar y utilizar los demostradores de RAG",
    version="1.0.0",
)

# Registrar routers
app.include_router(config.router)
app.include_router(configure.router)
app.include_router(predict.router)

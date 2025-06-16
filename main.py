from fastapi import FastAPI
from fast_api.routers import iris_endpoints
app = FastAPI()

app.include_router(iris_endpoints.router)

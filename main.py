from fastapi import FastAPI
from routers import iris_endpoints
app = FastAPI()

app.include_router(iris_endpoints.router)

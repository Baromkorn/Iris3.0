from fastapi import FastAPI
from routers import iris_endpoints_image
from routers.iris_endpoints_image import image_router

app = FastAPI(title="Iris Image Upload API")
app.include_router(image_router)
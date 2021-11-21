# #UNCOMMENT BELOW 2 LINES IF RUNNING LOCALLY
# from dotenv import load_dotenv
# load_dotenv(dotenv_path = '.env')

from fastapi import FastAPI
from api.app.routers import prediction_router
from mlcore.predictor import Predictor


app = FastAPI()


@app.get("/")
def start_svc():
    return {"Info": "Service is running"}


app.include_router(
    prediction_router.router,
    tags=['make predictions']
)

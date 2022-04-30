from pydantic import BaseModel
from datetime import datetime, date


class PredictionReqReg(BaseModel):
    id: int
    name:str

    neighbourhood_group: str
    neighbourhood: str
    room_type: str
    minimum_nights: int
    reviews_per_month: float
    calculated_host_listings_count: int
    number_of_reviews:int
    
    last_review: date
    latitude:  float
    longitude: float
    availability_365:int

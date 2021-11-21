from pydantic import BaseModel
from datetime import datetime


class PredictionReq(BaseModel):
    req_id: str
    id: int
    user_id: int
    store_id: int
    device: str
    platform: str
    channel: str
    created_at: str
    num_of_items_req: int

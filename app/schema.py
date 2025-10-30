from pydantic import BaseModel, Field
from typing import List, Tuple, Optional, Dict


class LandmarkPoint(BaseModel):
    x: float
    y: float

    def to_tuple(self):
        return (self.x, self.y)

class SubmitPayload(BaseModel):
    image: str
    landmarks: List[LandmarkPoint]
    segmentation_map: str


class SubmitResponse(BaseModel):
    id: Optional[int] = None
    status: str


class SvgResponse(BaseModel):
    svg: str
    mask_contours: Dict[str, List]
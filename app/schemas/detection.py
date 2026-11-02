from pydantic import BaseModel, ConfigDict, Field


class Detection(BaseModel):
    # "class" is a Python keyword, so the field is class_ with a JSON alias
    model_config = ConfigDict(populate_by_name=True)

    class_id: int = Field(..., description="Numeric class index", examples=[5])
    class_: str = Field(
        ..., alias="class", description="Human-readable class label", examples=["bus"]
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Detection confidence score", examples=[0.91]
    )
    bbox: list[float] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="[x1, y1, x2, y2] in original-image pixels",
        examples=[[22.0, 231.5, 805.0, 756.8]],
    )


class ImageInfo(BaseModel):
    width: int = Field(..., description="Original image width in pixels", examples=[810])
    height: int = Field(
        ..., description="Original image height in pixels", examples=[1080]
    )


class PredictResponse(BaseModel):
    filename: str | None = Field(None, examples=["bus.jpg"])
    image: ImageInfo
    count: int = Field(..., description="Number of detections", examples=[1])
    detections: list[Detection]
    inference_ms: float = Field(
        ..., description="Time spent in the model pipeline only", examples=[123.4]
    )


class ErrorResponse(BaseModel):
    detail: str = Field(..., description="What went wrong and how to fix it")


class NotReadyResponse(ErrorResponse):
    setup_errors: list[str] = Field(
        ..., description="Setup problems to fix. Same list as /health."
    )

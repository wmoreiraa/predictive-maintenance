from pydantic import BaseModel


class PredictPayload(BaseModel):
    type: list[str]  # noqa: A003
    air_temperature_k: list[float]
    process_temperature_k: list[float]
    rotational_speed_rpm: list[float]
    torque_nm: list[float]
    tool_wear_min: list[int]

    class Config:
        schema_extra = {
            "example": {
                "type": ["M"],
                "air_temperature_k": [308.2],
                "process_temperature_k": [400.2],
                "rotational_speed_rpm": [500.3],
                "torque_nm": [20.2],
                "tool_wear_min": [5],
            }
        }

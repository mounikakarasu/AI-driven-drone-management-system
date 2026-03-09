from pydantic import BaseModel

class DroneTelemetry(BaseModel):
    battery_level: float
    altitude: float
    velocity: float
    gps_signal: int
    obstacle_distance: float
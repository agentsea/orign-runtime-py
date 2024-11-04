import os


class Config:
    ORIGN_ADDR = os.getenv("ORIGN_ADDR", "https://orign.agentlabs.xyz")

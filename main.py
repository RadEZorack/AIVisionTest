from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from rtc import router as rtc_router

app = FastAPI()

app.include_router(rtc_router)

# Serve static frontend files
app.mount("/", StaticFiles(directory="templates", html=True), name="static")

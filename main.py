# from config import settings
from fastapi import FastAPI
import uvicorn
import routes


def start_app():
    app = FastAPI(
        title="FIRE DETECTION MODEL",
        description="REAL TIME FIRE DETECTION IN PYTHON - WORKS WITH BOTH IMAGES AND VIDEOS",
        version="0.1.0",
    )

    app.include_router(routes.router)

    return app


app = start_app()


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8080,
        # workers=4,
        reload=True,
        # log_config=LOGGING_CONFIG,
        # log_level="info",
        use_colors=True,
    )

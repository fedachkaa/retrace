from fastapi import FastAPI

import retrace_memory
from retrace_web.api import router


def create_app() -> FastAPI:
    app = FastAPI(title="RETRACE")
    app.state.memory_engine = retrace_memory
    app.include_router(router)
    return app


app = create_app()

"""BatteryGuard FastAPI application factory."""

from __future__ import annotations

from fastapi import FastAPI

from batteryguard.api.routes import cells, core, data, demo, policies
from batteryguard.constants import RESEARCH_ONLY_NOTICE, SOFTWARE_VERSION


def create_app() -> FastAPI:
    application = FastAPI(
        title="BatteryGuard Research API",
        version=SOFTWARE_VERSION,
        description=RESEARCH_ONLY_NOTICE,
    )
    application.include_router(core.router)
    application.include_router(data.router)
    application.include_router(cells.router)
    application.include_router(policies.router)
    application.include_router(demo.router)
    return application


app = create_app()

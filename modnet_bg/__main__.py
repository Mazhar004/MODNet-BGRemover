"""Run the app with a server that does not fork.

    python -m modnet_bg

Why this exists rather than "just use gunicorn": gunicorn forks its workers,
and Apple's Metal stack cannot re-establish its connection to
MTLCompilerService inside a forked process once that service has idled out.
The worker then aborts mid-request with

    MPSLibrary.mm:640: failed assertion `MPSKernel MTLComputePipelineStateCache
    unable to load function ... Unable to reach MTLCompilerService.'

and macOS Obj-C fork safety stops every replacement worker from booting, so
the server spins instead of recovering. Measured on this project: with a
240-second idle before the first MPS request, gunicorn died every time and a
threaded non-forking server completed every time.

Linux has no Metal, so gunicorn stays the right choice in Docker -- see the
Dockerfile. This entry point is for running natively, especially on a Mac
where you want the GPU.
"""

from __future__ import annotations

import logging

from .config import load_settings
from .web import create_app


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    settings = load_settings()
    app = create_app(settings)

    print(f"Background Remover on http://127.0.0.1:{settings.port}")
    # threaded=True gives one thread per request in this process. No fork, so
    # Metal keeps working for the life of the server.
    app.run(
        host="127.0.0.1",
        port=settings.port,
        threaded=True,
        debug=False,
        use_reloader=False,
    )


if __name__ == "__main__":
    main()

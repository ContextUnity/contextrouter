import asyncio

import typer

from contextunity.router.cli.registry import register_command

app = typer.Typer()


@app.callback(invoke_without_command=True)
def serve():
    """Start the ContextRouter gRPC service."""
    from contextunity.router.service.server import serve as grpc_serve

    asyncio.run(grpc_serve())


register_command("serve", app)

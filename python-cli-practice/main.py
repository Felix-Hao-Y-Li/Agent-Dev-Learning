import typer

from commands.hello import hello

app = typer.Typer()
app.command()(hello)

if __name__ == "__main__":
    app()


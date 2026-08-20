import typer

from commands.hello import hello
from commands import todo
from commands.weather import weather

app = typer.Typer()
app.command()(hello)
app.command()(weather)
app.add_typer(todo.app, name="todo")


if __name__ == "__main__":
    app()


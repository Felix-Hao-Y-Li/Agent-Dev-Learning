import typer

from logging_config import setup_logging
from commands.hello import hello
from commands import todo
from commands.weather import weather, weathers
from commands.wordcount import wordcount

setup_logging() # 在创建app之前先把日志配置好

app = typer.Typer()
app.command()(hello)
app.command()(weather)
app.command()(weathers)
app.command()(wordcount)
app.add_typer(todo.app, name="todo")
    

if __name__ == "__main__":
    app()


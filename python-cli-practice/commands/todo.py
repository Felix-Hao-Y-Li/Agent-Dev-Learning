import typer
import json
from pydantic import BaseModel

class TodoItem(BaseModel):
    id: int
    text: str
    done: bool = False

from pathlib import Path

Data_File = Path(__file__).parent.parent / "data" / "todos.json"

def load_todos() -> list[TodoItem]:
    if not Data_File.exists():
        return []
    raw = Data_File.read_text(encoding = "utf-8")
    return [TodoItem.model_validate(item) for item in json.loads(raw)]

def save_todos(todos: list[TodoItem]) -> None:
    Data_File.parent.mkdir(parents=True, exist_ok=True)
    data = [todo.model_dump() for todo in todos]
    Data_File.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

app = typer.Typer()

@app.command()
def add(text: str):
    """添加一条待办事项"""
    todos = load_todos()
    new_id = max((t.id for t in todos), default=0) + 1
    todos.append(TodoItem(id=new_id, text=text))
    save_todos(todos)
    print(f"已添加 #{new_id}: {text}")

@app.command(name="list")
def list_todos():
    """列出所有待办事项"""
    todos = load_todos()
    if not todos:
        print("暂无待办事项")
        return
    for todo in todos:
        status = "✔" if todo.done else "✖"
        print(f"#{todo.id} [{status}] {todo.text}")

@app.command()
def done(id: int):
    """将某条待办标记为已完成"""
    todos = load_todos()
    target = next((t for t in todos if t.id == id), None)
    if target is None:
        print(f"未找到 #{id}")
        raise typer.Exit(code=1)
    target.done = True
    save_todos(todos)
    print(f"已完成 #{id}: {target.text}")

@app.command()
def delete(id:int):
    """删除某条代办事项"""
    todos = load_todos()
    remaining = [t for t in todos if t.id != id]
    if len(remaining) == len(todos):
        print(f"未找到 #{id}")
        raise typer.Exit(code=1)
    save_todos(remaining)
    print(f"已删除 #{id}")
from commands import todo

def test_add_and_list(tmp_path, monkeypatch, capsys):
    # 把 todo 模块的 DATA_FILE 换成一个只属于本次测试的临时路径
    # 测试结束后 monkeypatch 会自动把 DATA_FILE 还原，不影响真实的 data/todos.json
    monkeypatch.setattr(todo, "Data_File", tmp_path / "todos.json")

    todo.add("买菜")       # 实际写入的是临时文件，不是项目真实数据
    todo.list_todos()      # 打印当前所有待办

    captured = capsys.readouterr()
    assert "买菜" in captured.out
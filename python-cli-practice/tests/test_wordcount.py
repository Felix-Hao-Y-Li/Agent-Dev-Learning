from commands.wordcount import wordcount

def test_wordcount_basic(tmp_path,capsys):
    sample_file = tmp_path / "sample.txt"
    sample_file.write_text("apple banana apple orange banana apple", encoding="utf-8")
    wordcount(sample_file, top = 2)
    captured = capsys.readouterr()
    assert "apple: 3" in captured.out
    assert "banana: 2" in captured.out

from commands.hello import hello

def test_hello_default(capsys):
    # capsys 是 pytest 自动注入的参数：只要测试函数写了这个参数名，
    # pytest 就会传入一个能捕获 print 输出的对象
    hello()  # 调用被测函数，不传参数，走默认值 name="world"

    captured = capsys.readouterr()  # 拿到刚才 print() 输出的内容
    assert captured.out == "Hello, world!\n"  # 断言输出内容和预期一致（print 会自动加换行符 \n）

def test_hello_with_name(capsys):
    hello("Alice")

    captured = capsys.readouterr()
    assert captured.out == "Hello, Alice!\n"
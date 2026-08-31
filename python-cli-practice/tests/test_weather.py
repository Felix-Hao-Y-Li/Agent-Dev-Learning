from commands import weather


# 下面两个是"假函数"（fake），专门用来替换掉真实的 geocode / fetch_weather，
# 这样测试时不会真的发起网络请求（不依赖网络、不依赖 Open-Meteo 服务是否可用、跑得也快）。
# 这种"用一个假的实现替换掉真实依赖"的做法在测试里很常见，叫 mock（模拟）。
async def fake_geocode(city: str) -> tuple[float, float]:
    # 无论传入什么城市名，都直接返回一组固定的假坐标
    return (39.9, 116.4)


async def fake_fetch_weather(lat: float, lon: float) -> dict:
    # 直接返回一份固定的假天气数据，格式和真实 Open-Meteo 接口返回的 current_weather 一致
    return {
        "temperature": 20.0,
        "windspeed": 5.0,
        "winddirection": 180,
        "weathercode": 1,
    }


def test_weather_command(monkeypatch, capsys):
    # monkeypatch.setattr(weather 模块, "geocode", 假函数)：
    # 把 weather 模块里的 geocode/fetch_weather 替换成上面的假实现。
    # query_weather() 内部调用 geocode(...)/fetch_weather(...) 时，
    # 是在调用时才去 weather 模块的命名空间里查找这两个名字，
    # 所以替换之后，实际执行的就是假函数，不会有真实网络请求发生。
    monkeypatch.setattr(weather, "geocode", fake_geocode)
    monkeypatch.setattr(weather, "fetch_weather", fake_fetch_weather)

    weather.weather("北京")  # weather() 是同步入口，内部自己调用 asyncio.run，这里直接调用即可

    captured = capsys.readouterr()
    assert "北京" in captured.out
    assert "20.0°C" in captured.out


def test_weathers_command(monkeypatch, capsys):
    # 同样的思路，测试并发查询多个城市的 weathers()
    monkeypatch.setattr(weather, "geocode", fake_geocode)
    monkeypatch.setattr(weather, "fetch_weather", fake_fetch_weather)

    weather.weathers(["北京", "上海"])

    captured = capsys.readouterr()
    # 两个城市的结果都应该出现在打印内容里
    assert "北京" in captured.out
    assert "上海" in captured.out
    assert captured.out.count("20.0°C") == 2  # 两个城市用的都是同一份假数据，应该各打印一次

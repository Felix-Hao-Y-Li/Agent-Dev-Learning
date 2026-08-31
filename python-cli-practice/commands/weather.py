import asyncio
import httpx
import logging

logger = logging.getLogger(__name__)
async def geocode(city: str) -> tuple[float, float]:
    async with httpx.AsyncClient() as client:
        response = await client.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1, "language": "zh"},
        )
        response.raise_for_status()
        data = response.json()
        result = data["results"][0]
        return result["latitude"],result["longitude"]

async def fetch_weather(lat: float, lon: float) -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.get(
            "https://api.open-meteo.com/v1/forecast",
            params={"latitude": lat, "longitude": lon, "current_weather": True},
        )
        response.raise_for_status()
        return response.json()["current_weather"]

async def query_weather(city:str):
    lat, lon = await geocode(city)
    weather_data = await fetch_weather(lat, lon)
    return weather_data

def weather(city: str):
    """Get weather for a given city."""
    current = asyncio.run(query_weather(city))
    print(f"{city}当前气温：{current['temperature']}°C，风速：{current['windspeed']} km/h，风向：{current['winddirection']}°，天气代码：{current['weathercode']}")

async def query_many_weathers(cities: list[str]) -> list[dict]:
    # 用生成器表达式给每个城市生成一个 query_weather 协程，
    # 前面的 * 是"解包"：把这一串协程当成多个独立参数传给 gather，
    # 而不是当成一个"列表"整体传进去（gather 要的是多个协程参数，不是一个列表参数）t
    results = await asyncio.gather(*(query_weather(city) for city in cities))
    #gather 会等所有协程都跑完，才把结果按传入顺序打包成一个列表返回
    return results

def weathers(cities: list[str]):
    """Get weather for a list of cities."""
    # Typer 看到参数类型是 list[str]，会把命令行里跟在后面的多个词都收进这个列表
    # 例如 `main.py weathers 北京 上海` -> cities = ["北京", "上海"]
    all_results = asyncio.run(query_many_weathers(cities))
    # zip 把"城市名"和"对应的天气结果"一一配对，方便一起遍历打印
    for city, current in zip(cities, all_results):
        print(f"{city}当前气温：{current['temperature']}°C，风速：{current['windspeed']} km/h，风向：{current['winddirection']}°，天气代码：{current['weathercode']}")
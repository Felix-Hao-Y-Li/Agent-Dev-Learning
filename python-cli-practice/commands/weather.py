import asyncio
import typer
import httpx

async def getcode(city: str) -> tuple[float, float]:
    async with httpx.AsyncClient() as client:
        response = await client.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1, "language": "zh"},
        )
        response.raise_for_status()
        data = response.json()
        result = data["results"][0]
        return result["latitude"],result["longitude"]

def weather(city: str):
    """Get weather for a given city."""
    lat, lon = asyncio.run(getcode(city))
    print(f"The latitude and longitude of {city} are: {lat}, {lon}")

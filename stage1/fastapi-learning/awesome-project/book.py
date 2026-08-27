from fastapi import APIRouter

# 创建一个独立的子路由对象
book_router = APIRouter()

@book_router.get("/")
async def get_books():
    return {"message": "List of books"}

@book_router.get("/{book_id}")
async def get_book(book_id: int):
    return {"book_id": book_id, "message": "Details of the book"}
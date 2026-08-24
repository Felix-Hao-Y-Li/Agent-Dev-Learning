from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def read_root():
    return {"Message": "Hello World！"}
#路径参数，路径参数item_id的值会作为参数item_id传递给你的函数
@app.get("/items/{item_id}")
async def read_item(item_id:int):
    """注意，函数接收并返回的值是 3（ int），不是 "3"（str）。
FastAPI 通过类型声明自动进行请求的“解析”。"""
    return {"item_id": item_id}

# 路径顺序很重要，由于路径操作是按顺序依次运行的，因此一定要在/users/{user_id}之前定义/users/me，否则会被/user_id捕获，导致无法访问/users/me。
@app.get("/users/me")
async def read_user_me():
    return {"user_id": "the current user"}
# 同样不能重复定义一个路径操作，由于路径首先匹配，始终会使用第一个定义的

@app.get("/users/{user_id}")
async def read_user(user_id: str):
    return {"user_id": user_id}

# 预设值
from enum import Enum

class ModelName(str, Enum):
    alexnet = "alexnet"
    resnet = "resnet"
    lenet = "lenet"

@app.get("/models/{model_name}")
async def get_model(model_name: ModelName):
    if model_name == ModelName.alexnet:
        return {"model_name": model_name, "message": "Deep Learning FTW!"}
    #即使嵌套在JSON请求体中，也可以从你的路径操作返回枚举成员
    if model_name.value == "lenet":
        return {"model_name": model_name, "message": "LeCNN all the images"}

    return {"model_name": model_name, "message": "Have some residuals"}

# 包含路径的路径参数，注意，包含 /home/johndoe/myfile.txt 的路径参数要以斜杠（/）开头。本例中的 URL 是 /files//home/johndoe/myfile.txt。注意，files 和 home 之间要使用双斜杠（//）。
@app.get("/files/{file_path:path}")
async def read_file(file_path: str):
    return {"file_path": file_path}

# 查询参数
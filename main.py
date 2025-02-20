from typing import Union
from fastapi import FastAPI

import model_and
import model_or

model_and = model_and.AndModel()
model_or = model_or.OrModule()

#클래스 생성자. app변수에 들어가게 됨.
app = FastAPI()

#app 객체를 실제로 실행하지 않음. 이 코드를 실행하는 방식이 특이함. 이렇게 짜는 것은 fastAPI 명령에 이해할 수 있는 코드를 짠 것임.
#main함수가 없어도 실행이 됨. 
#fastapi dev main.py -> dev는 개발모드로 화면을 띄우는 것임. 
@app.get("/")
def read_root():
    return {"Hello": "World"}
            
@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id" : item_id, "q": q}

#아래 두개는 model_and와 관련된 코드.
@app.get("/predict/left/{left}/right/{right}")
def predict(left: int, right : int):
    result1 = model_and.predict([left,right])
    result2 = model_or.predict([left,right])
    return {"And" : result1, "OR" : result2}

@app.post("/train")
def train():
    model_and.train()
    model_or.train()
    return {"result" : "OK"}

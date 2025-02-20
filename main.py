from typing import Union
from fastapi import FastAPI

import model

model = model.AndModel()


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

@app.get("/predict/left/{left}/right/{right}")
def predict(left: int, right : int):
    model.train()
    result = model.predict([left,right])
    return {"result" : result}

@app.get("/train")
def train():
    model.train()
    return {"result" : "OK"}
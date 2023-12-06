from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import numpy as np
import pickle

app = FastAPI()


model = pickle.load(open("model.pickle", 'rb'))  # получение модели из pickle файла
scaler = pickle.load(open("scaler.pickle", 'rb'))


class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str 
    engine: str
    max_power: str
    torque: str
    seats: float


class Items(BaseModel):
    objects: List[Item]


str2float = lambda x: 0.0 if str(x).split(" ")[0] == "" else float(str(x).split(" ")[0])

def mileage_to_float(x: str) -> List[float]:
    # если подается в таком же формате, как и в датасете
    # 0.0 if str(x).split(" ")[0] == "" else float(str(x).split(" ")[0])
    return [str2float(x)]


def engine_to_float(x: str) -> List[float]:
    # если подается в таком же формате, как и в датасете
    # 0.0 if str(x).split(" ")[0] == "" else float(str(x).split(" ")[0])
    return [str2float(x)]


def max_power_to_float(x: str) -> List[float]:
    # если подается в таком же формате, как и в датасете
    # 0.0 if str(x).split(" ")[0] == "" else float(str(x).split(" ")[0])
    return [str2float(x)]


def fuel_to_float(x: str) -> List[float]:
    dict_val = {
        "Diesel": 0,
        "LPG": 1,
        "Petrol": 2
    }
    res_arr = np.zeros(len(dict_val))
    try:
        idx = res_arr[x]
        res_arr[idx] = 1
        return res_arr.tolist()
    except Exception:
        return res_arr.tolist()


def seller_type_to_float(x: str) -> List[float]:
    dict_val = {
        "Individual": 0,
        "Trustmark Dealer": 1
    }
    res_arr = np.zeros(len(dict_val))
    try:
        idx = res_arr[x]
        res_arr[idx] = 1
        return res_arr.tolist()
    except Exception:
        return res_arr.tolist()


def transmission_to_float(x: str) -> List[float]:
    dict_val = {
        "Manual": 0
    }
    res_arr = np.zeros(len(dict_val))
    try:
        idx = res_arr[x]
        res_arr[idx] = 1
        return res_arr.tolist()
    except Exception:
        return res_arr.tolist()


def owner_to_float(x: str) -> List[float]:
    dict_val = {
        "Fourth & Above Owner": 0,
        "Second Owner": 1,
        "Test Drive Car": 2,
        "Third Owner": 3
    }
    res_arr = np.zeros(len(dict_val))
    try:
        idx = res_arr[x]
        res_arr[idx] = 1
        return res_arr.tolist()
    except Exception:
        return res_arr.tolist()


def seats_to_float(x: str) -> List[float]:
    x = str(int(x))
    dict_val = {
        "14": 0,
        "2": 1,
        "4": 2,
        "5": 3,
        "6": 4,
        "7": 5,
        "8": 6,
        "9": 7,
    }
    res_arr = np.zeros(len(dict_val))
    try:
        idx = res_arr[x]
        res_arr[idx] = 1
        return res_arr.tolist()
    except Exception:
        return res_arr.tolist()
    

premium_marks = ["mercedes-benz", "bmw", "audi", "volvo", "jaguar", "land", "lexus"]
preproc_name = lambda x: 1 if (x.split(" ")[0]).lower() in premium_marks else 0


def proccess_input(item: Item) -> list:
    features_lst = []
    
    features_lst.append(item.year ** 2)
    features_lst.append(item.km_driven)
    features_lst += mileage_to_float(item.mileage)
    features_lst += engine_to_float(item.engine)
    features_lst += max_power_to_float(item.max_power)
    features_lst.append(preproc_name(item.name))
    features_lst += fuel_to_float(item.fuel)
    features_lst += seller_type_to_float(item.seller_type)
    features_lst += transmission_to_float(item.transmission)
    features_lst += owner_to_float(item.owner)
    features_lst += seats_to_float(item.seats)

    cat_lst = features_lst[5:]
    num_lst = features_lst[:5]

    num_lst = scaler.transform(np.array([num_lst]))[0, :].tolist()

    return num_lst + cat_lst


@app.post("/predict_item")
def predict_item(item: Item) -> float:
    # надо полученные значения предобработать определенным образом
    # при чем у категориальных и числовых фичей необходимо сохранить порядок, который был
    # при обучении

    curr_input = np.array([proccess_input(item)])
    pred = model.predict(curr_input)[0]

    return np.exp(pred) - 1


@app.post("/predict_items")
def predict_items(items: List[Item]) -> List[float]:
    print()
    print(111111)
    print()
    inputs_lst = np.array([proccess_input(item) for item in items])

    return [item[0] for item in (np.exp(model.predict(inputs_lst)) - 1).tolist()]
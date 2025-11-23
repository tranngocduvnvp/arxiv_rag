# file: main.py
import time
from fastapi import FastAPI
import numpy as np

app = FastAPI()



def nhan_ma_tran(A, B):
    # Ki?m tra di?u ki?n nhân ma tr?n: s? c?t c?a A = s? hàng c?a B
    if len(A[0]) != len(B):
        raise ValueError("S? c?t c?a ma tr?n A ph?i b?ng s? hàng c?a ma tr?n B.")

    # Kh?i t?o ma tr?n k?t qu? v?i các ph?n t? b?ng 0
    ket_qua = [[0 for _ in range(len(B[0]))] for _ in range(len(A))]

    # Th?c hi?n phép nhân
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                ket_qua[i][j] += A[i][k] * B[k][j]
    
    return ket_qua


@app.get("/")
def read_root():
    return {"message": "Xin chào FastAPI!"}

@app.get("/hello/{name}/{times}")
async def read_item(name: str, times: int):
    # Delay 3 giây
    start_time = time.time()
    # time.sleep(int(times))
    A = np.random.rand(500, 10)
    B = np.random.rand(10, 10000)
    nhan_ma_tran(A, B)
    
    end_time = time.time()
    duration = end_time - start_time
    return {
        "message": f"Xin chào {name} delay {times} s!",
        "execution_time_seconds": round(duration, 3)    
    }

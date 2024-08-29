#선형 분석 - 주택 가격 맞추기
import numpy as np
#데이터 패치
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()
print(housing.keys())
print("문제:",housing.data.shape,"정답:",housing.target.shape)
print("열의 이름:",housing.feature_names)
#데이터 구조
print(housing.data[0])

import pandas as pd
df = pd.DataFrame(housing.data,columns=housing.feature_names)
print(df.info())
print(df.describe())

# 결측치 입력
df.loc[10,"HouseAge"]=None
df.loc[50,"HouseAge"]=None
# 결측치 확인
print("결측값:",df["HouseAge"].isna().sum())
print("정상값:",df["HouseAge"].notnull().sum())
# 결측값 보정 ( 평균값으로 대체. )
df["HouseAge"] = df["HouseAge"].fillna(df["HouseAge"].mean())
print("결측값:",df["HouseAge"].isna().sum())

cpdf = df.loc[:,:]  #테스트를 위해 사본 데이터를 저장
print(cpdf)

# 훈련데이터 분리
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(df,housing.target,random_state=11)
print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)
# 정규분포화가 되지 않는 상태 훈련 확인
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
model = Sequential()
model.add(Dense(32,activation="relu",input_shape=(8,)))
model.add(Dense(16,activation="relu"))
model.add(Dense(8,activation="relu"))
model.add(Dense(1))
model.compile(loss="mse",optimizer="Adam",metrics=["mse"])
hist = model.fit(x_train,y_train,epochs=50,batch_size=10)

print(hist.history.keys())
import matplotlib.pyplot as plt
plt.plot(hist.history["loss"])
plt.legend(["loss"])
plt.figure(figsize=(5,5))
plt.show()

# 평가
y_pred = model.predict(x_test)
plt.scatter(y_pred,y_test)
plt.plot(y_test,y_test,"r")
plt.show()

# 데이터 표준정규화로 훈련
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
cpdf = scaler.fit_transform(cpdf)
dfpd = pd.DataFrame(cpdf,columns=housing.feature_names)
print(dfpd.head(5))

# 훈련데이터 분할
xx_train, xx_test, yy_train, yy_test = train_test_split(cpdf,housing.target,random_state=11)
print(xx_train.shape,xx_test.shape,yy_train.shape,yy_test.shape)

model_1 = Sequential()
model_1.add(Dense(32,activation="relu",input_shape=(8,)))
model_1.add(Dense(16,activation="relu"))
model_1.add(Dense(8,activation="relu"))
model_1.add(Dense(1,activation="linear"))
model_1.compile(loss="mse",optimizer="Adam",metrics=["acc"])
hist_1 = model_1.fit(xx_train,yy_train,epochs=50,batch_size=10)

print(hist_1.history.keys())
plt.plot(hist_1.history["loss"])
plt.plot(hist_1.history["acc"])
plt.legend(["loss","acc"])
plt.figure(figsize=(5,5))
plt.show()

# 평가
yy_pred = model_1.predict(xx_test)
print(housing.target[120])
print(yy_test[120])
print(yy_pred[120])
plt.scatter(yy_pred,yy_test)
plt.plot(yy_test,yy_test,"r")
plt.show()

eval = model_1.evaluate(xx_test,yy_test) ; # 선형 회귀라서 정확도 손실 평가는 의미 없음. 그래프와 데이터의 방향이 일치하는지만 보면 된다.
print(eval)

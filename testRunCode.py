import numpy as np
import pandas as pd
from model import model
from sklearn.preprocessing import PolynomialFeatures



answer = 0
input_array = np.array([[5,5,5,5,4,5,3,5,4]])
group1_input = pd.DataFrame(input_array, columns=['1_1', '1_2', '1_3', '1_4', '1_5', '1_6', '1_7', '1_8', '1_9'])

group1_input_transformed = PolynomialFeatures(degree=3, include_bias=False).fit_transform(group1_input)

group1_predict = model.predict(group1_input_transformed)

for i in group1_predict:
    answer = round(i)

print(answer)

import config
import data_management
import preprocessors
import valid
from sklearn.linear_model import LinearRegression

# Reading the DataFrame 
Dataset = data_management.load_dataset(config.DataFramePath)

# Seperating the data into features and labels
X = preprocessors.input_column(Dataset, config.StartColumn, config.EndColumn)
Y = preprocessors.predicting_column(Dataset, config.PredictingColumn)

# Generating polynomial features 
Z = preprocessors.make_polynomial_features(X)
# Dividing the dataset into test and train data
X_train, X_test, Y_train, Y_test = preprocessors.split_dataset(Z, Y)

# Selecting the linear regression method from the scikit-learn library
model = LinearRegression().fit(X_train, Y_train)

# Evaluating the trained model on training data
#valid.validate(model, X_train, Y_train)

# Evaluating our trained model on test data
#valid.validate(model, X_test, Y_test)
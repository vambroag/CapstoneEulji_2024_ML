from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

def input_column(_data, StartColumn, EndColumn):
    _input_column = _data[_data.columns[StartColumn:EndColumn]]
    return _input_column

def predicting_column(_data, PredictingColumn):
    _predicting_column = _data[PredictingColumn]
    return _predicting_column

def make_polynomial_features(_input_column):
    _input_column_polynomial_features = PolynomialFeatures(degree=3, include_bias=False).fit_transform(_input_column)
    return _input_column_polynomial_features

def split_dataset(_input_column_polynomial_features, _predicting_column):
    X_train, X_test, Y_train, Y_test = train_test_split(_input_column_polynomial_features, _predicting_column, test_size=0.3, random_state=10)
    return X_train, X_test, Y_train, Y_test
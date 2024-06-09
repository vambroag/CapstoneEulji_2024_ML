from sklearn import metrics

def validate(model, Input, Label):
    prediction = model.predict(Input)
    predict = []
    for i in prediction:
        predict.append(round(i))
    print("MAE on train data= " , metrics.mean_absolute_error(Label, predict))
    print("R2 score on train data= " , metrics.r2_score(Label, predict))
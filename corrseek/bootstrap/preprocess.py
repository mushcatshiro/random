from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler


def scale(X, preprocess):
    if preprocess == 1:
        scaler = MinMaxScaler()
    elif preprocess == 2:
        scaler = MaxAbsScaler
    else:
        raise Exception("method not found")
    return scaler.fit_transform(X)
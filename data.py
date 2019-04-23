import numpy as np

def prepare_data(data, set, zero_center=False, flatten=False):
    data_set = data[set]
    data_set = data_set.reshape(data_set.shape[0], data_set.shape[1], data_set.shape[2], 1)
    data_set = data_set.astype("float32")
    if zero_center:
        data_set = (data_set - 127.5) / 127.5
    else:
        data_set /= 255
    if flatten:
        data_set = data_set.reshape(data_set.shape[0], data_set.shape[1] * data_set.shape[2])
    return data_set

def get_data(zero_center=False, path="/home/wzwietering/Data/Data/MNIST/mnist.npz"):
    data = np.load(path)
    testX = prepare_data(data, "x_test")
    testY = data["y_test"]
    trainX = prepare_data(data, "x_train")
    trainY = data["y_train"]
    return trainX, trainY, testX, testY

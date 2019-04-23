from keras.datasets import mnist

def prepare_data(data_set, zero_center=False, flatten=False):
    data_set = data_set.reshape(data_set.shape[0], data_set.shape[1], data_set.shape[2], 1)
    data_set = data_set.astype("float32")
    if zero_center:
        data_set = (data_set - 127.5) / 127.5
    else:
        data_set /= 255
    if flatten:
        data_set = data_set.reshape(data_set.shape[0], data_set.shape[1] * data_set.shape[2])
    return data_set

def get_data(zero_center=False, flatten=False):
    (trainX, trainY), (testX, testY) = mnist.load_data()
    testX = prepare_data(testX, zero_center, flatten)
    trainX = prepare_data(trainX, zero_center, flatten)
    return trainX, trainY, testX, testY

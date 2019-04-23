import tensorflow as tf

from keras.backend.tensorflow_backend import set_session
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, recall_score, precision_score

import data

def make_model(input_shape=(28,28,1)):
    model = Sequential()
    model.add(Conv2D(28, kernel_size=(3,3), input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())

    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.2))

    model.add(Dense(10, activation="softmax"))
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    return model

if __name__ == "__main__":
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    set_session(sess)

    model = make_model()
    trainX, trainY, testX, testY = data.get_data()
    model.fit(trainX, trainY, epochs=10)
    val_predict = model.predict(testX)

    confusion = confusion_matrix(testY, val_predict.argmax(axis=1))
    f1 = f1_score(testY,
                  val_predict.argmax(axis=1),
                  average="macro")
    precision = precision_score(testY,
                  val_predict.argmax(axis=1),
                  average="macro")
    recall = recall_score(testY,
                  val_predict.argmax(axis=1),
                  average="macro")
    accuracy = accuracy_score(testY, val_predict.argmax(axis=1))

    print(confusion)
    print(f"F1 score: {f1}")
    print(f"Precision score: {precision}")
    print(f"Recall score: {recall}")
    print(f"Accuracy: {accuracy}")
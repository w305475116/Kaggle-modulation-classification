import csv
import time
import h5py
from keras.callbacks import Callback, ModelCheckpoint
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
import numpy as np
import pandas as pd
from keras.wrappers.scikit_learn import KerasRegressor

import CONSTANTS
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostRegressor

f = h5py.File("./dataset/data.hdf5", 'r')

x_train = np.array(f['train'])
x_test =  np.array(f['test'])

y_train = []
train_labels = pd.read_csv("./dataset/train_labels.csv")
for line in train_labels.values:
    labelName = line[1]
    y_train.append(CONSTANTS.DICT[labelName])
# reverse the original sequence to make a new
x_train = np.append(x_train, np.flip(x_train), axis = 0)
y_train = np.append(y_train, np.flip(y_train))
y_train = np.array(y_train)


def simple_model():
    model = Sequential()
    model.add(Conv1D(64, 3, activation='relu', input_shape=(1024, 2)))
    model.add(MaxPooling1D(3))
    model.add(Conv1D(64, 3, activation='relu'))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.5))
    # model.add(Dense(2048, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    # keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9)
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    return model


# model = load_model("models/Tue_May__5_14:50:42_2020")
startT = time.asctime().replace(" ","_")
# checkpointer = ModelCheckpoint(filepath='./models/startAtBest' + startT + '.hdf5', verbose=1, save_best_only=True)
# history = model.fit(x_train, y_train, validation_split=0.25, batch_size=16, epochs=30, callbacks=[checkpointer])
# model = load_model('./models/startAtBest' + startT + '.hdf5')
ann_estimator = KerasRegressor(build_fn= simple_model, epochs=20, batch_size=16, verbose=1,
                               validation_split=0.25)
boosted_ann = AdaBoostRegressor(base_estimator= ann_estimator)
boosted_ann.fit(x_train, y_train)
y_test = boosted_ann.predict(x_test)


with open('./predicts/'+startT+'.csv', 'w') as csvfile:
    filewriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    filewriter.writerow(['Id', 'Category'])
    for i in range(len(y_test)):
        filewriter.writerow([i, CONSTANTS.DICT_ITER[y_test[i]]])

# # 绘制训练 & 验证的准确率值
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('Model accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='upper left')
# plt.savefig('./plots/'+startT+'acc.png')
# plt.show()
#
#
# # 绘制训练 & 验证的损失值
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('Model loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='upper left')
# plt.savefig('./plots/'+startT+'err.png')
# plt.show()

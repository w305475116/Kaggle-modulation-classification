import csv
import time
import h5py
from keras.models import Sequential, load_model
import numpy as np
import CONSTANTS


model_name = './models/startAtBestThu_May__7_23:13:38_2020.hdf5'


startT = time.asctime().replace(" ","_")
f = h5py.File("./dataset/data.hdf5", 'r')
x_test =  np.array(f['test'])
model = load_model(model_name)
y_test = model.predict_classes(x_test)

with open('./predicts/'+startT+'.csv', 'w') as csvfile:
    filewriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    filewriter.writerow(['Id', 'Category'])
    for i in range(len(y_test)):
        filewriter.writerow([i, CONSTANTS.DICT_ITER[y_test[i]]])
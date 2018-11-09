# 第二种,可以节省内存
'''
Created on 2018-4-11

fit_generate.txt，后面两列为lable,已经one-hot编码
1 2 0 1
2 3 1 0
1 3 0 1
1 4 0 1
2 4 1 0
2 5 1 0
'''
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

count =1


def generate_arrays_from_file(path):
    global count
    while 1:
        datas = np.loadtxt(path,delimiter=' ',dtype="int")
        print('读取数据大小', datas.shape)
        x = datas[:,:2]
        y = datas[:,2:]
        print("count:"+str(count))
        count = count+1
        yield (x,y)


x_valid = np.array([[1,2],[2,3]])
y_valid = np.array([[0,1],[1,0]])
model = Sequential()
model.add(Dense(units=1000, activation='relu', input_dim=2))
model.add(Dense(units=2, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

model.fit_generator(generate_arrays_from_file("fit_generate.txt"),steps_per_epoch=10,
                    epochs=2,max_queue_size=1,validation_data=(x_valid, y_valid),workers=1)

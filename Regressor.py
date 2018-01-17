<<<<<<< HEAD
# _*_ coding: utf-8 _*_
=======
>>>>>>> 661603744b9888e67065144582f1c4eb53a563ff
# Regressor example
# https://github.com/keloli/KerasPractise/edit/master/Regressor.py

import numpy as np
np.random.seed(1337)    # for reproducibility
                        # 当我们设置相同的seed，每次生成的随机数相同。如果不设置seed，则每次会生成不同的随机数
from keras.models import Sequential # 一个层一个层增加的线性模型
from keras.layers import Dense
import matplotlib.pyplot as plt

# create some data
X = np.linspace(-1, 1, 200)
# print (X)
np.random.shuffle(X)    # randomize the data
<<<<<<< HEAD
print (X)
=======
>>>>>>> 661603744b9888e67065144582f1c4eb53a563ff
Y = 0.5 * X + 2 + np.random.normal(0, 0.05, (200, ))
# plot data
plt.scatter(X, Y)
plt.show()

X_train, Y_train = X[:160], Y[:160]     # first 160 data points
X_test, Y_test = X[160:], Y[160:]       # last 40 data points

# build a neural network from the 1st layer to the last layer
model = Sequential()

model.add(Dense(output_dim=1, input_dim=1)) # Dense是全连接层，第一层需要定义输入，
				       	    # 第二层只需要定义输出，第二层一第一层的输出作为输入

# choose loss function and optimizing method
model.compile(loss='mse', optimizer='sgd')

# training
print('Training -----------')
for step in range(301):
    cost = model.train_on_batch(X_train, Y_train)
    if step % 100 == 0:
        print('train cost: ', cost)

# test
print('\nTesting ------------')
cost = model.evaluate(X_test, Y_test, batch_size=40)
print('test cost:', cost)
W, b = model.layers[0].get_weights()	# 查看训练出的网络参数
					# 由于我们网络只有一层，且每次训练的输入只有一个，输出只有一个
					# 因此第一层训练出Y=WX+B这个模型，其中W,b为训练出的参数
print('Weights=', W, '\nbiases=', b)

# plotting the prediction
Y_pred = model.predict(X_test)
plt.scatter(X_test, Y_test)
plt.plot(X_test, Y_pred)
plt.show()

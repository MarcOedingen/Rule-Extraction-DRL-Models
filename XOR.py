import numpy as np
import tensorflow as tf
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import np_utils
from ruleex import deepred
from ruleex.deepred.model import DeepRedFCNet
from tensorflow import keras

import DR_helper_functions as gf


def xor_rule(A, B):
    if A >= 0.5:
        if B >= 0.5:
            return 0
        else:
            return 1
    else:
        if B < 0.5:
            return 0
        else:
            return 1


x_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train = np.array([0, 1, 1, 0])
y_train = np_utils.to_categorical(y_train, 2)

myANN = Sequential()
myANN.add(Dense(16, input_dim=2, kernel_initializer='normal', activation='sigmoid', use_bias=True))
myANN.add(Dense(8, kernel_initializer='normal', activation='sigmoid', use_bias=True))
myANN.add(Dense(2, kernel_initializer='normal', activation='sigmoid', use_bias=True))
myANN.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=0.01),
              metrics=['accuracy'])
history = myANN.fit(x_train, y_train, batch_size=16, epochs=500)

accuracy = myANN.evaluate(x_train, y_train, verbose=1)[1]
print("Training accuracy: " + str(accuracy * 100) + "%")

myANN.save('XOR_NN')
model_ANN = keras.models.load_model('XOR_NN')

x_train = []
y_train = []
training_size = np.power(10, 5)
for i in range(training_size):
    x_train.append([np.random.randint(2), np.random.randint(2)])
    y_train.append(xor_rule(x_train[i][0], x_train[i][1]))

weights = model_ANN.get_weights()
weights = gf.reshape_weights(weights)
layer_sizes = gf.get_layer_sizes(weights)
deepred_net = DeepRedFCNet(layer_sizes)
deepred_net.init_eval_weights(weights=[weights[0], weights[1]])
tf.compat.v1.disable_eager_execution()
l_activation = deepred_net.eval_layers(np.array(x_train).reshape(training_size, 2))

dr_params = dict()
dr_params[deepred.VARBOSE] = 1

rt = deepred.deepred(l_activation[:-1], dr_params)
rt.view_graph()
gf.Ruletree_to_string(rule=rt.root, depth=0)

y_DeepRED = rt.eval_all(np.array(x_train))
correct = 0
for i in range(len(y_DeepRED)):
    if y_DeepRED[i] == y_train[i]:
        correct += 1
print(f"DeepRED Accuracy: {(correct / len(y_DeepRED)) * 100}%")

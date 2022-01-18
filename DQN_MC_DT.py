import gym
import numpy as np
import pydotplus
import tensorflow as tf
from joblib import dump, load
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import np_utils
from ruleex import deepred
from ruleex.deepred.model import DeepRedFCNet
from sklearn import tree
from stable_baselines3 import DQN
from tensorflow import keras

import DR_helper_functions as gf

CREATE_MODEL = True
CREATE_TREE = True
CREATE_DNN = True
TREE_DEPTH = 3
LOG_NAME = "DQN_MC_TENSORBOARD"
ENVIRONMENT = gym.make("MountainCar-v0")
tf.compat.v1.disable_eager_execution()
EPISODES = 100

# ------------------------------------------------------------------------------------------------------------------
# DEEP Q-NETWORK ORACLE
# ------------------------------------------------------------------------------------------------------------------
policy_kwargs = dict(net_arch=[256, 256])
DQN_MC_MODEL = DQN("MlpPolicy", ENVIRONMENT, policy_kwargs=policy_kwargs, learning_rate=4e-3, batch_size=128,
                   buffer_size=20000, learning_starts=3000, target_update_interval=600, train_freq=16,
                   gradient_steps=8, gamma=0.98, exploration_fraction=0.2, exploration_final_eps=0.0, verbose=0,
                   tensorboard_log=f"./{LOG_NAME}/")

if CREATE_MODEL:
    DQN_MC_MODEL.learn(total_timesteps=int(1.2e5))
    DQN_MC_MODEL.save("DQN_MC")
else:
    DQN_MC_MODEL = DQN.load("DQN_MC")

DQN_MC_REWARDS = []
DQN_MC_OBSERVATIONS = []
DQN_MC_ACTIONS = []
DQN_MC_REWARD_EPISODE = 0
CURRENT_EPISODE = 1

obs = ENVIRONMENT.reset()
while True:
    action, states = DQN_MC_MODEL.predict(observation=obs, deterministic=True)
    DQN_MC_OBSERVATIONS.append(obs)
    DQN_MC_ACTIONS.append(action)
    obs, reward, done, info = ENVIRONMENT.step(action)
    DQN_MC_REWARD_EPISODE += reward
    if done:
        obs = ENVIRONMENT.reset()
        DQN_MC_REWARDS.append(DQN_MC_REWARD_EPISODE)
        CURRENT_EPISODE += 1
        DQN_MC_REWARD_EPISODE = 0
        if CURRENT_EPISODE > EPISODES:
            break
print(f"Average Reward Oracle: {np.round(np.mean(DQN_MC_REWARDS), 2)} +/- {np.round(np.std(DQN_MC_REWARDS), 2)}")

# ------------------------------------------------------------------------------------------------------------------
# Classification Tree
# ------------------------------------------------------------------------------------------------------------------
INPUT_TREE_X = DQN_MC_OBSERVATIONS
INPUT_TREE_Y = DQN_MC_ACTIONS

if CREATE_TREE:
    CLASSIFICATION_TREE = tree.DecisionTreeClassifier(max_depth=TREE_DEPTH)
    CLASSIFICATION_TREE.fit(X=INPUT_TREE_X, y=INPUT_TREE_Y)
    dump(CLASSIFICATION_TREE, "DQN_MC_CLF")
else:
    CLASSIFICATION_TREE = load("DQN_MC_CLF")

TREE_REWARDS = []
TREE_REWARD_EPISODE = 0
CURRENT_EPISODE = 1

obs = ENVIRONMENT.reset()
while True:
    action = CLASSIFICATION_TREE.predict(obs.reshape(1, -1))
    obs, reward, done, info = ENVIRONMENT.step(action[0])
    TREE_REWARD_EPISODE += reward
    if done:
        obs = ENVIRONMENT.reset()
        TREE_REWARDS.append(TREE_REWARD_EPISODE)
        CURRENT_EPISODE += 1
        TREE_REWARD_EPISODE = 0
        if CURRENT_EPISODE > EPISODES:
            break
print(f"Average Reward Tree with depth {TREE_DEPTH}: {np.round(np.mean(TREE_REWARDS), 2)} +/- "
      f"{np.round(np.std(TREE_REWARDS), 2)}")

dot_data = tree.export_graphviz(CLASSIFICATION_TREE, out_file=None,
                                feature_names=["Auto Position", "Auto Geschwindigkeit"],
                                class_names=["Beschleunige Links", "Beschleunige Nicht", "Beschleunige Rechts"],
                                filled=True)

graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_png(f'DQN_MC_DT_{TREE_DEPTH}.png')

DT_text = tree.export_text(CLASSIFICATION_TREE, feature_names=["obs[0]", "obs[1]"])
print(DT_text.replace('--- class:', 'return').replace('--- obs', 'if obs').replace('--- value: ', 'return ').
      replace('\n', ':\n').replace('|', '').replace('return 0:', 'return 0').replace('return 1:', 'return 1').
      replace('return 2:', 'return 2').replace('\t', '').replace(']:', ']'))

# ------------------------------------------------------------------------------------------------------------------
# Deep Neural Network / DeepRED
# ------------------------------------------------------------------------------------------------------------------

x_train = np.array(DQN_MC_OBSERVATIONS)
y_train = np.array(DQN_MC_ACTIONS)
y_train = np_utils.to_categorical(y_train, 3)

DQN_MC_DNN = Sequential()
DQN_MC_DNN.add(Dense(256, input_dim=2, kernel_initializer='normal', activation='sigmoid', use_bias=True))
DQN_MC_DNN.add(Dense(256, kernel_initializer='normal', activation='sigmoid', use_bias=True))
DQN_MC_DNN.add(Dense(3, kernel_initializer='normal', activation='sigmoid', use_bias=True))
DQN_MC_DNN.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=0.01),
                   metrics=['accuracy'])
if CREATE_DNN:
    history = DQN_MC_DNN.fit(x_train, y_train, batch_size=256, epochs=500)
    accuracy = DQN_MC_DNN.evaluate(x_train, y_train, verbose=1)[1]
    print("Training accuracy: " + str(accuracy * 100) + "%")
    DQN_MC_DNN.save('DQN_MC_DNN')
else:
    DQN_MC_DNN = keras.models.load_model('DQN_MC_DNN')

DNN_REWARDS = []
DNN_REWARD_EPISODE = 0
CURRENT_EPISODE = 1

obs = ENVIRONMENT.reset()
while True:
    action = DQN_MC_DNN.predict(obs.reshape(1, -1))
    obs, reward, done, info = ENVIRONMENT.step(np.argmax(action))
    DNN_REWARD_EPISODE += reward
    if done:
        obs = ENVIRONMENT.reset()
        DNN_REWARDS.append(DNN_REWARD_EPISODE)
        CURRENT_EPISODE += 1
        DNN_REWARD_EPISODE = 0
        if CURRENT_EPISODE > EPISODES:
            break
print(f"Average Reward DNN: {np.round(np.mean(DNN_REWARDS), 2)} +/- "
      f"{np.round(np.std(DNN_REWARDS), 2)}")

weights = DQN_MC_DNN.get_weights()
weights = gf.reshape_weights(weights)
layer_sizes = gf.get_layer_sizes(weights)
deepred_net = DeepRedFCNet(layer_sizes)
deepred_net.init_eval_weights(weights=weights)
l_activation = deepred_net.eval_layers(np.array(x_train).reshape(len(x_train), 2))

dr_params = dict()
dr_params[deepred.VARBOSE] = 2

rt = deepred.deepred(l_activation[:-1], dr_params)
rt.view_graph()
gf.Ruletree_to_string(rule=rt.root, depth=0)

y_DeepRED = rt.eval_all(x_train)
correct = 0
for i in range(len(y_DeepRED)):
    if y_DeepRED[i] == DQN_MC_ACTIONS[i]:
        correct += 1
print(f"DeepRED Accuracy: {(correct / len(y_DeepRED)) * 100}%")

DEEPRED_REWARDS = []
DEEPRED_REWARD_EPISODE = 0
CURRENT_EPISODE = 1

obs = ENVIRONMENT.reset()
while True:
    action = rt.eval_one(obs)
    obs, reward, done, info = ENVIRONMENT.step(list(action)[0])
    DEEPRED_REWARD_EPISODE += reward
    if done:
        obs = ENVIRONMENT.reset()
        DEEPRED_REWARDS.append(DEEPRED_REWARD_EPISODE)
        CURRENT_EPISODE += 1
        DEEPRED_REWARD_EPISODE = 0
        if CURRENT_EPISODE > EPISODES:
            break
print(f"Average Reward DeepRED: {np.round(np.mean(DEEPRED_REWARDS), 2)} +/- "
      f"{np.round(np.std(DEEPRED_REWARDS), 2)}")

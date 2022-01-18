import numpy as np
import pydotplus
import tensorflow as tf
import torch as th
from joblib import dump, load
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import np_utils
from ruleex import deepred
from ruleex.deepred.model import DeepRedFCNet
from sklearn import tree
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, VecMonitor
from tensorflow import keras

import DR_helper_functions as gf

CREATE_DRL_MODEL = True
CREATE_TREE = True
CREATE_DNN = True
TREE_DEPTH = 3
NUMBER_ENVIRONMENTS = 16
LOG_NAME = "PPO_AB_TENSORBOARD"
ENVIRONMENT = make_vec_env("Acrobot-v1", n_envs=NUMBER_ENVIRONMENTS)
ENVIRONMENT = VecNormalize(ENVIRONMENT, norm_obs=True, norm_reward=True)
tf.compat.v1.disable_eager_execution()
EPISODES = 100

# ------------------------------------------------------------------------------------------------------------------
# PROXIMAL POLICY OPTIMIZATION
# ------------------------------------------------------------------------------------------------------------------

policy_kwargs = dict(activation_fn=th.nn.Tanh, net_arch=[dict(pi=[64, 64], vf=[64, 64])], ortho_init=False)
venv = VecMonitor(venv=ENVIRONMENT)

PPO_AB_MODEL = PPO('MlpPolicy', venv, policy_kwargs=policy_kwargs, n_steps=8, gamma=0.999, batch_size=512,
                   learning_rate=0.00015312200830971642, ent_coef=1.777074275127635e-05, clip_range=0.3,
                   n_epochs=20, gae_lambda=0.98, max_grad_norm=1, vf_coef=0.3448244058995513, verbose=1,
                   tensorboard_log=f"./{LOG_NAME}/")

if CREATE_DRL_MODEL:
    PPO_AB_MODEL.learn(total_timesteps=int(5e5))
    PPO_AB_MODEL.save("PPO_AB")
else:
    PPO_AB_MODEL = PPO.load("PPO_AB")

PPO_OBSERVATIONS = []
PPO_ACTIONS = []

episode_rewards = []
episode_lengths = []

episode_counts = np.zeros(NUMBER_ENVIRONMENTS, dtype="int")
episode_count_targets = np.array([(EPISODES + i) // NUMBER_ENVIRONMENTS for i in range(NUMBER_ENVIRONMENTS)],
                                 dtype="int")

current_rewards = np.zeros(NUMBER_ENVIRONMENTS)
current_lengths = np.zeros(NUMBER_ENVIRONMENTS, dtype="int")
observations = ENVIRONMENT.reset()
states = None
while (episode_counts < episode_count_targets).any():
    actions, states = PPO_AB_MODEL.predict(observations, state=states, deterministic=True)
    PPO_OBSERVATIONS.append(observations)
    PPO_ACTIONS.append(actions)
    observations, rewards, dones, infos = ENVIRONMENT.step(actions)
    current_rewards += VecNormalize.unnormalize_reward(ENVIRONMENT, rewards)
    current_lengths += 1
    for i in range(NUMBER_ENVIRONMENTS):
        if episode_counts[i] < episode_count_targets[i]:
            if dones[i]:
                episode_rewards.append(current_rewards[i])
                episode_lengths.append(current_lengths[i])
                episode_counts[i] += 1
                current_rewards[i] = 0
                current_lengths[i] = 0

PPO_AB_REWARDS_MEAN = np.mean(episode_rewards)
PPO_AB_REWARDS_STD = np.std(episode_rewards)

print(f"Average Reward Oracle: {np.round(PPO_AB_REWARDS_MEAN, 2)} +/- {np.round(PPO_AB_REWARDS_STD, 2)}")

# ------------------------------------------------------------------------------------------------------------------
# Classification Tree
# ------------------------------------------------------------------------------------------------------------------

INPUT_TREE_X = []
INPUT_TREE_Y = []

for i in range(len(PPO_OBSERVATIONS)):
    for j in range(NUMBER_ENVIRONMENTS):
        INPUT_TREE_X.append(PPO_OBSERVATIONS[i][j])
        INPUT_TREE_Y.append(PPO_ACTIONS[i][j])

if CREATE_TREE:
    CLASSIFICATION_TREE = tree.DecisionTreeClassifier(max_depth=TREE_DEPTH)
    CLASSIFICATION_TREE.fit(X=INPUT_TREE_X, y=INPUT_TREE_Y)
    dump(CLASSIFICATION_TREE, "PPO_AB_CLF")
else:
    CLASSIFICATION_TREE = load("PPO_AB_CLF")

TREE_REWARDS = []
TREE_REWARD_EPISODE = 0
CURRENT_EPISODE = 1

ENVIRONMENT = make_vec_env("Acrobot-v1", n_envs=1)
ENVIRONMENT = VecNormalize(ENVIRONMENT, norm_obs=True, norm_reward=True)
obs = ENVIRONMENT.reset()

while True:
    action = CLASSIFICATION_TREE.predict(obs[0].reshape(1, -1))
    obs, reward, done, info = ENVIRONMENT.step([action[0]])
    TREE_REWARD_EPISODE += VecNormalize.unnormalize_reward(ENVIRONMENT, reward)
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
                                feature_names=["cos theta1", "sin theta1", "cos theta2", "sin theta2", "v1", "v2"],
                                class_names=["Drehe Links", "Drehe Nicht", "Drehe Rechts"],
                                filled=True)

graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_png(f'PPO_AB_DT_{TREE_DEPTH}.png')

DT_text = tree.export_text(CLASSIFICATION_TREE, feature_names=["obs[0]", "obs[1]", "obs[2]", "obs[3]", "obs[4]",
                                                               "obs[5]"])
print(DT_text.replace('--- class:', 'return').replace('--- obs', 'if obs').replace('--- value: ', 'return ').
      replace('\n', ':\n').replace('|', '').replace('return 0:', 'return 0').replace('return 1:', 'return 1').replace(
    'return 2:', 'return 2').replace('\t', '').replace(']:', ']'))

# ------------------------------------------------------------------------------------------------------------------
# Deep Neural Network / DeepRED
# ------------------------------------------------------------------------------------------------------------------

x_train = np.array(INPUT_TREE_X)
y_train = np.array(INPUT_TREE_Y)
y_train = np_utils.to_categorical(y_train, 3)

PPO_AB_DNN = Sequential()
PPO_AB_DNN.add(Dense(256, input_dim=6, kernel_initializer='random_uniform', activation='sigmoid', use_bias=True))
PPO_AB_DNN.add(Dense(256, kernel_initializer='random_uniform', activation='sigmoid', use_bias=True))
PPO_AB_DNN.add(Dense(3, kernel_initializer='random_uniform', activation='sigmoid', use_bias=True))
PPO_AB_DNN.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=0.01),
                   metrics=['accuracy'])
if CREATE_DNN:
    history = PPO_AB_DNN.fit(x_train, y_train, batch_size=256, epochs=150)
    accuracy = PPO_AB_DNN.evaluate(x_train, y_train, verbose=1)[1]
    print("Training accuracy: " + str(accuracy * 100) + "%")
    PPO_AB_DNN.save('PPO_AB_DNN')
else:
    PPO_AB_DNN = keras.models.load_model('PPO_AB_DNN')

DNN_REWARDS = []
DNN_REWARD_EPISODE = 0
CURRENT_EPISODE = 1

obs = ENVIRONMENT.reset()
while True:
    action = PPO_AB_DNN.predict(obs[0].reshape(1, -1))
    obs, reward, done, info = ENVIRONMENT.step([np.argmax(action)])
    DNN_REWARD_EPISODE += VecNormalize.unnormalize_reward(ENVIRONMENT,reward)
    if done:
        obs = ENVIRONMENT.reset()
        DNN_REWARDS.append(DNN_REWARD_EPISODE)
        CURRENT_EPISODE += 1
        DNN_REWARD_EPISODE = 0
        if CURRENT_EPISODE > EPISODES:
            break
print(f"Average Reward DNN: {np.round(np.mean(DNN_REWARDS), 2)} +/- "
      f"{np.round(np.std(DNN_REWARDS), 2)}")

weights = PPO_AB_DNN.get_weights()
weights = gf.reshape_weights(weights)
layer_sizes = gf.get_layer_sizes(weights)
deepred_net = DeepRedFCNet(layer_sizes)
deepred_net.init_eval_weights(weights=weights)
l_activation = deepred_net.eval_layers(np.array(x_train).reshape(len(x_train), 6))

dr_params = dict()
dr_params[deepred.VARBOSE] = 2

rt = deepred.deepred(l_activation[:-1], dr_params)
rt.view_graph()
gf.Ruletree_to_string(rule=rt.root, depth=0)

y_DeepRED = rt.eval_all(x_train)
correct = 0
for i in range(len(y_DeepRED)):
    if y_DeepRED[i] == INPUT_TREE_Y[i]:
        correct += 1
print(f"DeepRED Accuracy: {(correct/len(y_DeepRED)) * 100}%")

DEEPRED_REWARDS = []
DEEPRED_REWARD_EPISODE = 0
CURRENT_EPISODE = 1

obs = ENVIRONMENT.reset()
while True:
    action = rt.eval_one(obs[0])
    obs, reward, done, info = ENVIRONMENT.step([list(action)[0]])
    DEEPRED_REWARD_EPISODE += VecNormalize.unnormalize_reward(ENVIRONMENT, reward)
    if done:
        obs = ENVIRONMENT.reset()
        DEEPRED_REWARDS.append(DEEPRED_REWARD_EPISODE)
        CURRENT_EPISODE += 1
        DEEPRED_REWARD_EPISODE = 0
        if CURRENT_EPISODE > EPISODES:
            break
print(f"Average Reward DeepRED: {np.round(np.mean(DEEPRED_REWARDS), 2)} +/- "
      f"{np.round(np.std(DEEPRED_REWARDS), 2)}")

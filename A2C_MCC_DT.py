from typing import Callable

import numpy as np
import pydotplus
import torch as th
from joblib import dump, load
from keras.layers import Dense
from keras.models import Sequential
from sklearn import tree
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, VecMonitor
from tensorflow import keras

CREATE_MODEL = True
CREATE_TREE = True
CREATE_DNN = True
TREE_DEPTH = 3
NUMBER_ENVIRONMENTS = 4
LOG_NAME = "A2C_MCC_TENSORBOARD"
ENVIRONMENT = make_vec_env("MountainCarContinuous-v0", n_envs=NUMBER_ENVIRONMENTS)
ENVIRONMENT = VecNormalize(ENVIRONMENT, norm_obs=True, norm_reward=True)
EPISODES = 100


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value

    return func


# ------------------------------------------------------------------------------------------------------------------
# Advantage Actor Critic
# ------------------------------------------------------------------------------------------------------------------

policy_kwargs = dict(ortho_init=False, activation_fn=th.nn.Tanh, net_arch=[dict(pi=[64, 64], vf=[64, 64])])
venv = VecMonitor(venv=ENVIRONMENT)

A2C_MCC_MODEL = A2C("MlpPolicy", ENVIRONMENT, policy_kwargs=policy_kwargs, use_rms_prop=True, verbose=1, n_steps=3,
                    use_sde=True, tensorboard_log=f"./{LOG_NAME}/")

if CREATE_MODEL:
    A2C_MCC_MODEL.learn(total_timesteps=int(1.5e5))
    A2C_MCC_MODEL.save("A2C_MCC")
else:
    A2C_MCC_MODEL = A2C.load("A2C_MCC")

A2C_OBSERVATIONS = []
A2C_ACTIONS = []

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
    actions, states = A2C_MCC_MODEL.predict(observations, state=states, deterministic=True)
    A2C_OBSERVATIONS.append(observations)
    A2C_ACTIONS.append(actions)
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

A2C_MCC_REWARDS_MEAN = np.mean(episode_rewards)
A2C_MCC_REWARDS_STD = np.std(episode_rewards)

print(f"Average Reward Oracle: {np.round(A2C_MCC_REWARDS_MEAN, 2)} +/- {np.round(A2C_MCC_REWARDS_STD, 2)}")

# ------------------------------------------------------------------------------------------------------------------
# Regression Tree
# ------------------------------------------------------------------------------------------------------------------

INPUT_TREE_X = []
INPUT_TREE_Y = []

for i in range(len(A2C_OBSERVATIONS)):
    for j in range(NUMBER_ENVIRONMENTS):
        INPUT_TREE_X.append(A2C_OBSERVATIONS[i][j])
        INPUT_TREE_Y.append(A2C_ACTIONS[i][j])

if CREATE_TREE:
    REGRESSION_TREE = tree.DecisionTreeRegressor(max_depth=TREE_DEPTH)
    REGRESSION_TREE.fit(X=INPUT_TREE_X, y=INPUT_TREE_Y)
    dump(REGRESSION_TREE, "A2C_MCC_RGR")
else:
    REGRESSION_TREE = load("A2C_MCC_RGR")

TREE_REWARDS = []
TREE_REWARD_EPISODE = 0
CURRENT_EPISODE = 1

ENVIRONMENT = make_vec_env("MountainCarContinuous-v0", n_envs=1)
ENVIRONMENT = VecNormalize(ENVIRONMENT, norm_obs=True, norm_reward=True)
obs = ENVIRONMENT.reset()
EPISODES = 100

while True:
    action = REGRESSION_TREE.predict(obs[0].reshape(1, -1))
    obs, reward, done, info = ENVIRONMENT.step([action])
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

dot_data = tree.export_graphviz(REGRESSION_TREE, out_file=None,
                                feature_names=["Auto Position", "Auto Geschwindigkeit"],
                                class_names=["Leistungs-Koeffizient"],
                                filled=True)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_png(f'A2C_MCC_DT_{TREE_DEPTH}.png')

DT_text = tree.export_text(REGRESSION_TREE, feature_names=["obs[0]", "obs[1]"])
print(DT_text.replace('--- class:', 'return').replace('--- obs', 'if obs').replace('--- value: ', 'return ').
      replace('\n', ':\n').replace('|', '').replace(']:', ']'))

# ------------------------------------------------------------------------------------------------------------------
# Deep Neural Network
# ------------------------------------------------------------------------------------------------------------------

x_train = np.array(INPUT_TREE_X)
y_train = np.array(INPUT_TREE_Y)

A2C_MCC_DNN = Sequential()
A2C_MCC_DNN.add(Dense(512, input_dim=2, kernel_initializer='normal', activation='selu', use_bias=True))
A2C_MCC_DNN.add(Dense(512, kernel_initializer='normal', activation='selu', use_bias=True))
A2C_MCC_DNN.add(Dense(512, kernel_initializer='normal', activation='selu', use_bias=True))
A2C_MCC_DNN.add(Dense(1, kernel_initializer='normal', activation='selu', use_bias=True))
A2C_MCC_DNN.compile(loss='mae', optimizer=keras.optimizers.RMSprop(learning_rate=0.0001))
if CREATE_DNN:
    history = A2C_MCC_DNN.fit(x_train, y_train, batch_size=128, epochs=100)
    accuracy = A2C_MCC_DNN.evaluate(x_train, y_train, verbose=1)
    A2C_MCC_DNN.save('A2C_MCC_DNN')
else:
    A2C_MC_DNN = keras.models.load_model('A2C_MCC_DNN')

DNN_REWARDS = []
DNN_REWARD_EPISODE = 0
CURRENT_EPISODE = 1

obs = ENVIRONMENT.reset()
while True:
    action = A2C_MCC_DNN.predict(obs[0].reshape(1, -1))
    obs, reward, done, info = ENVIRONMENT.step([action[0]])
    DNN_REWARD_EPISODE += VecNormalize.unnormalize_reward(ENVIRONMENT, reward)
    if done:
        obs = ENVIRONMENT.reset()
        DNN_REWARDS.append(DNN_REWARD_EPISODE)
        CURRENT_EPISODE += 1
        DNN_REWARD_EPISODE = 0
        if CURRENT_EPISODE > EPISODES:
            break
print(f"Average Reward DNN: {np.round(np.mean(DNN_REWARDS), 2)} +/- "
      f"{np.round(np.std(DNN_REWARDS), 2)}")

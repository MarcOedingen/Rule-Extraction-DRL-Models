import numpy as np
import pydotplus
import torch as th
from joblib import dump, load
from keras.layers import Dense
from keras.models import Sequential
from sklearn import tree
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, VecMonitor
from tensorflow import keras

CREATE_DRL_MODEL = True
CREATE_TREE = True
CREATE_DNN = True
TREE_DEPTH = 3
NUMBER_ENVIRONMENTS = 4
LOG_NAME = "PPO_MCC_TENSORBOARD"
ENVIRONMENT = make_vec_env("MountainCarContinuous-v0", n_envs=NUMBER_ENVIRONMENTS)
ENVIRONMENT = VecNormalize(ENVIRONMENT, norm_obs=True, norm_reward=True)
EPISODES = 100

# ------------------------------------------------------------------------------------------------------------------
# PROXIMAL POLICY OPTIMIZATION
# ------------------------------------------------------------------------------------------------------------------

policy_kwargs = dict(ortho_init=False, activation_fn=th.nn.Tanh, net_arch=[dict(pi=[64, 64], vf=[64, 64])])
venv = VecMonitor(venv=ENVIRONMENT)

PPO_MCC_MODEL = PPO('MlpPolicy', ENVIRONMENT, policy_kwargs=policy_kwargs, batch_size=32, n_steps=64,
                    use_sde=True, verbose=0)

if CREATE_DRL_MODEL:
    PPO_MCC_MODEL.learn(total_timesteps=int(8e4))
    PPO_MCC_MODEL.save("PPO_MCC")
else:
    PPO_MCC_MODEL = PPO.load("PPO_MCC")

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
    actions, states = PPO_MCC_MODEL.predict(observations, state=states, deterministic=True)
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

PPO_MCC_REWARDS_MEAN = np.mean(episode_rewards)
PPO_MCC_REWARDS_STD = np.std(episode_rewards)

print(f"Average Reward Oracle: {np.round(PPO_MCC_REWARDS_MEAN, 2)} +/- {np.round(PPO_MCC_REWARDS_STD, 2)}")

# ------------------------------------------------------------------------------------------------------------------
# Regression Tree
# ------------------------------------------------------------------------------------------------------------------

INPUT_TREE_X = []
INPUT_TREE_Y = []

for i in range(len(PPO_OBSERVATIONS)):
    for j in range(NUMBER_ENVIRONMENTS):
        INPUT_TREE_X.append(PPO_OBSERVATIONS[i][j])
        INPUT_TREE_Y.append(PPO_ACTIONS[i][j])

if CREATE_TREE:
    REGRESSION_TREE = tree.DecisionTreeRegressor(max_depth=TREE_DEPTH)
    REGRESSION_TREE.fit(X=INPUT_TREE_X, y=INPUT_TREE_Y)
    dump(REGRESSION_TREE, "PPO_MCC_RGR")
else:
    REGRESSION_TREE = load("PPO_MCC_RGR")

TREE_REWARDS = []
TREE_REWARD_EPISODE = 0
CURRENT_EPISODE = 1

ENVIRONMENT = make_vec_env("MountainCarContinuous-v0", n_envs=1)
ENVIRONMENT = VecNormalize(ENVIRONMENT, norm_obs=True, norm_reward=True)
obs = ENVIRONMENT.reset()

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
graph.write_png(f'PPO_MCC_DT_{TREE_DEPTH}.png')

DT_text = tree.export_text(REGRESSION_TREE, feature_names=["obs[0]", "obs[1]"])
print(DT_text.replace('--- class:', 'return').replace('--- obs', 'if obs').replace('--- value: ', 'return ').
      replace('\n', ':\n').replace('|', '').replace('\t', '').replace(']:', ']'))

# ------------------------------------------------------------------------------------------------------------------
# Deep Neural Network
# ------------------------------------------------------------------------------------------------------------------

x_train = np.array(INPUT_TREE_X)
y_train = np.array(INPUT_TREE_Y)

PPO_MCC_DNN = Sequential()
PPO_MCC_DNN.add(Dense(512, input_dim=2, kernel_initializer='normal', activation='selu', use_bias=True))
PPO_MCC_DNN.add(Dense(512, kernel_initializer='normal', activation='selu', use_bias=True))
PPO_MCC_DNN.add(Dense(512, kernel_initializer='normal', activation='selu', use_bias=True))
PPO_MCC_DNN.add(Dense(1, kernel_initializer='normal', activation='selu', use_bias=True))
PPO_MCC_DNN.compile(loss='mae', optimizer=keras.optimizers.RMSprop(learning_rate=0.0001))
if CREATE_DNN:
    history = PPO_MCC_DNN.fit(x_train, y_train, batch_size=128, epochs=100)
    accuracy = PPO_MCC_DNN.evaluate(x_train, y_train, verbose=1)
    PPO_MCC_DNN.save('PPO_MCC_DNN')
else:
    PPO_MC_DNN = keras.models.load_model('PPO_MCC_DNN')

DNN_REWARDS = []
DNN_REWARD_EPISODE = 0
CURRENT_EPISODE = 1

obs = ENVIRONMENT.reset()
while True:
    action = PPO_MCC_DNN.predict(obs[0].reshape(1, -1))
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

import pydotplus
import numpy as np
import tensorflow as tf
from sklearn import tree
from ruleex import deepred
from tensorflow import keras
from joblib import dump, load
import general_functions as gf
from keras.layers import Dense
from keras.utils import np_utils
from stable_baselines3 import PPO
from keras.models import Sequential
from ruleex.deepred.model import DeepRedFCNet
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

CREATE_DRL_MODEL = False
CREATE_TREE = False
CREATE_DNN = True
TREE_DEPTH = 1
NUMBER_ENVIRONMENTS = 8
LOG_NAME = "PPO_MC_TENSORBOARD"
ENVIRONMENT = make_vec_env("CartPole-v1", n_envs=NUMBER_ENVIRONMENTS)
tf.compat.v1.disable_eager_execution()
EPISODES = 100

# ------------------------------------------------------------------------------------------------------------------
# PROXIMAL POLICY OPTIMIZATION
# ------------------------------------------------------------------------------------------------------------------
env = make_vec_env("CartPole-v1", n_envs=8)
PPO_CP_MODEL = PPO("MlpPolicy", env, n_steps=32, batch_size=256, gae_lambda=0.8, gamma=0.98, n_epochs=20, ent_coef=0.0,
                   learning_rate=0.001, clip_range=0.2, verbose=1, tensorboard_log=f"./{LOG_NAME}/")

if CREATE_DRL_MODEL:
    PPO_CP_MODEL.learn(total_timesteps=int(1e5))
    PPO_CP_MODEL.save("PPO_CP")
    mean_reward, std_reward = evaluate_policy(PPO_CP_MODEL, ENVIRONMENT, n_eval_episodes=100)
    print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
else:
    PPO_CP_MODEL = PPO.load("PPO_CP")

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
    actions, states = PPO_CP_MODEL.predict(observations, state=states, deterministic=True)
    PPO_OBSERVATIONS.append(observations)
    PPO_ACTIONS.append(actions)
    observations, rewards, dones, infos = ENVIRONMENT.step(actions)
    current_rewards += rewards
    current_lengths += 1
    for i in range(NUMBER_ENVIRONMENTS):
        if episode_counts[i] < episode_count_targets[i]:
            if dones[i]:
                episode_rewards.append(current_rewards[i])
                episode_lengths.append(current_lengths[i])
                episode_counts[i] += 1
                current_rewards[i] = 0
                current_lengths[i] = 0

PPO_MCP_REWARDS_MEAN = np.mean(episode_rewards)
PPO_CP_REWARDS_STD = np.std(episode_rewards)

print(f"Average Reward Oracle: {np.round(PPO_MCP_REWARDS_MEAN, 2)} +/- {np.round(PPO_CP_REWARDS_STD, 2)}")

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
    dump(CLASSIFICATION_TREE, "PPO_CP_CLF")
else:
    CLASSIFICATION_TREE = load("PPO_CP_CLF")

TREE_REWARDS = []
TREE_REWARD_EPISODE = 0
CURRENT_EPISODE = 1

ENVIRONMENT = make_vec_env("CartPole-v1", n_envs=1)
obs = ENVIRONMENT.reset()
EPISODES = 100

while True:
    action = CLASSIFICATION_TREE.predict(obs)
    obs, reward, done, info = ENVIRONMENT.step([action[0]])
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
                                feature_names=["Wagen Position", "Wagen Geschwindigkeit", "Stange Winkel",
                                               "Stange Winkel Geschwindigkeit"],
                                class_names=["Schiebe Links", "Schiebe Rechts"],
                                filled=True)

graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_png(f'PPO_CP_DT_{TREE_DEPTH}.png')

DT_text = tree.export_text(CLASSIFICATION_TREE, feature_names=["obs[0]", "obs[1]", "obs[2]", "obs[3]"])
print(DT_text.replace('--- class:', 'return').replace('--- obs', 'if obs').replace('--- value: ', 'return ').
      replace('\n', ':\n').replace('|', '').replace('return 0:', 'return 0').replace('return 1:', 'return 1').
      replace('\t', '').replace(']:', ']'))

# ------------------------------------------------------------------------------------------------------------------
# Deep Neural Network / DeepRED
# ------------------------------------------------------------------------------------------------------------------

x_train = np.array(INPUT_TREE_X)
y_train = np.array(INPUT_TREE_Y)
y_train = np_utils.to_categorical(y_train, 2)

PPO_CP_DNN = Sequential()
PPO_CP_DNN.add(Dense(256, input_dim=4, kernel_initializer='normal', activation='sigmoid', use_bias=True))
PPO_CP_DNN.add(Dense(256, kernel_initializer='normal', activation='sigmoid', use_bias=True))
PPO_CP_DNN.add(Dense(2, kernel_initializer='normal', activation='sigmoid', use_bias=True))
PPO_CP_DNN.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.RMSprop(learning_rate=0.01),
                   metrics=['accuracy'])
if CREATE_DNN:
    history = PPO_CP_DNN.fit(x_train, y_train, batch_size=256, epochs=150)
    accuracy = PPO_CP_DNN.evaluate(x_train, y_train, verbose=1)[1]
    print("Training accuracy: " + str(accuracy * 100) + "%")
    PPO_CP_DNN.save('PPO_CP_DNN')
else:
    PPO_CP_DNN = keras.models.load_model('PPO_CP_DNN')

DNN_REWARDS = []
DNN_REWARD_EPISODE = 0
CURRENT_EPISODE = 1

obs = ENVIRONMENT.reset()
while True:
    action = PPO_CP_DNN.predict(obs[0].reshape(1, -1))
    obs, reward, done, info = ENVIRONMENT.step([np.argmax(action)])
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

weights = PPO_CP_DNN.get_weights()
weights = gf.reshape_weights(weights)
layer_sizes = gf.get_layer_sizes(weights)
deepred_net = DeepRedFCNet(layer_sizes)
deepred_net.init_eval_weights(weights=weights)
l_activation = deepred_net.eval_layers(np.array(x_train).reshape(len(x_train), 4))

dr_params = dict()
dr_params[deepred.VARBOSE] = 2

rt = deepred.deepred(l_activation[:-1], dr_params)
rt.view_graph()
gf.Ruletree_to_string(rule=rt.root, depth=0)

x = rt.eval_all(x_train)
correct = 0
for i in range(len(x)):
    if x[i] == INPUT_TREE_Y[i]:
        correct += 1
print(f"DeepRED Accuracy: {(correct/len(x)) * 100}%")

DEEPRED_REWARDS = []
DEEPRED_REWARD_EPISODE = 0
CURRENT_EPISODE = 1

obs = ENVIRONMENT.reset()
while True:
    action = rt.eval_one(obs[0])
    obs, reward, done, info = ENVIRONMENT.step([list(action)[0]])
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

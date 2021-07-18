import os
import copy
import json
import numpy as np
from model import Shared_Model
from tensorflow.keras.optimizers import Adam, RMSprop
from datetime import datetime
from tensorboardX import SummaryWriter


class CustomAgent:
    # A custom Bitcoin trading agent
    def __init__(self, lookback_window_size=50, lr=0.00005, epochs=1, optimizer=Adam, batch_size=32, model="", depth=0,
                 comment=""):
        self.lookback_window_size = lookback_window_size
        self.model = model
        self.comment = comment
        self.depth = depth

        # Action space from 0 to 3, 0 is hold, 1 is buy, 2 is sell
        self.action_space = np.array([0, 1, 2])

        # folder to save models
        self.log_name = datetime.now().strftime("%Y_%m_%d_%H_%M")

        # State size contains Market+Orders+Indicators history for the last lookback_window_size steps
        self.state_size = (lookback_window_size, 5 + depth)  # 5 standard OHCL information + market and indicators

        # Neural Networks part bellow
        self.lr = lr
        self.epochs = epochs
        self.optimizer = optimizer
        self.batch_size = batch_size

        # Create shared Actor-Critic network model
        self.Actor = self.Critic = Shared_Model(input_shape=self.state_size, action_space=self.action_space.shape[0],
                                                lr=self.lr, optimizer=self.optimizer, model=self.model)
        # Create Actor-Critic network model
        # self.Actor = Actor_Model(input_shape=self.state_size, action_space = self.action_space.shape[0], lr=self.lr, optimizer = self.optimizer)
        # self.Critic = Critic_Model(input_shape=self.state_size, action_space = self.action_space.shape[0], lr=self.lr, optimizer = self.optimizer)

    # create tensorboard writer
    def create_writer(self, initial_balance, normalize_value, train_episodes):
        self.replay_count = 0
        self.writer = SummaryWriter('runs/' + self.log_name)

        # Create folder to save models
        if not os.path.exists(self.log_name):
            os.makedirs(self.log_name)

        self.start_training_log(initial_balance, normalize_value, train_episodes)

    def start_training_log(self, initial_balance, normalize_value, train_episodes):
        # save training parameters to Parameters.json file for future
        current_date = datetime.now().strftime('%Y-%m-%d %H:%M')
        params = {
            "training start": current_date,
            "initial balance": initial_balance,
            "training episodes": train_episodes,
            "lookback window size": self.lookback_window_size,
            "depth": self.depth,
            "lr": self.lr,
            "epochs": self.epochs,
            "batch size": self.batch_size,
            "normalize value": normalize_value,
            "model": self.model,
            "comment": self.comment,
            "saving time": "",
            "Actor name": "",
            "Critic name": "",
        }
        with open(self.log_name + "/Parameters.json", "w") as write_file:
            json.dump(params, write_file, indent=4)

    def get_gaes(self, rewards, dones, values, next_values, gamma=0.99, lamda=0.95, normalize=True):
        deltas = [r + gamma * (1 - d) * nv - v for r, d, nv, v in zip(rewards, dones, next_values, values)]
        deltas = np.stack(deltas)
        gaes = copy.deepcopy(deltas)
        for t in reversed(range(len(deltas) - 1)):
            gaes[t] = gaes[t] + (1 - dones[t]) * gamma * lamda * gaes[t + 1]

        target = gaes + values
        if normalize:
            gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-8)
        return np.vstack(gaes), np.vstack(target)

    def replay(self, states, actions, rewards, predictions, dones, next_states):
        # reshape memory to appropriate shape for training
        states = np.vstack(states)
        next_states = np.vstack(next_states)
        actions = np.vstack(actions)
        predictions = np.vstack(predictions)

        # Get Critic network predictions
        values = self.Critic.critic_predict(states)
        next_values = self.Critic.critic_predict(next_states)

        # Compute advantages
        advantages, target = self.get_gaes(rewards, dones, np.squeeze(values), np.squeeze(next_values))

        # stack everything to numpy array
        y_true = np.hstack([advantages, predictions, actions])

        # training Actor and Critic networks
        a_loss = self.Actor.Actor.fit(states, y_true, epochs=self.epochs, verbose=0,
                                      batch_size=self.batch_size)
        c_loss = self.Critic.Critic.fit(states, target, epochs=self.epochs, verbose=0,
                                        batch_size=self.batch_size)

        self.writer.add_scalar('Data/actor_loss_per_replay', np.sum(a_loss.history['loss']), self.replay_count)
        self.writer.add_scalar('Data/critic_loss_per_replay', np.sum(c_loss.history['loss']), self.replay_count)
        self.replay_count += 1

        return np.sum(a_loss.history['loss']), np.sum(c_loss.history['loss'])

    def act(self, state):
        # TODO: see the probability
        # Use the network to predict the next action to take, using the model
        prediction = self.Actor.actor_predict(np.expand_dims(state, axis=0))[0]
        action = np.random.choice(self.action_space, p=prediction)
        return action, prediction

    def save(self, name="", score="", args=[]):
        # save keras model weights
        self.Actor.Actor.save_weights(f"{self.log_name}/{score}_{name}_Actor.h5")
        self.Critic.Critic.save_weights(f"{self.log_name}/{score}_{name}_Critic.h5")

        # update json file settings
        if score != "":
            with open(self.log_name + "/Parameters.json", "r") as json_file:
                params = json.load(json_file)
            params["saving time"] = datetime.now().strftime('%Y-%m-%d %H:%M')
            params["Actor name"] = f"{score}_{name}_Actor.h5"
            params["Critic name"] = f"{score}_{name}_Critic.h5"
            with open(self.log_name + "/Parameters.json", "w") as write_file:
                json.dump(params, write_file, indent=4)

        # log saved model arguments to file
        if len(args) > 0:
            with open(f"{self.log_name}/log.txt", "a+") as log:
                current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                arguments = ""
                for arg in args:
                    arguments += f", {arg}"
                log.write(f"{current_time}{arguments}\n")

    def load(self, folder, name):
        # load keras model weights
        self.Actor.Actor.load_weights(os.path.join(folder, f"{name}_Actor.h5"))
        self.Critic.Critic.load_weights(os.path.join(folder, f"{name}_Critic.h5"))

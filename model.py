import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Conv1D, MaxPooling1D, LSTM
from tensorflow.keras import backend as K

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices: tf.config.experimental.set_memory_growth(device, True)

class Shared_Model:
    def __init__(self, input_shape, action_space, lr, optimizer, model="Dense"):
        X_input = Input(input_shape)
        self.action_space = action_space # 100 timesteps (batch_size), 15 features (market_history)

        # Shared CNN layers:
        # TODO: seperate see
        X = Conv1D(filters=64, kernel_size=6, padding="same", activation="tanh")(X_input) # 100 rows, 64 features
        X = MaxPooling1D(pool_size=2)(X) # 50 rows, 64 features
        X = Conv1D(filters=32, kernel_size=3, padding="same", activation="tanh")(X)  # 50 rows, 32 features
        X = MaxPooling1D(pool_size=2)(X)  # 25 rows, 32 features
        X = LSTM(32, return_sequences=True, input_shape=(X[0], 1))(X)
        X = Flatten()(X)

        # Critic model
        V = Dense(64, activation="relu")(X)
        V = Dense(32, activation="relu")(V)
        value = Dense(1, activation=None)(V)

        self.Critic = Model(inputs=X_input, outputs=value)
        self.Critic.compile(loss=self.critic_PPO2_loss, optimizer=optimizer(learning_rate=lr))

        # Actor model
        A = Dense(64, activation="relu")(X)
        A = Dense(32, activation="relu")(A)
        output = Dense(self.action_space, activation="softmax")(A)

        self.Actor = Model(inputs=X_input, outputs=output)
        self.Actor.compile(loss=self.ppo_loss, optimizer=optimizer(learning_rate=lr))
        # print(self.Actor.summary())

    def ppo_loss(self, y_true, y_pred):
        # Defined in https://arxiv.org/abs/1707.06347
        advantages, prediction_picks, actions = y_true[:, :1], y_true[:, 1:1 + self.action_space], y_true[:,
                                                                                                   1 + self.action_space:]
        LOSS_CLIPPING = 0.2
        ENTROPY_LOSS = 0.001

        prob = actions * y_pred
        old_prob = actions * prediction_picks

        prob = K.clip(prob, 1e-10, 1.0)
        old_prob = K.clip(old_prob, 1e-10, 1.0)

        ratio = K.exp(K.log(prob) - K.log(old_prob))

        p1 = ratio * advantages
        p2 = K.clip(ratio, min_value=1 - LOSS_CLIPPING, max_value=1 + LOSS_CLIPPING) * advantages

        actor_loss = -K.mean(K.minimum(p1, p2))

        entropy = -(y_pred * K.log(y_pred + 1e-10))
        entropy = ENTROPY_LOSS * K.mean(entropy)

        total_loss = actor_loss - entropy

        return total_loss

    def actor_predict(self, state):
        return self.Actor.predict(state)

    def critic_PPO2_loss(self, y_true, y_pred):
        value_loss = K.mean((y_true - y_pred) ** 2)  # standard PPO loss
        return value_loss

    def critic_predict(self, state):
        # return self.Critic.predict([state, np.zeros((state.shape[0], 1))])
        return self.Critic.predict(state)


class Actor_Model:
    def __init__(self, input_shape, action_space, lr, optimizer):
        X_input = Input(input_shape)
        self.action_space = action_space

        X = Conv1D(filters=64, kernel_size=6, padding="same", activation="tanh")(X_input)  # 100 rows, 64 features
        X = MaxPooling1D(pool_size=2)(X)  # 50 rows, 64 features
        X = Conv1D(filters=32, kernel_size=3, padding="same", activation="tanh")(X)  # 50 rows, 32 features
        X = MaxPooling1D(pool_size=2)(X)  # 25 rows, 32 features
        X = LSTM(32, return_sequences=True, input_shape=(X[0], 1))(X)
        X = Flatten()(X)
        A = Dense(64, activation="relu")(X)
        A = Dense(32, activation="relu")(A)
        output = Dense(self.action_space, activation="softmax")(A)

        self.Actor = Model(inputs=X_input, outputs=output)
        self.Actor.compile(loss=self.ppo_loss, optimizer=optimizer(lr=lr))

    def ppo_loss(self, y_true, y_pred):
        # Defined in https://arxiv.org/abs/1707.06347
        advantages, prediction_picks, actions = y_true[:, :1], y_true[:, 1:1 + self.action_space], y_true[:,
                                                                                                   1 + self.action_space:]
        LOSS_CLIPPING = 0.2
        ENTROPY_LOSS = 0.001

        prob = actions * y_pred
        old_prob = actions * prediction_picks

        prob = K.clip(prob, 1e-10, 1.0)
        old_prob = K.clip(old_prob, 1e-10, 1.0)

        ratio = K.exp(K.log(prob) - K.log(old_prob))

        p1 = ratio * advantages
        p2 = K.clip(ratio, min_value=1 - LOSS_CLIPPING, max_value=1 + LOSS_CLIPPING) * advantages

        actor_loss = -K.mean(K.minimum(p1, p2))

        entropy = -(y_pred * K.log(y_pred + 1e-10))
        entropy = ENTROPY_LOSS * K.mean(entropy)

        total_loss = actor_loss - entropy

        return total_loss

    def actor_predict(self, state):
        return self.Actor.predict(state)


class Critic_Model:
    def __init__(self, input_shape, action_space, lr, optimizer):
        X_input = Input(input_shape)
        self.action_space = action_space  # 100 timesteps (batch_size), 15 features (market_history)

        X = Conv1D(filters=64, kernel_size=6, padding="same", activation="tanh")(X_input)  # 100 rows, 64 features
        X = MaxPooling1D(pool_size=2)(X)  # 50 rows, 64 features
        X = Conv1D(filters=32, kernel_size=3, padding="same", activation="tanh")(X)  # 50 rows, 32 features
        X = MaxPooling1D(pool_size=2)(X)  # 25 rows, 32 features
        X = LSTM(32, return_sequences=True, input_shape=(X[0], 1))(X)
        X = Flatten()(X)
        V = Dense(64, activation="relu")(X)
        V = Dense(32, activation="relu")(V)
        value = Dense(1, activation=None)(V)
        self.Critic = Model(inputs=X_input, outputs=value)
        self.Critic.compile(loss=self.critic_PPO2_loss, optimizer=optimizer(lr=lr))

    def critic_PPO2_loss(self, y_true, y_pred):
        value_loss = K.mean((y_true - y_pred) ** 2)  # standard PPO loss
        return value_loss

    def critic_predict(self, state):
        return self.Critic.predict(state)
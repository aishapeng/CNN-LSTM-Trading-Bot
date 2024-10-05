import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Flatten, Conv1D, MaxPooling1D, LSTM, TimeDistributed
from tensorflow.keras import backend as K

# Enable XLA optimization
tf.config.optimizer.set_jit(True)  # Enabling XLA
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices: 
    tf.config.experimental.set_memory_growth(device, True)

class Shared_Model:
    def __init__(self, input_shape, action_space, lr, optimizer, model="Dense"):
        X_input = Input(input_shape)  # timesteps(lookback_window_size), features(depth)
        self.action_space = action_space # 100 timesteps (batch_size), 15 features (market_history)

        # Shared CNN layers:
        # TODO: seperate see
        # X = Conv1D(filters=32, kernel_size=3, padding="same", activation="tanh")(X_input) # 100 rows, 64 features
        # X = MaxPooling1D(pool_size=2)(X) # 50 rows, 64 features
        # # X = Conv1D(filters=32, kernel_size=3, padding="same", activation="tanh")(X)  # 50 rows, 32 features
        # # X = MaxPooling1D(pool_size=2)(X)  # 25 rows, 32 features
        # # print(X[0])
        # X = LSTM(32, return_sequences=True, input_shape=(1, X[1], X[2]))(X)
        # X = Flatten()(X)



        # Critic model
        # V = Dense(64, activation="relu")(X)
        # V = Dense(32, activation="relu")(X)
        # value = Dense(1, activation=None)(V)
        #
        # self.Critic = model.build(X_input)
        # self.Critic.compile(loss=self.critic_PPO2_loss, optimizer=optimizer(learning_rate=lr))
        #
        # # Actor model
        # # A = Dense(64, activation="relu")(X)
        # A = Dense(32, activation="relu")(X)
        # output = Dense(self.action_space, activation="softmax")(A)
        #
        # self.Actor = Model(inputs=X_input, outputs=output)
        # self.Actor.compile(loss=self.ppo_loss, optimizer=optimizer(learning_rate=lr))
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

    # def actor_predict(self, state):
    #     return self.Actor.predict(state)

    def critic_PPO2_loss(self, y_true, y_pred):
        value_loss = K.mean((y_true - y_pred) ** 2)  # standard PPO loss
        return value_loss

    # def critic_predict(self, state):
    #     # return self.Critic.predict([state, np.zeros((state.shape[0], 1))])
    #     return self.Critic.predict(state)


class Actor_Model:
    def __init__(self, input_shape, action_space, lr, optimizer):
        X_input = Input(input_shape)
        self.action_space = action_space

        # Optimized CNN + LSTM layers
        X = Conv1D(filters=64, kernel_size=3, padding="same", activation="relu")(X_input)
        X = MaxPooling1D(pool_size=2)(X)
        X = LSTM(64)(X)
        A = Dense(128, activation="relu")(X)
        output = Dense(self.action_space, activation="softmax")(A)

        self.Actor = Model(inputs=X_input, outputs=output)
        self.Actor.compile(loss=self.ppo_loss, optimizer=optimizer(learning_rate=lr))
        print(self.Actor.summary())

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

        # define CNN model
        # cnn = Sequential()
        # cnn.add(Conv1D(filters=32, kernel_size=3, padding="same", activation="relu", input_shape=input_shape))
        # cnn.add(MaxPooling1D(pool_size=2))
        # cnn.add(Flatten())
        # print(cnn.summary())
        # # define LSTM model
        # model = Sequential()
        # model.add(TimeDistributed(cnn))
        # model.add(LSTM(32))
        # model.add(Dense(32, activation="relu"))
        # model.add(Dense(1, activation=None))
        # model.compile(loss=self.critic_PPO2_loss, optimizer=optimizer(learning_rate=lr))
        # model.build(X_input)

        X = Conv1D(filters=32, kernel_size=3, padding="same", activation="tanh")(X_input)
        X = MaxPooling1D(pool_size=2)(X)  # 50 rows, 64 features
        X = LSTM(32)(X)
        X = Flatten()(X)
        V = Dense(32, activation="relu")(X)
        value = Dense(1, activation=None)(V)

        self.Critic = Model(inputs=X_input, outputs=value)
        self.Critic.compile(loss=self.critic_PPO2_loss, optimizer=optimizer(learning_rate=lr))


    def critic_PPO2_loss(self, y_true, y_pred):
        value_loss = K.mean((y_true - y_pred) ** 2)  # standard PPO loss
        return value_loss

    # def critic_predict(self, state):
    #     return self.Critic.predict(state)

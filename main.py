import json
import random
from tqdm import tqdm
from collections import deque
from agent import CustomAgent
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from utils import Normalizing
from datetime import datetime
from indicators import *


class CustomEnv:
    # A custom Bitcoin trading environment
    def __init__(self, df, df_original, initial_balance=1000, lookback_window_size=50):  # 40000
        # Define action space and state size and other custom parameters
        self.df = df.reset_index(drop=True)
        self.df_original = df_original.reset_index(drop=True)
        self.df_total_steps = len(self.df) - 1
        self.initial_balance = initial_balance
        self.lookback_window_size = lookback_window_size

        # Orders history contains the balance, net_worth, crypto_bought, crypto_sold, crypto_held values for the last lookback_window_size steps
        self.orders_history = deque(maxlen=self.lookback_window_size)

        # Market history contains the OHCL values for the last lookback_window_size prices
        self.market_history = deque(maxlen=self.lookback_window_size)

        self.fees = 0.001  # default Binance 0.1% order fees

        self.columns = list(self.df.columns[1:])

    # Reset the state of the environment to an initial state
    def reset(self, env_steps_size=0):

        self.trades = []
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.prev_net_worth = self.initial_balance
        self.crypto_held = 0
        self.crypto_sold = 0
        self.crypto_bought = 0
        self.episode_orders = 0  # track episode orders count
        self.prev_episode_orders = 0  # track previous episode orders count
        self.rewards = deque(maxlen=self.lookback_window_size)
        self.env_steps_size = env_steps_size
        # self.punish_value = 0
        if env_steps_size > 0:  # used for training dataset
            self.start_step = random.randint(self.lookback_window_size, self.df_total_steps - env_steps_size)
            self.end_step = self.start_step + env_steps_size
        else:  # used for testing dataset
            self.start_step = self.lookback_window_size
            self.end_step = self.df_total_steps

        self.current_step = self.start_step

        for i in reversed(range(self.lookback_window_size)):
            current_step = self.current_step - i
            self.market_history.append([self.df.loc[current_step, column] for column in self.columns])

        state = np.array(self.market_history)

        return state

    # Get the data points for the given current_step
    def next_observation(self):
        self.market_history.append([self.df.loc[self.current_step, column] for column in self.columns])
        obs = np.array(self.market_history)

        return obs

    # Execute one time step within the environment
    def step(self, action):
        self.crypto_bought = 0
        self.crypto_sold = 0
        self.current_step += 1

        current_price = self.df_original.loc[self.current_step, 'Open']
        timestamp = self.df_original.loc[self.current_step, 'Timestamp']
        high = self.df_original.loc[self.current_step, 'High']
        low = self.df_original.loc[self.current_step, 'Low']

        if action == 0:  # Hold
            pass

        elif action == 1 and self.balance > 10: # min 10 USD binance
            # Buy with 100% of current balance
            self.crypto_bought = self.balance / current_price
            self.balance -= self.crypto_bought * current_price
            self.crypto_bought *= (1 - self.fees)  # substract fees
            self.crypto_held += self.crypto_bought
            self.trades.append({'Timestamp': timestamp, 'High': high, 'Low': low, 'Total': self.crypto_bought, 'Type': "Buy",
                                'Current price': current_price})
            self.episode_orders += 1

        elif action == 2 and self.crypto_held * current_price > 10: # min 10 USD binance
            # Sell 100% of current crypto held
            self.crypto_sold = self.crypto_held
            self.crypto_held -= self.crypto_sold
            self.crypto_sold *= (1 - self.fees)  # substract fees
            self.balance += self.crypto_sold * current_price
            self.trades.append({'Timestamp': timestamp, 'High': high, 'Low': low, 'Total': self.crypto_sold, 'Type': "Sell",
                                'Current price': current_price})
            self.episode_orders += 1

        self.prev_net_worth = self.net_worth
        self.net_worth = self.balance + self.crypto_held * current_price

        # Calculate reward
        reward = self.net_worth - self.prev_net_worth

        if self.net_worth <= self.initial_balance / 2:
            done = True
        else:
            done = False

        obs = self.next_observation()

        return obs, reward, done

def train_agent(env, agent, train_dataset, train_episodes=50, training_batch_size=64):
    agent.create_writer(env.initial_balance, train_episodes, training_batch_size)  # create TensorBoard writer
    total_average = deque(maxlen=20)  # save recent 20 episodes net worth
    best_average = 0  # used to track best average net worth

    # Progress bar for episodes
    for episode in tqdm(range(1, train_episodes + 1), desc="Training Progress", ascii=True, unit='episodes'):
        for batch_data in tqdm(train_dataset, desc=f"Episode {episode} Batches", ascii=True, leave=False):
            # Get the state from the dataset
            state = env.reset(env_steps_size=training_batch_size)
            
            states, actions, rewards, predictions, dones, next_states = [], [], [], [], [], []
            
            for step in batch_data:
                action, prediction = agent.act(state)
                next_state, reward, done = env.step(action)
                
                states.append(tf.expand_dims(state, axis=0)) 
                next_states.append(tf.expand_dims(next_state, axis=0))
                
                action_onehot = np.zeros(3)
                action_onehot[action] = 1
                actions.append(action_onehot)
                rewards.append(reward)
                dones.append(done)
                predictions.append(prediction)
                state = next_state

            a_loss, c_loss = agent.replay(states, actions, rewards, predictions, dones, next_states)
            total_average.append(env.net_worth)
            average = np.average(total_average)

            # Log to TensorBoard
            agent.writer.add_scalar('Data/episode net_worth', env.net_worth, episode)
            agent.writer.add_scalar('Data/average net_worth', average, episode)
            agent.writer.add_scalar('Data/episode_orders', env.episode_orders, episode)
            
            print(f"episode: {episode} net worth: {env.net_worth} average: {average} orders: {env.episode_orders}")
            
            if episode > len(total_average) and best_average < average:
                best_average = average
                agent.save(score="{:.2f}".format(best_average), args=[episode, average, env.episode_orders, a_loss, c_loss])

            agent.save(score="latest")


def test_agent(test_df, test_df_original, folder="", name="", comment=""):
    with open(folder + "/Parameters.json", "r") as json_file:
        params = json.load(json_file)
    params["Actor name"] = f"{name}_Actor.h5"
    params["Critic name"] = f"{name}_Critic.h5"

    agent = CustomAgent(lookback_window_size=params["lookback window size"], optimizer=Adam, depth=params["depth"], comment=comment)

    env = CustomEnv(df=test_df, df_original=test_df_original, lookback_window_size=params["lookback window size"])

    agent.create_writer(env.initial_balance)
    agent.load(folder, name)
    state = env.reset()

    while True:
        # env.render(visualize)
        action, prediction = agent.act(state)
        state, reward, done = env.step(action)
        if not env.current_step % 1000:
            print("step: {:<5}, net_worth: {:<7.2f}, orders: {}".format(env.current_step, env.net_worth,
                                                                        env.episode_orders))
        if env.current_step == env.end_step - 1:
            break

    # save test results to test_results.txt file
    trades_text = ""
    for i, trades in enumerate(env.trades):
        trades_text += "Timestamp: {}  \nType: {}  \nCurrent price: {}  \n{action}: {}  \n\n".format(
            trades["Timestamp"], trades["Type"], trades["Current price"], trades["Total"],
            action="Bought" if trades["Type"] == "Buy" else "Sold")
    agent.writer.add_text("Trades", trades_text, 0)

    with open(f"test_results_{folder}", "a+") as results:
        current_date = datetime.now().strftime('%Y-%m-%d %H:%M')
        results.write(f'date: {current_date}')
        results.write(
            f', net worth:{env.net_worth}, orders per episode:{env.episode_orders}')

def create_tf_dataset(df):
    dataset = tf.data.Dataset.from_tensor_slices(dict(df))
    return dataset

def preprocess_dataset(dataset, batch_size, shuffle_buffer_size):
    # Shuffle the dataset
    dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
    
    # Batch the dataset
    dataset = dataset.batch(batch_size)
    
    # Prefetch for better performance
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    
    return dataset

if __name__ == "__main__":
    # Check if GPU is available
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        print("GPUs are available.")
        print(f"Number of GPUs available: {len(gpus)}")
    else:
        print("No GPU available. Using CPU.")
        
    pd.set_option('display.max_columns', 100)
    pd.set_option('display.width', 1000)

    ########## TRAIN ##########
    train_df_1 = pd.read_csv('./btc_1h_data_training.csv')

    train_df_1 = train_df_1.rename(columns={'time': 'Timestamp', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close',
                            'volume': 'Volume'})
    train_df_original = train_df_1.copy()
    print(train_df_original.head(5))
    print(train_df_1.head(5))

    
    train_df_1 = AddIndicators(train_df_1)  # insert indicators
    
    train_df_1 = train_df_1[100:] # remove first 100 columns for indicators calc
    
    train_df_normalized = Normalizing(train_df_1).dropna() # normalize values
    
    train_df_original = train_df_original.sort_values('Timestamp')
    train_df_normalized = train_df_normalized.sort_values('Timestamp')
    train_df_original = train_df_original[1:] # remove nan from normalizing
    train_df_normalized = train_df_normalized[1:]  # remove nan from normalizing

    train_dataset = create_tf_dataset(train_df_normalized)
    train_dataset = preprocess_dataset(train_dataset, batch_size=64, shuffle_buffer_size=1000)      
    
    depth = len(list(train_df_normalized.columns[1:]))
    lookback_window_size = 100
    
    agent = CustomAgent(lookback_window_size=lookback_window_size, lr=0.00001, epochs=5, optimizer=Adam, batch_size=64,
                        depth=depth, comment="training with last saved and ")
    
    train_env = CustomEnv(df=train_df_normalized, df_original=train_df_original, lookback_window_size=lookback_window_size)
    train_agent(env=train_env, agent=agent, train_dataset=train_dataset, train_episodes=4000)

    ########## TEST ##########
    # test_df_original = pd.read_csv('./BTCUSDT_cycle3.csv')
    # test_df_original = test_df_original.rename(
    #     columns={'time': 'Timestamp', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close',
    #              'volume': 'Volume'})

    # test_df = AddIndicators(test_df_original)  # insert indicators
    # test_df = test_df[100:] # remove invalid indicators value
    # test_df_original = test_df_original[100:]
    # test_df_original = test_df_original[test_df_original[:] != 0] # remove 0 to prevent math error from logging

    # test_df = Normalizing(test_df).dropna()
    # test_df_original = test_df_original[1:] # remove nan from normalizing
    # test_df = test_df[1:]

    # test_agent(test_df, test_df_original, folder="2021_09_10_16_15", name="latest", comment="2021_09_10_16_15")

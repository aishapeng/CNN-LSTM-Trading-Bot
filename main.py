import random
from tqdm import tqdm
from collections import deque
from agent import CustomAgent
from tensorflow.keras.optimizers import Adam
from utils import TradingGraph, Normalizing
from datetime import datetime
from indicators import *
import json


class CustomEnv:
    # A custom Bitcoin trading environment
    def __init__(self, df, initial_balance=1000, lookback_window_size=50, Render_range=100,
                 Show_reward=False, Show_indicators=False, normalize_value=1):  # 40000
        # Define action space and state size and other custom parameters
        self.df = df.reset_index(drop=True)
        self.df_total_steps = len(self.df) - 1
        self.initial_balance = initial_balance
        self.lookback_window_size = lookback_window_size
        self.Render_range = Render_range  # render range in visualization
        self.Show_reward = Show_reward  # show order reward in rendered visualization
        self.Show_indicators = Show_indicators  # show main indicators in rendered visualization

        # Orders history contains the balance, net_worth, crypto_bought, crypto_sold, crypto_held values for the last lookback_window_size steps
        self.orders_history = deque(maxlen=self.lookback_window_size)

        # Market history contains the OHCL values for the last lookback_window_size prices
        self.market_history = deque(maxlen=self.lookback_window_size)

        self.normalize_value = normalize_value

        self.fees = 0.001  # default Binance 0.1% order fees

        self.columns = list(self.df.columns[1:])

    # Reset the state of the environment to an initial state
    def reset(self, visualization=False, env_steps_size=0):
        # if visualization:
        #     self.visualization = TradingGraph(Render_range=self.Render_range, Show_reward=self.Show_reward,
        #                                       Show_indicators=self.Show_indicators)  # init visualization

        self.trades = deque(maxlen=self.Render_range)  # limited orders memory for visualization

        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.prev_net_worth = self.initial_balance
        self.crypto_held = 0
        self.crypto_sold = 0
        self.crypto_bought = 0
        self.episode_orders = 0  # track episode orders count
        self.prev_episode_orders = 0  # track previous episode orders count
        self.rewards = deque(maxlen=self.Render_range)
        self.env_steps_size = env_steps_size
        self.punish_value = 0
        if env_steps_size > 0:  # used for training dataset
            self.start_step = random.randint(self.lookback_window_size, self.df_total_steps - env_steps_size)
            self.end_step = self.start_step + env_steps_size
        else:  # used for testing dataset
            self.start_step = self.lookback_window_size
            self.end_step = self.df_total_steps

        self.current_step = self.start_step

        for i in reversed(range(self.lookback_window_size)):
            current_step = self.current_step - i
            self.orders_history.append([self.balance / self.normalize_value,
                                        self.net_worth / self.normalize_value,
                                        self.crypto_bought / self.normalize_value,
                                        self.crypto_sold / self.normalize_value,
                                        self.crypto_held / self.normalize_value
                                        ])

            # one line for loop to fill market history withing reset call
            self.market_history.append([self.df.loc[current_step, column] for column in self.columns])

        # state = np.concatenate((self.orders_history, self.market_history), axis=1)
        state = np.array(self.market_history)

        return state

    # Get the data points for the given current_step
    def next_observation(self):
        self.market_history.append([self.df.loc[self.current_step, column] for column in self.columns])
        # obs = np.concatenate((self.orders_history, self.market_history), axis=1)
        obs = np.array(self.market_history)

        return obs

    # Execute one time step within the environment
    def step(self, action):
        self.crypto_bought = 0
        self.crypto_sold = 0
        self.current_step += 1

        current_price = self.df.loc[self.current_step, 'Open']
        Timestamp = self.df.loc[self.current_step, 'Timestamp']  # for visualization
        High = self.df.loc[self.current_step, 'High']  # for visualization
        Low = self.df.loc[self.current_step, 'Low']  # for visualization

        if action == 0:  # Hold
            pass

        elif action == 1 and self.balance > self.initial_balance * 0.05:
            # Buy with 100% of current balance
            self.crypto_bought = self.balance / current_price
            self.crypto_bought *= (1 - self.fees)  # substract fees
            self.balance -= self.crypto_bought * current_price
            self.crypto_held += self.crypto_bought
            self.trades.append({'Timestamp': Timestamp, 'High': High, 'Low': Low, 'total': self.crypto_bought, 'type': "buy",
                                'current_price': current_price})
            self.episode_orders += 1

        elif action == 2 and self.crypto_held * current_price > self.initial_balance * 0.05:
            # Sell 100% of current crypto held
            self.crypto_sold = self.crypto_held
            self.crypto_sold *= (1 - self.fees)  # substract fees
            self.balance += self.crypto_sold * current_price
            self.crypto_held -= self.crypto_sold
            self.trades.append({'Timestamp': Timestamp, 'High': High, 'Low': Low, 'total': self.crypto_sold, 'type': "sell",
                                'current_price': current_price})
            self.episode_orders += 1

        self.prev_net_worth = self.net_worth
        self.net_worth = self.balance + self.crypto_held * current_price

        self.orders_history.append([self.balance / self.normalize_value,
                                    self.net_worth / self.normalize_value,
                                    self.crypto_bought / self.normalize_value,
                                    self.crypto_sold / self.normalize_value,
                                    self.crypto_held / self.normalize_value
                                    ])
        # Calculate reward
        reward = self.net_worth - self.prev_net_worth
        # Receive calculated reward
        # reward = self.get_reward()

        if self.net_worth <= self.initial_balance / 2:
            done = True
        else:
            done = False

        obs = self.next_observation()

        return obs, reward, done

    # Calculate reward
    def get_reward(self):
        if self.episode_orders > 1 and self.episode_orders > self.prev_episode_orders:
            self.prev_episode_orders = self.episode_orders
            if self.trades[-1]['type'] == "buy" and self.trades[-2]['type'] == "sell":
                reward = self.trades[-2]['total'] * self.trades[-2]['current_price'] - self.trades[-2]['total'] * \
                         self.trades[-1]['current_price']
                self.trades[-1]["Reward"] = reward
                return reward
            elif self.trades[-1]['type'] == "sell" and self.trades[-2]['type'] == "buy":
                reward = self.trades[-1]['total'] * self.trades[-1]['current_price'] - self.trades[-2]['total'] * \
                         self.trades[-2]['current_price']
                self.trades[-1]["Reward"] = reward
                return reward
        else:
            # return 0
            return self.punish_value

    # render environment
    def render(self, visualize=False):
        # print(f'Step: {self.current_step}, Net Worth: {self.net_worth}')
        if visualize:
            # Render the environment to the screen
            img = self.visualization.render(self.df.loc[self.current_step], self.net_worth, self.trades)
            return img


def train_agent(env, agent, visualize=False, train_episodes=50, training_batch_size=500):
    agent.create_writer(env.initial_balance, env.normalize_value, train_episodes, training_batch_size)  # create TensorBoard writer
    total_average = deque(maxlen=20)  # save recent 20 episodes net worth
    best_average = 0  # used to track best average net worth
    for episode in tqdm(range(1, train_episodes + 1), ascii=True, unit='episodes'):
        state = env.reset(env_steps_size=training_batch_size)

        states, actions, rewards, predictions, dones, next_states = [], [], [], [], [], []
        for t in range(training_batch_size):
            # env.render(visualize)
            action, prediction = agent.act(state)
            next_state, reward, done = env.step(action)
            # states.append(state)
            # next_states.append(next_state)
            states.append(np.expand_dims(state, axis=0))
            next_states.append(np.expand_dims(next_state, axis=0))
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

        agent.writer.add_scalar('Data/episode net_worth', env.net_worth, episode)
        agent.writer.add_scalar('Data/average net_worth', average, episode)
        agent.writer.add_scalar('Data/episode_orders', env.episode_orders, episode)

        print("episode: {:<5} net worth {:<7.2f} average: {:<7.2f} orders: {}".format(episode, env.net_worth, average,
                                                                                      env.episode_orders))
        print("Pred:", prediction)

        if episode > len(total_average):
            if best_average < average:
                best_average = average
                print("Saving model")
                agent.save(score="{:.2f}".format(best_average),
                           args=[episode, average, env.episode_orders, a_loss, c_loss])
            agent.save(score="latest")

            trades_text = ""
            for i, trades in enumerate(env.trades):
                trades_text += "Timestamp: {}  \nType: {}  \nCurrent price: {}  \n{action}: {}  \n\n".format(
                    trades["Timestamp"], trades["type"], trades["current_price"], trades["total"],
                    action="Bought" if trades["type"] == "buy" else "Sold")
            agent.writer.add_text("Trades", trades_text, episode)


def test_agent(test_df, visualize=False, folder="", name="", comment="",
               Show_reward=False, Show_indicators=False):
    with open(folder + "/Parameters.json", "r") as json_file:
        params = json.load(json_file)
    # if name != "":
    params["Actor name"] = f"{name}_Actor.h5"
    params["Critic name"] = f"{name}_Critic.h5"
    # name = params["Actor name"][:-9]

    agent = CustomAgent(lookback_window_size=params["lookback window size"], optimizer=Adam, depth=params["depth"])

    env = CustomEnv(df=test_df, lookback_window_size=params["lookback window size"],
                    Show_reward=Show_reward, Show_indicators=Show_indicators)

    agent.load(folder, name)
    # average_net_worth = 0
    # average_orders = 0
    # no_profit_episodes = 0
    # for episode in tqdm(range(1, test_episodes + 1), ascii=True, unit='episodes'):
    state = env.reset()

    while True:
        # env.render(visualize)
        action, prediction = agent.act(state)
        state, reward, done = env.step(action)
        if not env.current_step % 100:
            print("step: {:<5}, net_worth: {:<7.2f}, orders: {}".format(env.current_step, env.net_worth,
                                                                        env.episode_orders))
        if env.current_step == env.end_step - 1:
            # average_net_worth += env.net_worth
            # average_orders += env.episode_orders
            # if env.net_worth < env.initial_balance:
            #     no_profit_episodes += 1  # calculate episode count where we had negative profit through episode

            break

    # print("average {} episodes agent net_worth: {}, orders: {}".format(test_episodes, average_net_worth / test_episodes,
    #                                                                    average_orders / test_episodes))
    # print("No profit episodes: {}".format(no_profit_episodes))
    # save test results to test_results.txt file
    with open(f"test_results_{folder}", "a+") as results:
        current_date = datetime.now().strftime('%Y-%m-%d %H:%M')
        results.write(f'date: {current_date}')
        results.write(
            f', net worth:{env.net_worth}, orders per episode:{env.episode_orders}')
        # results.write(f', no profit episodes:{no_profit_episodes}, comment: {comment}\n')


if __name__ == "__main__":
    pd.set_option('display.max_columns', 100)
    pd.set_option('display.width', 1000)

    ########## TRAIN ##########
    train_df = pd.read_csv('./BTCUSDT_cycle1.csv')
    train_df = train_df.rename(columns={'time': 'Timestamp', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close',
                           'volume': 'Volume'})
    train_df = train_df.sort_values('Timestamp')

    train_df = AddIndicators(train_df)  # insert indicators
    train_df = train_df [100:]

    depth = len(list(train_df.columns[1:]))  # OHCL + indicators without Date
    lookback_window_size = 100

    train_df = Normalizing(train_df).dropna()
    agent = CustomAgent(lookback_window_size=lookback_window_size, lr=0.00001, epochs=3, optimizer=Adam, batch_size=32,
                        depth=depth, comment="removed seasonality")

    # train_env = CustomEnv(df=train_df, lookback_window_size=lookback_window_size)
    # train_agent(train_env, agent, visualize=False, train_episodes=1000, training_batch_size=500)

    ########## TEST ##########
    test_df = pd.read_csv('./BTCUSDT_cycle1.csv')
    test_df = test_df.rename(
        columns={'time': 'Timestamp', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close',
                 'volume': 'Volume'})
    test_df = test_df.sort_values('Timestamp')

    test_df = AddIndicators(test_df)  # insert indicators
    test_df = test_df[100:]

    test_df = Normalizing(test_df).dropna()

    # test_env = CustomEnv(test_df, lookback_window_size)

    test_agent(test_df, folder="2021_09_04_15_25", name="", comment="")

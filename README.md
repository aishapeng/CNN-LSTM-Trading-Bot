# TradingBot

<h2>DRL Automated Trading Bot</h2>

<p>Given the cryptocurrency market domain, the environment is custom-built to adapt to the cryptocurrency market domain but it generally includes functions to allow the trading agent to take steps and actions. With each step, the states are a lookback window of 100 hours consisting of the market information including the technical indicator. Where the actions will be either buy, hold or sell. As the goal of the model training is to increase the agent’s net worth, the reward is calculated by subtracting its previous net worth from its current net worth</p>

<h3>PPO algorithm</h3>
<p>The reason PPO is chosen as the RL algorithm comes down to a few factors. Firstly, it must be a model-free algorithm. This is due to the volatility of the crypto market, where the prices are extremely dynamic, and short-term decline can happen unexpectedly. Implementing the model-free approach allows the agent in this research to rapidly adapt when the environment changes its way of reacting to the agent’s actions. </p>

<h3>CNN-LSTM architecture</h3>
<p>As CNNs are not usually adapted to manage complex and long temporal dependencies, the LSTM network is used in this research to cope with temporal correlations. </p>


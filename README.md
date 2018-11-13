# vipshop
## Data Description
> In the recommendation task, a recommender agent interacts with environment (users) by sequentially recommending items over a sequence of time steps, so as to maximize its cumulative reward.
### Questions to be discussed on Nov 13.
- Does the data format cover everything we plan to work on?
- Is it possible that we can get the recommending list on the product page?
- Is there any answer for the two questions we discussed last time. How is the time limitation decided? How often do the products repeatedly show up? (How often do they have brand new products?)
- How much data do we want? We plan to use the data of 1000 customers for at least one-year length.
### One session example
| Step | Product Page | Recommender Action | User Feedback |
| :--: | :----------: | :----------------: | :-----------: |
| 0 | 1 | 2,3,4 | click 3 |
| 1 | 3 | 5,6,7 | click nothing |
| 2 | 8 | 9,10,11 | click 9 |
| 3 | 9 | 3,5,12 | click 3 |
| 4 | 3 | 6,7,1 | cart 3 |
| 5 | - | - | order 3 |

- The `product page` includes information like product ID, remaining time, name, price, discounts, etc.
- The `recommender Action` means the recommended items by the recommender system. The items are generated based on the history and the expected future reward of the user.
- The `user feedback` means the users' reactions given the recommended items. They can click, cart, order, (maybe return) or do nothing. (I saw there's a variable called order_valid=1 if the order is not rejected or returned.)


### Framework
Install the dependencies and devDependencies and start the server.

```python
# -*- coding: utf-8 -*-
import random
import gym
import numpy as np
from gym import spaces
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

class Customers(gym.Env):
    def __init__(self):
        
        self.action_space = spaces.MultiDiscrete(50) 
        self.state = None
        self.session = []#get from data
    
    def step(self, action,targetitem,behavior):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))

        done = bool(behavior)
        if not done:#done
            reward = -0.1
        else:
            if action == targetitem:
                reward = 10
            else:
                reward = 0
        self.state = targetitem

        return self.state, reward, done, {}
    
    def reset(s):
        self.state = s
        return self.state
        
    def close(self):
        return None

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


if __name__ == "__main__":
    training_data = get_input_data('train')
    testing_data = get_input_data('test')
    env = Customers()
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    # agent.load("./save/cartpole-dqn.h5")
    done = False
    batch_size = 32

    # Offline Training -------------------------------------------------------
    for e in range(len(training_data)):#session numbers
        state = env.reset(s)
        state = np.reshape(state, [1, state_size])
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action,targetitem)
            reward = reward if not done else 10
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print("episode: {}/{}, score: {}, e: {:.2}"
                      .format(e, EPISODES, time, agent.epsilon))
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        # if e % 10 == 0:
        #     agent.save("./save/cartpole-dqn.h5")

    # Online Testing ---------------------------------------------------------
    suc = 0
    fail = 0
    for e in range(len(testing_data)):#session numbers
        env.reset()
        observation, _, done, _ = env.step(None)
        reward = None
        done = None
        while not done:
            action = agent.act(observation, reward, done)
            observation, reward, done, info = env.step(action)

            if reward:
                suc = suc + 1
            else:
                fail = fail + 1

```


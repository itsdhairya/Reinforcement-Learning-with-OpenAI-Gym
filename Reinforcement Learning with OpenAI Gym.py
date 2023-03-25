import gym
import numpy as np

# Create environment
env = gym.make('CartPole-v0')

# Initialize Q-table
Q = np.zeros([env.observation_space.n, env.action_space.n])

# Set hyperparameters
alpha = 0.1
gamma = 0.99
epsilon = 1
epsilon_decay = 0.01

# Train agent
for episode in range(1, 10001):
    state = env.reset()
    done = False
    while not done:
        # Choose action
        if np.random.uniform() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state, :])

        # Take action and observe new state and reward
        new_state, reward, done, info = env.step(action)

        # Update Q-table
        Q[state, action] = (1 - alpha) * Q[state, action] + alpha * (reward + gamma * np.max(Q[new_state, :]))

        # Update state
        state = new_state

    # Decay epsilon
    epsilon *= (1 - epsilon_decay)

    # Print episode results
    print(f'Episode {episode}: Score = {info["score"]}, Epsilon = {epsilon}')

# Test agent
state = env.reset()
done = False
while not done:
    action = np.argmax(Q[state, :])
    state, reward, done, info = env.step(action)
    env.render()
env.close()

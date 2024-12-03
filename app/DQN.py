import random
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)  
        self.fc2 = nn.Linear(128, 64)         
        self.fc3 = nn.Linear(64, action_size) 

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)  

class DQNAgent:
    def __init__(self, state_size, action_size, gamma=0.99, lr=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma  
        self.epsilon = 1.0  
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995

        self.policy_net = DQN(state_size, action_size).to(device)
        self.target_net = DQN(state_size, action_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self.memory = []  
        self.memory_size = 10000
        self.batch_size = 64

    def remember(self, state, action, reward, next_state, done):
        if len(self.memory) >= self.memory_size:
            self.memory.pop(0)  
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() < self.epsilon: 
            return np.random.choice(self.action_size)
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        return torch.argmax(q_values).item() 

    def replay(self):
        if len(self.memory) < self.batch_size:
            return  

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(states, dtype=torch.float32).to(device)
        actions = torch.tensor(actions, dtype=torch.int64).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
        dones = torch.tensor(dones, dtype=torch.float32).to(device)

        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()

        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = self.loss_fn(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def train(env, agent, episodes=1000, target_update_freq=10):
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0

            while True:
                action = agent.act(state)

                next_state, reward, done = env.step(action)
                total_reward += reward

                agent.remember(state, action, reward, next_state, done)

                agent.replay()

                state = next_state

                if done:
                    break

            if episode % target_update_freq == 0:
                agent.update_target_network()

            agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)
            print(f"Episode {episode + 1}/{episodes} | Total Reward: {total_reward}")


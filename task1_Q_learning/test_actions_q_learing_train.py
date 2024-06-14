##导入库
from pathlib import Path
import numpy as np
import random
import copy
import math

from robobo_interface import (
    IRobobo,
    Emotion,
    LedId,
    LedColor,
    SoundEmotion,
    SimulationRobobo,
    HardwareRobobo,
)

from robobo_interface.datatypes import (
    Emotion,
    LedColor,
    LedId,
    Acceleration,
    Position,
    Orientation,
    WheelPosition,
    SoundEmotion,
)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#定义模型
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )

class DQN:
    def __init__(self, state_dim, action_dim, discount=0.99, tau=0.005, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discount = discount
        self.tau = tau
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.q_network = QNetwork(state_dim, action_dim).to(device)
        self.target_q_network = copy.deepcopy(self.q_network)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=3e-4)

    def select_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_dim)
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.q_network(state).argmax().item()

    def train(self, replay_buffer, batch_size=128):
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            target_q = reward + not_done * self.discount * self.target_q_network(next_state).max(1)[0].unsqueeze(1)

        current_q = self.q_network(state).gather(1, action.long())

        loss = F.mse_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_target_network(self):
        for param, target_param in zip(self.q_network.parameters(), self.target_q_network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filename):
        torch.save(self.q_network.state_dict(), filename + "_q_network")
        torch.save(self.optimizer.state_dict(), filename + "_optimizer")

    def load(self, filename):
        self.q_network.load_state_dict(torch.load(filename + "_q_network"))
        self.optimizer.load_state_dict(torch.load(filename + "_optimizer"))
        self.target_q_network = copy.deepcopy(self.q_network)

def get_position(rob):
    return rob.get_position()

def get_state(rob):
    IRS_data = read_irs_data(rob)
    return IRS_data.cpu().numpy().flatten()

def read_irs_data(rob):
    IRS_data = rob.read_irs()
    normalized_data = [normalize_sensor(v) for v in IRS_data]
    return torch.tensor(normalized_data, dtype=torch.float32).unsqueeze(0).to(device)

def normalize_sensor(v):
    if v == 0:
        return 0
    else:
        return 1 / math.log2(v)


def take_action(rob, action):
    if action == 0:
        rob.move_blocking(20, 20, 500)
    elif action == 1:
        rob.move_blocking(-20, -20, 500)
    elif action == 2:
        rob.move_blocking(-10, 20, 500)
    elif action == 3:
        rob.move_blocking(20, -10, 500)
    elif action == 4:
        rob.move_blocking(20, -10, 200)
    elif action == 5:
        rob.move_blocking(20, -10, 200)
    elif action == 6:
        rob.move_blocking(0, 0, 600)


# Training and Evaluation
def train_model(rob, policy, replay_buffer, initial_position, target_position, max_timesteps=1000):
    for t in range(max_timesteps):
        state = get_state(rob)
        action = policy.select_action(np.array(state))
        take_action(rob, action)
        next_state = get_state(rob)
        reward = calculate_reward(rob, target_position)
        done = is_done(rob, target_position)
        replay_buffer.add(state, action, next_state, reward, done)
        policy.train(replay_buffer)
        policy.update_target_network()
        if done:
            rob.set_position(position=initial_position)
            rob.reset()

def calculate_reward(rob, target_position):
    # Read sensor data
    IRS_data = rob.read_irs()
    
    # Define thresholds
    FrontL_threshold = 200
    FrontLL_threshold = 200
    FrontC_threshold = 100
    FrontR_threshold = 200
    FrontRR_threshold = 200
    BackL_threshold = 200  # Back left sensor threshold
    BackR_threshold = 200  # Back right sensor threshold
    BackC_threshold = 200  # Back center sensor threshold
    
    # Initial reward
    reward = 0
    
    # Check if there are obstacles in the front
    if (IRS_data[2] + IRS_data[7] + IRS_data[4]) > (FrontL_threshold + FrontLL_threshold + FrontC_threshold) or (IRS_data[3] + IRS_data[5] + IRS_data[4]) > (FrontR_threshold + FrontRR_threshold + FrontC_threshold):
        max_sensor = IRS_data.index(max(IRS_data[2:6] + [IRS_data[7]]))
        if max_sensor == 4:  # Obstacle in front center
            temp_IRS_data = IRS_data[:]
            temp_IRS_data[4] = float('-inf')
            max_sensor_2 = temp_IRS_data.index(max(temp_IRS_data[2:6] + [temp_IRS_data[7]]))

            if max_sensor_2 == 3 or max_sensor_2 == 5:  # Obstacle on the right
                reward -= 10  # Turn left
            else:  # Obstacle on the left or directly in front, default to turn right
                reward -= 10  # Turn right

        elif max_sensor == 3 or max_sensor == 5:  # Obstacle on the right
            reward -= 5  # Slightly turn left
               
        elif max_sensor == 2 or max_sensor == 7:  # Obstacle on the left
            reward -= 5  # Slightly turn right
    
    # Check if there are obstacles in the back
    if IRS_data[0] > BackL_threshold or IRS_data[1] > BackR_threshold or IRS_data[6] > BackC_threshold:
        reward -= 10  # Penalty for obstacles in the back

    # Collision detection
    if IRS_data[4] > 500:
        reward -= 100  # Collision penalty

    # Add a slight penalty for each time step to encourage faster completion of the task
    reward -= 1

    # Calculate distance reward
    current_position = get_position(rob)
    current_x, current_y = current_position.x, current_position.y
    target_x, target_y = target_position.x, target_position.y
    distance = np.sqrt((current_x - target_x) ** 2 + (current_y - target_y) ** 2)
    
    # The closer the distance, the greater the reward
    reward += max(0, 10 - distance)

    if is_done(rob, target_position):
        reward += 500

    print(reward)
    return reward

def is_done(rob: IRobobo, target_position):
    current_position = get_position(rob)
    print(current_position)
    # Assuming Position object has x and y attributes
    current_x, current_y = current_position.x, current_position.y
    target_x, target_y = target_position.x, target_position.y

    # Calculate the distance to the target position
    distance = np.sqrt((current_x - target_x) ** 2 + (current_y - target_y) ** 2)
    
    distance_threshold = 0.1  # Distance threshold to reach the target
    return distance <= distance_threshold



def eval_policy(rob, policy, initial_position,target_position):
    avg_fitness = 0
    num_evaluations = 10
    for _ in range(num_evaluations):
        rob.play_simulation()
        rob.set_position(position=initial_position)
        state, done = get_state(rob), False
        total_reward = 0
        while not done:
            action = policy.select_action(np.array(state))
            take_action(rob, action)
            state = get_state(rob)
            reward = calculate_reward(rob)
            total_reward += reward
            done = is_done(rob,target_position)
        avg_fitness += total_reward
        print(num_evaluations,avg_fitness)
        rob.stop_simulation()
    avg_fitness /= num_evaluations
    return avg_fitness


def run_all_actions(rob: IRobobo):
    random.seed(43)
    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()
    state_dim = 8
    action_dim = 7
    policy = DQN(state_dim=state_dim, action_dim=action_dim)
    replay_buffer = ReplayBuffer(state_dim=state_dim, action_dim=action_dim)
    
    initial_position = get_position(rob)
    target_position = get_position(rob)
    target_position.x = 0.05072
    target_position.y = 0.7205
    
    num_iterations = 10
    for iteration in range(num_iterations):
        print(f"Iteration: {iteration+1}")
        train_model(rob, policy, replay_buffer, initial_position,target_position, max_timesteps=1000)
        
        avg_fitness = eval_policy(rob, policy, initial_position,target_position)
        print(f"Average Fitness: {avg_fitness}")

        policy.save(f"dqn_policy_{iteration+1}")

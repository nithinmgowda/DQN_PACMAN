import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from env.pacman_env import PacmanEnv
from config import *
import matplotlib.pyplot as plt
from datetime import datetime
import os

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        
        # Calculate input dimensions
        self.height = SCREEN_HEIGHT // CELL_SIZE
        self.width = SCREEN_WIDTH // CELL_SIZE
        
        # Convolutional layers
        self.conv_layers = nn.Sequential()
        prev_channels = 5  # Input channels (Pacman, Ghosts, Food, Power pellets, Walls)
        
        for i, layer_config in enumerate(CONV_LAYERS):
            self.conv_layers.add_module(
                f'conv{i+1}',
                nn.Conv2d(
                    layer_config['in_channels'],
                    layer_config['out_channels'],
                    layer_config['kernel_size'],
                    layer_config['stride'],
                    padding=1  # Add padding to maintain spatial dimensions
                )
            )
            self.conv_layers.add_module(f'relu{i+1}', nn.ReLU())
            
            # Optional: Add pooling layer to reduce spatial dimensions
            if i < len(CONV_LAYERS) - 1:  # Don't pool after last conv layer
                self.conv_layers.add_module(f'pool{i+1}', nn.MaxPool2d(2, 2))
        
        # Calculate the size of flattened features
        self.feature_size = self._get_conv_output_size()
        
        # Fully connected layers
        self.fc_layers = nn.Sequential()
        prev_size = self.feature_size
        
        for i, fc_size in enumerate(FC_LAYERS):
            self.fc_layers.add_module(f'fc{i+1}', nn.Linear(prev_size, fc_size))
            self.fc_layers.add_module(f'relu_fc{i+1}', nn.ReLU())
            prev_size = fc_size
        
        # Output layer
        self.output = nn.Linear(prev_size, OUTPUT_SIZE)
    
    def _get_conv_output_size(self):
        # Create a dummy input to calculate the size of flattened features
        x = torch.zeros(1, 5, self.height, self.width)
        x = self.conv_layers(x)
        return int(np.prod(x.size()))
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return self.output(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

def train():
    env = PacmanEnv(render_mode="human")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    policy_net = DQN().to(device)
    target_net = DQN().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    
    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    memory = ReplayBuffer(MEMORY_SIZE)
    
    epsilon = EPSILON_START
    episode_rewards = []
    
    for episode in range(NUM_EPISODES):
        state, _ = env.reset()
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        episode_reward = 0
        
        while True:
            # Epsilon-greedy action selection
            if random.random() > epsilon:
                with torch.no_grad():
                    action = policy_net(state).max(1)[1].item()
            else:
                action = env.action_space.sample()
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            
            next_state = torch.FloatTensor(next_state).unsqueeze(0).to(device)
            
            # Store transition in memory
            memory.push(state, action, reward, next_state, done)
            
            state = next_state
            
            # Train if enough samples are available
            if len(memory) > BATCH_SIZE:
                transitions = memory.sample(BATCH_SIZE)
                batch = list(zip(*transitions))
                
                state_batch = torch.cat(batch[0])
                action_batch = torch.tensor(batch[1], device=device)
                reward_batch = torch.tensor(batch[2], device=device, dtype=torch.float)
                next_state_batch = torch.cat(batch[3])
                done_batch = torch.tensor(batch[4], device=device, dtype=torch.float)
                
                # Compute Q(s_t, a)
                current_q = policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
                
                # Compute max Q(s_{t+1}, a) for all next states
                next_q = target_net(next_state_batch).max(1)[0].detach()
                target_q = reward_batch + (1 - done_batch) * GAMMA * next_q
                
                # Compute loss
                loss = nn.MSELoss()(current_q.squeeze(), target_q)
                
                # Optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            if done:
                break
        
        # Update target network
        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
        
        # Decay epsilon
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
        
        # Record episode reward
        episode_rewards.append(episode_reward)
        
        # Print progress
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Episode {episode + 1}/{NUM_EPISODES}, Avg Reward: {avg_reward:.2f}, Epsilon: {epsilon:.2f}")
            
            # Save the model periodically
            if (episode + 1) % 100 == 0:
                model_dir = "models"
                os.makedirs(model_dir, exist_ok=True)
                torch.save(policy_net.state_dict(),
                         f"{model_dir}/pacman_dqn_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth")
    
    # Plot training results
    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards)
    plt.title('Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.savefig('training_progress.png')
    plt.close()
    
    env.close()
    return policy_net

if __name__ == "__main__":
    trained_model = train()

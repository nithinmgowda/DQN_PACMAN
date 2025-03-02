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

# Configure matplotlib for interactive plotting
plt.ion()

# Enable interactive plotting
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
    # Create models directory if it doesn't exist
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    os.makedirs(model_dir, exist_ok=True)
    
    env = PacmanEnv(render_mode="human")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    policy_net = DQN().to(device)
    target_net = DQN().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    
    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    memory = ReplayBuffer(MEMORY_SIZE)
    
    epsilon = EPSILON_START
    episode_rewards = []
    moving_averages = []
    
    # Create and configure the plot window
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.title('Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    reward_line, = plt.plot([], [], 'b-', label='Episode Reward')
    avg_line, = plt.plot([], [], 'r-', label='Moving Average (10 episodes)')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    epsilon_line, = plt.plot([], [], 'g-', label='Exploration Rate (ε)')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    def update_plots():
        episodes = list(range(len(episode_rewards)))
        
        # Update reward plot
        reward_line.set_data(episodes, episode_rewards)
        
        # Update moving average
        if len(episode_rewards) >= 10:
            moving_avg = np.convolve(episode_rewards, np.ones(10)/10, mode='valid')
            avg_line.set_data(range(9, len(episode_rewards)), moving_avg)
            moving_averages.append(moving_avg[-1])
        
        # Update epsilon plot
        epsilon_values = [EPSILON_START * (EPSILON_DECAY ** i) for i in range(len(episodes))]
        epsilon_line.set_data(episodes, epsilon_values)
        
        # Update plot limits
        for ax in plt.gcf().axes:
            ax.relim()
            ax.autoscale_view()
        
        # Force update the display
        plt.gcf().canvas.draw()
        plt.gcf().canvas.flush_events()
    
    for episode in range(NUM_EPISODES):
        state, _ = env.reset()
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        episode_reward = 0
        
        # Save model every 5 episodes
        if episode > 0 and episode % 5 == 0:
            model_path = os.path.join(model_dir, f"pacman_dqn_checkpoint_{episode}.pth")
            torch.save({
                'episode': episode,
                'model_state_dict': policy_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, model_path)
            print(f"Saved model checkpoint at episode {episode}")
        
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
        
        # Record episode reward and update plot
        episode_rewards.append(episode_reward)
        update_plots()
        
        # Print progress and save model
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Episode {episode + 1}/{NUM_EPISODES}, Avg Reward: {avg_reward:.2f}, Epsilon: {epsilon:.2f}")
            
            # Save model checkpoint
            checkpoint_path = os.path.join(model_dir, f"pacman_dqn_checkpoint_{episode + 1}.pth")
            torch.save({
                'episode': episode + 1,
                'model_state_dict': policy_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epsilon': epsilon,
                'episode_rewards': episode_rewards,
                'moving_averages': moving_averages
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
    
    # Save final model and plots
    final_model_path = os.path.join(model_dir, "pacman_dqn_final.pth")
    torch.save({
        'episode_count': NUM_EPISODES,
        'model_state_dict': policy_net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'final_epsilon': epsilon,
        'episode_rewards': episode_rewards,
        'moving_averages': moving_averages
    }, final_model_path)
    print(f"\nTraining completed! Final model saved to {final_model_path}")
    
    # Save final plot
    plt.savefig(os.path.join(model_dir, 'training_progress.png'))
    plt.close()
    
    env.close()
    return policy_net

if __name__ == "__main__":
    trained_model = train()

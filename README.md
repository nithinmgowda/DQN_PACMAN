# AI-Powered Pac-Man with Reinforcement Learning

This project implements a Pac-Man game where an AI agent learns to play using Deep Reinforcement Learning techniques. The game is designed to run in Google Colab and uses modern RL algorithms for training.

## Project Structure
- `env/pacman_env.py`: Custom Gymnasium environment for Pac-Man
- `config.py`: Configuration settings for game and training
- `models/`: Directory for storing trained models
- `train.py`: Script for training the RL agent
- `play.py`: Script for running the game with trained agent

## Setup Instructions
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Train the model:
   ```bash
   python train.py
   ```

3. Play with trained agent:
   ```bash
   python play.py
   ```

## Environment Details
- State space: Grid representation of game state
- Action space: Discrete(4) - [UP, DOWN, LEFT, RIGHT]
- Reward structure:
  - +10 for eating food pellets
  - +50 for eating power pellets
  - +200 for eating ghosts
  - -500 for game over
  - -1 per step (to encourage efficient paths)

## Training Parameters
- Algorithm: DQN (Deep Q-Network)
- Neural Network: CNN-based architecture
- Experience Replay: 100,000 transitions
- Learning rate: 0.0001
- Discount factor (gamma): 0.99
- Epsilon decay: 0.995

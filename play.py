import torch
import argparse
from env.pacman_env import PacmanEnv
from train import DQN
import time

def play(model_path):
    # Initialize environment
    env = PacmanEnv(render_mode="human")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = DQN().to(device)
    try:
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint)
        print(f"Successfully loaded model from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    model.eval()
    
    # Play episodes
    while True:
        state, _ = env.reset()
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        total_reward = 0
        done = False
        
        while not done:
            # Get action from model
            with torch.no_grad():
                action = model(state).max(1)[1].item()
            
            # Execute action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            
            # Update state
            state = torch.FloatTensor(next_state).unsqueeze(0).to(device)
            
            # Add delay to make the game viewable
            time.sleep(0.1)
        
        print(f"Game Over! Total Reward: {total_reward}")
        
        # Ask if user wants to play again
        play_again = input("Play again? (y/n): ")
        if play_again.lower() != 'y':
            break
    
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Play Pac-Man with a trained model')
    parser.add_argument('--model', type=str, required=True,
                      help='Path to the trained model file')
    args = parser.parse_args()
    play(args.model)

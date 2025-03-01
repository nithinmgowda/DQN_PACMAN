# Game Configuration
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
CELL_SIZE = 20  # This will give us a 40x30 grid
FPS = 60

# Colors
BLACK = (0, 0, 0)
YELLOW = (255, 255, 0)
BLUE = (0, 0, 255)
WHITE = (255, 255, 255)
RED = (255, 0, 0)

# Game Parameters
GHOST_COUNT = 4
PELLET_REWARD = 10
POWER_PELLET_REWARD = 50
GHOST_REWARD = 200
DEATH_PENALTY = -100
STEP_PENALTY = -1
POWER_PELLET_DURATION = 300  # frames

# Training Parameters
LEARNING_RATE = 0.001  # Increased for faster learning
BATCH_SIZE = 32  # Reduced batch size
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.9  # More aggressive decay
MEMORY_SIZE = 10000  # Reduced memory size
TARGET_UPDATE = 100  # More frequent updates
NUM_EPISODES = 20  # Very short training for testing

# Neural Network Architecture
CONV_LAYERS = [
    {'in_channels': 5, 'out_channels': 16, 'kernel_size': 3, 'stride': 1},
    {'in_channels': 16, 'out_channels': 32, 'kernel_size': 3, 'stride': 1},
    {'in_channels': 32, 'out_channels': 32, 'kernel_size': 3, 'stride': 1}
]
FC_LAYERS = [512, 128]
OUTPUT_SIZE = 4  # number of possible actions

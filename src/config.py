SAMPLE_RATE = 16000

NUM_MELS = 40  # No. of mel filters

WIN_LEN = (int)(SAMPLE_RATE * 0.025)  # Window length
WIN_HOP = (int)(SAMPLE_RATE * 0.01)  # Hop length

NFFT = 512  # No. of FFT points

TOP_DB = 30  # DB value used in VAD

NUM_FRAMES = 150  # 90 % utterances have more than 150 valid frames

# LSTM parameters, from GE2E paper
NUM_HIDDEN_NODES = 768
NUM_LAYERS = 3
NUM_PROJ = 256

# Optimizer parameters, from GE2E paper
SGD_LR = 0.01
SGD_EPOCH_HALF = 30
SGD_LR_GAMMA = 0.5

# Batch size
# We use `TRAIN_SPKS` speakers in each batch,
# with each speaker sampled `TRAIN_UTTS` utterances
TRAIN_SPKS = 4
TRAIN_UTTS = 6

# We use `TEST_SPKS` speakers in each validation batch,
# with each speaker sampled `TEST_UTTS` utterances,
# of which `TEST_ENROLL` utterances used in enrollment,
# and others used in verification.
TEST_SPKS = 4
TEST_UTTS = 6
TEST_ENROLL = 2

# Epoch
NUM_EPOCH = 10

# Checkpoint path
CKPT_PATH = "net.pt"
# To continue training from a checkpoint,
# Set `RESUME` to True
RESUME = False

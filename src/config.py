# this file stores all hyperparameters and file paths used across the project

class Config:
    # video settings
    NUM_FRAMES = 16
    IMAGE_SIZE = (224, 224)

    # model architecture
    EMBED_DIM = 256
    NUM_HEADS = 4
    FF_DIM = 512
    NUM_LAYERS = 4
    VOCAB_SIZE = 10000
    MAX_SEQ_LEN = 40

    # training config
    BATCH_SIZE = 8
    EPOCHS = 20

    # data paths
    DATA_PATH = "data/raw_videos/" # folder with folders of .jpg frames
    CAPTION_CSV = "data/captions.csv" # CSV file: video_id, caption
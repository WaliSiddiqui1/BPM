class Config:
    NUM_FRAMES = 16
    IMAGE_SIZE = (224, 224)
    BATCH_SIZE = 8
    EPOCHS = 20
    EMBED_DIM = 256
    NUM_HEADS = 4
    FF_DIM = 512
    NUM_LAYERS = 4
    VOCAB_SIZE = 10000  # adjust based on tokenizer
    MAX_SEQ_LEN = 40

    DATA_PATH = "data/raw_videos/"
    CAPTION_CSV = "data/captions.csv"
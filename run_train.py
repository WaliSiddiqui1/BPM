# this is the main launcher script for training the model
# this needs to be called inside the Oscar `.sh` job

from src.train import train_model

if __name__ == "__main__":
    train_model()
import tensorflow as tf
from src.config import Config
from src.model import VideoCaptionModel
from src.utils import create_tokenizer, tokenize_captions
from src.data_loader import load_dataset

# entry point to train the full video captioning model
def train_model():
    # load and preprocess the dataset
    data = load_dataset(Config.DATA_PATH, Config.CAPTION_CSV)
    frames, captions = zip(*data) # separate frame tensors and text

    # tokenize captions
    tokenizer = create_tokenizer(captions, Config.VOCAB_SIZE)
    cap_tokens = tokenize_captions(tokenizer, captions, Config.MAX_SEQ_LEN)

    # wrap into tf.data.Dataset pipeline
    dataset = tf.data.Dataset.from_tensor_slices((np.array(frames), cap_tokens))
    dataset = dataset.shuffle(500).batch(Config.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # initialize the model
    model = VideoCaptionModel(Config)
    optimizer = tf.keras.optimizers.Adam()
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # loop through epochs
    for epoch in range(Config.EPOCHS):
        print(f"Epoch {epoch+1}")
        for step, (vids, caps) in enumerate(dataset):
            with tf.GradientTape() as tape:
                # shift caption tokens for teacher forcing
                input_caps = caps[:, :-1]
                output_caps = caps[:, 1:]

                # run the model and compute the loss
                preds = model(vids, input_caps)
                loss = loss_fn(output_caps, preds)
            
            # update the weights
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            if step % 10 == 0:
                print(f"Step {step}: Loss = {loss.numpy():.4f}")
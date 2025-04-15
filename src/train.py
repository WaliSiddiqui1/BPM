import tensorflow as tf
from src.config import Config
from src.model import VideoCaptionModel
from src.utils import create_tokenizer, tokenize_captions
from src.data_loader import load_dataset

def train_model():
    data = load_dataset(Config.DATA_PATH, Config.CAPTION_CSV)
    frames, captions = zip(*data)

    tokenizer = create_tokenizer(captions, Config.VOCAB_SIZE)
    cap_tokens = tokenize_captions(tokenizer, captions, Config.MAX_SEQ_LEN)

    dataset = tf.data.Dataset.from_tensor_slices((np.array(frames), cap_tokens))
    dataset = dataset.shuffle(500).batch(Config.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    model = VideoCaptionModel(Config)
    optimizer = tf.keras.optimizers.Adam()
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    for epoch in range(Config.EPOCHS):
        print(f"Epoch {epoch+1}")
        for step, (vids, caps) in enumerate(dataset):
            with tf.GradientTape() as tape:
                input_caps = caps[:, :-1]
                output_caps = caps[:, 1:]
                preds = model(vids, input_caps)
                loss = loss_fn(output_caps, preds)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            if step % 10 == 0:
                print(f"Step {step}: Loss = {loss.numpy():.4f}")
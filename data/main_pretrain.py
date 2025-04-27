from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import random
import os
import time
import json
import argparse
from modules.tokenization import BertTokenizer
from modules.modeling import UniVLTF
from modules.optimization import create_optimizer
from dataloaders.dataloader_tf import create_pretrain_dataset
from util import get_logger

# Global logger
logger = None

def get_args(description='UniVL on Pretrain with TensorFlow'):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--do_pretrain", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")

    parser.add_argument('--train_csv', type=str, default='data/HowTo100M_v1.csv', help='train csv')
    parser.add_argument('--features_path', type=str, default='feature', help='feature path for 2D features')
    parser.add_argument('--data_path', type=str, default='data/data.pickle', help='data json file path')

    parser.add_argument('--num_thread_reader', type=int, default=1, help='')
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
    parser.add_argument('--epochs', type=int, default=20, help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--batch_size_val', type=int, default=3500, help='batch size eval')
    parser.add_argument('--lr_decay', type=float, default=0.9, help='Learning rate exp epoch decay')
    parser.add_argument('--n_display', type=int, default=100, help='Information display frequency')
    parser.add_argument('--video_dim', type=int, default=1024, help='video feature dimension')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--max_words', type=int, default=20, help='')
    parser.add_argument('--max_frames', type=int, default=100, help='')
    parser.add_argument('--min_words', type=int, default=0, help='')
    parser.add_argument('--feature_framerate', type=int, default=1, help='')
    parser.add_argument('--min_time', type=float, default=5.0, help='Gather small clips')
    parser.add_argument('--margin', type=float, default=0.1, help='margin for loss')
    parser.add_argument('--hard_negative_rate', type=float, default=0.5, help='rate of intra negative sample')
    parser.add_argument('--negative_weighting', type=int, default=1, help='Weight the loss for intra negative')
    parser.add_argument('--n_pair', type=int, default=1, help='Num of pair to output from data loader')

    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--bert_model", default="bert-base-uncased", type=str, required=True,
                        help="Bert pre-trained model")
    parser.add_argument("--visual_model", default="visual-base", type=str, required=False, help="Visual module")
    parser.add_argument("--cross_model", default="cross-base", type=str, required=False, help="Cross module")
    parser.add_argument("--decoder_model", default="decoder-base", type=str, required=False, help="Decoder module")
    parser.add_argument("--init_model", default=None, type=str, required=False, help="Initial model.")
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")

    parser.add_argument('--mixed_precision', action='store_true',
                        help="Whether to use mixed precision")
                        
    parser.add_argument('--use_tpu', action='store_true', help="Whether to use TPU or GPU/CPU.")
    parser.add_argument('--tpu_name', type=str, default=None, help="TPU name")
    parser.add_argument('--tpu_zone', type=str, default=None, help="TPU zone")
    parser.add_argument('--gcp_project', type=str, default=None, help="GCP project with TPU")

    parser.add_argument('--coef_lr', type=float, default=0.1, help='coefficient for bert branch.')
    parser.add_argument('--use_mil', action='store_true', help="Whether use MIL as Miech et. al. (2020).")
    parser.add_argument('--sampled_use_mil', action='store_true', help="Whether use MIL, has a high priority than use_mil.")

    parser.add_argument('--text_num_hidden_layers', type=int, default=12, help="Layer NO. of text.")
    parser.add_argument('--visual_num_hidden_layers', type=int, default=6, help="Layer NO. of visual.")
    parser.add_argument('--cross_num_hidden_layers', type=int, default=2, help="Layer NO. of cross.")
    parser.add_argument('--decoder_num_hidden_layers', type=int, default=3, help="Layer NO. of decoder.")

    parser.add_argument('--stage_two', action='store_true', help="Whether training with decoder.")
    parser.add_argument('--pretrain_enhance_vmodal', action='store_true', help="Enhance visual and other modalities when pretraining.")

    parser.add_argument("--load_checkpoint", action="store_true")
    parser.add_argument("--checkpoint_prefix", default="tf_model", type=str, 
                        help="Checkpoint prefix")

    args = parser.parse_args()

    if args.sampled_use_mil:  # sample from each video, has a higher priority than use_mil.
        args.use_mil = True

    # Check parameters
    if args.gradient_accumulation_steps < 1:
        raise ValueError(
            "Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(args.gradient_accumulation_steps))
    if not args.do_pretrain:
        raise ValueError("`do_pretrain` must be True.")

    args.global_batch_size = args.batch_size
    args.batch_size = int(args.batch_size / args.gradient_accumulation_steps)

    return args

def set_seed_logger(args):
    global logger
    # Set seed
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    logger = get_logger(os.path.join(args.output_dir, "log.txt"))

    logger.info("Effective parameters:")
    for key in sorted(args.__dict__):
        logger.info("  <<< {}: {}".format(key, args.__dict__[key]))

    return args

def init_device(args):
    global logger

    if args.use_tpu:
        # TPU setup
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
            tpu=args.tpu_name, zone=args.tpu_zone, project=args.gcp_project)
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        strategy = tf.distribute.TPUStrategy(resolver)
        logger.info("Running on TPU: {}".format(args.tpu_name))
    else:
        # GPU setup
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            try:
                for gpu in physical_devices:
                    tf.config.experimental.set_memory_growth(gpu, True)
                strategy = tf.distribute.MirroredStrategy()
                logger.info("Running on {} GPUs".format(strategy.num_replicas_in_sync))
            except:
                strategy = tf.distribute.get_strategy()
                logger.info("Running on single GPU")
        else:
            strategy = tf.distribute.get_strategy()
            logger.info("Running on CPU")

    # Mixed precision
    if args.mixed_precision:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        logger.info("Using mixed precision")

    return strategy

def init_model(args, strategy):
    with strategy.scope():
        model = UniVLTF(
            bert_model_name=args.bert_model,
            visual_model_name=args.visual_model,
            cross_model_name=args.cross_model,
            decoder_model_name=args.decoder_model,
            text_num_hidden_layers=args.text_num_hidden_layers,
            visual_num_hidden_layers=args.visual_num_hidden_layers,
            cross_num_hidden_layers=args.cross_num_hidden_layers,
            decoder_num_hidden_layers=args.decoder_num_hidden_layers,
            stage_two=args.stage_two,
            use_mil=args.use_mil,
            do_lower_case=args.do_lower_case,
            margin=args.margin,
            hard_negative_rate=args.hard_negative_rate,
            negative_weighting=args.negative_weighting
        )
    
    return model

def prep_optimizer(args, num_train_steps, strategy):
    with strategy.scope():
        optimizer, lr_schedule = create_optimizer(
            init_lr=args.lr,
            num_train_steps=num_train_steps,
            num_warmup_steps=int(num_train_steps * args.warmup_proportion),
            coef_lr=args.coef_lr
        )
    
    return optimizer, lr_schedule

def train_step(model, batch, optimizer, args, training=True):
    # Unpack batch
    input_ids, input_mask, segment_ids, video, video_mask, \
    pairs_masked_text, pairs_token_labels, masked_video, video_labels_index, \
    pairs_input_caption_ids, pairs_decoder_mask, pairs_output_caption_ids = batch
    
    with tf.GradientTape() as tape:
        loss = model(
            input_ids, 
            segment_ids, 
            input_mask, 
            video, 
            video_mask,
            pairs_masked_text=pairs_masked_text, 
            pairs_token_labels=pairs_token_labels,
            masked_video=masked_video, 
            video_labels_index=video_labels_index,
            input_caption_ids=pairs_input_caption_ids, 
            decoder_mask=pairs_decoder_mask,
            output_caption_ids=pairs_output_caption_ids, 
            training=training
        )
        
        if args.mixed_precision:
            scaled_loss = optimizer.get_scaled_loss(loss)
    
    if training:
        if args.mixed_precision:
            scaled_gradients = tape.gradient(scaled_loss, model.trainable_variables)
            gradients = optimizer.get_unscaled_gradients(scaled_gradients)
        else:
            gradients = tape.gradient(loss, model.trainable_variables)
        
        # Gradient clipping
        gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return loss

def train_epoch(epoch, args, model, train_dataset, optimizer, checkpoint_manager, global_step, strategy):
    global logger
    tf.keras.backend.clear_session()
    
    # Setup metrics
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    
    # Start time for logging
    start_time = time.time()
    log_step = args.n_display
    
    # Training loop
    for step, batch in enumerate(train_dataset):
        # Distribute batch
        loss = strategy.run(train_step, args=(model, batch, optimizer, args, True))
        mean_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, loss, axis=None)
        
        # Update metrics
        train_loss.update_state(mean_loss)
        
        global_step += 1
        
        # Logging
        if global_step % log_step == 0:
            current_loss = train_loss.result().numpy()
            logger.info("Epoch: %d/%s, Step: %d, Loss: %f, Time/step: %f", 
                        epoch + 1, args.epochs, step + 1, current_loss,
                        (time.time() - start_time) / log_step)
            start_time = time.time()
            train_loss.reset_states()
            
            # Save checkpoint
            checkpoint_manager.save()
    
    return train_loss.result().numpy(), global_step

def main():
    global logger
    args = get_args()
    args = set_seed_logger(args)
    strategy = init_device(args)
    
    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    
    # Create dataset
    with strategy.scope():
        train_dataset = create_pretrain_dataset(
            args=args,
            tokenizer=tokenizer,
            _strategy=strategy
        )
    
    # Calculate training steps
    num_train_examples = args.batch_size * len(train_dataset)
    num_train_steps = (num_train_examples // args.global_batch_size) * args.epochs
    
    # Initialize model
    model = init_model(args, strategy)
    
    # Initialize optimizer
    optimizer, lr_schedule = prep_optimizer(args, num_train_steps, strategy)
    
    # Setup checkpointing
    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer, step=tf.Variable(0))
    checkpoint_prefix = os.path.join(args.output_dir, args.checkpoint_prefix)
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint, checkpoint_prefix, max_to_keep=5)
    
    # Load checkpoint if specified
    global_step = 0
    if args.load_checkpoint:
        latest_checkpoint = checkpoint_manager.latest_checkpoint
        if latest_checkpoint:
            checkpoint.restore(latest_checkpoint)
            global_step = int(checkpoint.step)
            logger.info("Restored from checkpoint: {}".format(latest_checkpoint))
            logger.info("Global step set to: {}".format(global_step))
        else:
            logger.info("No checkpoint found, starting from scratch")
    
    logger.info("***** Running pretraining *****")
    logger.info("  Num examples = %d", num_train_examples)
    logger.info("  Batch size = %d", args.batch_size)
    logger.info("  Num steps = %d", num_train_steps)
    
    # Training loop
    for epoch in range(args.epochs):
        logger.info("Starting epoch %d", epoch + 1)
        
        # Train for one epoch
        tr_loss, global_step = train_epoch(
            epoch, args, model, train_dataset, optimizer, 
            checkpoint_manager, global_step, strategy
        )
        
        logger.info("Epoch %d/%s Finished, Train Loss: %f", 
                   epoch + 1, args.epochs, tr_loss)
        
        # Save model after each epoch
        checkpoint.step.assign(global_step)
        checkpoint_manager.save()
        
        # Save the model weights directly for easy loading
        model_path = os.path.join(args.output_dir, f"model_epoch_{epoch+1}")
        model.save_weights(model_path)
        logger.info(f"Model weights saved to {model_path}")

if __name__ == "__main__":
    main()

#-------------------------------------------------------

import tensorflow as tf
from tensorflow import keras
from transformers import TFBertModel, BertConfig
import numpy as np

class UniVLTF(keras.Model):
    def __init__(self, 
                bert_model_name, 
                visual_model_name,
                cross_model_name,
                decoder_model_name,
                text_num_hidden_layers=12,
                visual_num_hidden_layers=6,
                cross_num_hidden_layers=2,
                decoder_num_hidden_layers=3,
                stage_two=False,
                use_mil=False,
                do_lower_case=True,
                margin=0.1,
                hard_negative_rate=0.5,
                negative_weighting=1.0
                ):
        super(UniVLTF, self).__init__()
        
        self.stage_two = stage_two
        self._stage_one = not stage_two
        self.use_mil = use_mil
        self.margin = margin
        self.hard_negative_rate = hard_negative_rate
        self.negative_weighting = negative_weighting
        
        # Initialize the BERT model for text
        self.bert_config = BertConfig.from_pretrained(bert_model_name)
        self.bert_config.num_hidden_layers = text_num_hidden_layers
        self.bert = TFBertModel.from_pretrained(bert_model_name, config=self.bert_config)
        
        # Visual Encoder (adapted from the visual transformer)
        self.visual_embeddings = keras.layers.Dense(self.bert_config.hidden_size, activation=None)
        
        # Visual Transformer
        visual_config = BertConfig.from_pretrained(bert_model_name)
        visual_config.num_hidden_layers = visual_num_hidden_layers
        self.visual_encoder = TFBertModel(config=visual_config)
        
        # Cross-Modal Transformer
        cross_config = BertConfig.from_pretrained(bert_model_name)
        cross_config.num_hidden_layers = cross_num_hidden_layers
        self.cross_encoder = TFBertModel(config=cross_config)
        
        # Decoder for captioning
        if self.stage_two:
            decoder_config = BertConfig.from_pretrained(bert_model_name)
            decoder_config.num_hidden_layers = decoder_num_hidden_layers
            self.decoder = TFBertModel(config=decoder_config)
            self.decoder_classifier = keras.layers.Dense(self.bert_config.vocab_size, activation=None)
            
        # MLM and MFM (Masked Language/Frame Modeling) heads
        self.mlm_classifier = keras.layers.Dense(self.bert_config.vocab_size, activation=None)
        self.mfm_classifier = keras.layers.Dense(self.bert_config.hidden_size, activation=None)
        
        # Loss functions
        self.loss_fct = keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
        self.mse_loss = keras.losses.MeanSquaredError(reduction='none')
        
    def call(self, 
             input_ids, 
             segment_ids, 
             input_mask, 
             video, 
             video_mask,
             pairs_masked_text=None, 
             pairs_token_labels=None,
             masked_video=None, 
             video_labels_index=None,
             input_caption_ids=None, 
             decoder_mask=None,
             output_caption_ids=None, 
             training=True):
        
        # Get text features using BERT
        text_features = self.bert(
            input_ids=input_ids,
            attention_mask=input_mask,
            token_type_ids=segment_ids,
            return_dict=True
        )
        
        text_embedding = text_features.last_hidden_state  # [B, L, D]
        text_cls_embedding = text_embedding[:, 0, :]  # [B, D]
        
        # Process visual features
        video_embedding = self.visual_embeddings(video)  # [B, T, D]
        
        # Visual encoder
        extended_video_mask = tf.cast(video_mask[:, tf.newaxis, tf.newaxis, :], dtype=tf.float32)
        extended_video_mask = (1.0 - extended_video_mask) * -10000.0
        
        visual_features = self.visual_encoder(
            inputs_embeds=video_embedding,
            attention_mask=video_mask,
            return_dict=True
        )
        
        video_embedding = visual_features.last_hidden_state  # [B, T, D]
        video_cls_embedding = video_embedding[:, 0, :]  # [B, D]
        
        # Cross-modal encoding
        concat_embedding = tf.concat([text_embedding, video_embedding], axis=1)  # [B, L+T, D]
        concat_mask = tf.concat([input_mask, video_mask], axis=1)  # [B, L+T]
        
        cross_features = self.cross_encoder(
            inputs_embeds=concat_embedding,
            attention_mask=concat_mask,
            return_dict=True
        )
        
        cross_embedding = cross_features.last_hidden_state  # [B, L+T, D]
        
        # Retrieve and separate the cross outputs
        text_len = tf.shape(text_embedding)[1]
        cross_text_embedding = cross_embedding[:, :text_len, :]
        cross_video_embedding = cross_embedding[:, text_len:, :]
        
        # Calculate similarity scores for contrastive learning
        retrieve_logits = tf.matmul(text_cls_embedding, tf.transpose(video_cls_embedding))  # [B, B]
        retrieve_logits = retrieve_logits / 0.07  # Temperature scaling
        
        # Create labels for contrastive loss (diagonal is positive, rest are negatives)
        batch_size = tf.shape(retrieve_logits)[0]
        labels = tf.range(batch_size)
        
        # Contrastive loss
        contrastive_loss = tf.reduce_mean(
            self.loss_fct(labels, retrieve_logits) + self.loss_fct(labels, tf.transpose(retrieve_logits))
        )
        
        # Initialize total loss
        loss = contrastive_loss
        
        # MLM (Masked Language Modeling) if provided
        if pairs_masked_text is not None and pairs_token_labels is not None:
            text_mlm_features = self.bert(
                input_ids=pairs_masked_text,
                attention_mask=input_mask,
                token_type_ids=segment_ids,
                return_dict=True
            )
            
            mlm_embedding = text_mlm_features.last_hidden_state
            mlm_logits = self.mlm_classifier(mlm_embedding)
            
            # Calculate MLM loss
            active_mask = tf.cast(pairs_token_labels > 0, tf.float32)
            active_logits = tf.reshape(mlm_logits, [-1, self.bert_config.vocab_size])
            active_labels = tf.reshape(pairs_token_labels, [-1])
            
            mlm_loss_raw = self.loss_fct(active_labels, active_logits)
            mlm_loss = tf.reduce_sum(mlm_loss_raw * tf.reshape(active_mask, [-1])) / tf.reduce_sum(active_mask)
            
            loss += mlm_loss
        
        # MFM (Masked Frame Modeling) if provided
        if masked_video is not None and video_labels_index is not None:
            visual_mfm_features = self.visual_encoder(
                inputs_embeds=self.visual_embeddings(masked_video),
                attention_mask=video_mask,
                return_dict=True
            )
            
            mfm_embedding = visual_mfm_features.last_hidden_state
            mfm_logits = self.mfm_classifier(mfm_embedding)
            
            # Calculate MFM loss (regression)
            active_video_labels = tf.gather(video, video_labels_index, batch_dims=1)
            active_video_embedding = tf.gather(mfm_logits, video_labels_index, batch_dims=1)
            
            mfm_loss = tf.reduce_mean(self.mse_loss(active_video_labels, active_video_embedding))
            
            loss += mfm_loss
        
        # Decoder for captioning if in stage_two mode
        if self.stage_two and input_caption_ids is not None and output_caption_ids is not None:
            # Use cross_video_embedding as context for decoder
            decoder_features = self.decoder(
                input_ids=input_caption_ids,
                attention_mask=decoder_mask,
                encoder_hidden_states=cross_video_embedding,
                encoder_attention_mask=video_mask,
                return_dict=True
            )
            
            decoder_embedding = decoder_features.last_hidden_state
            caption_logits = self.decoder_classifier(decoder_embedding)
            
            # Calculate Caption loss
            caption_loss = tf.reduce_mean(self.loss_fct(output_caption_ids, caption_logits))
            
            loss += caption_loss
        
        return loss

#----------------------------------------------------

class AdamWeightDecay(tf.keras.optimizers.Adam):
    """Adam with weight decay and custom learning rate scales for different parameter groups."""

    def __init__(self,
                learning_rate=0.001,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-7,
                amsgrad=False,
                weight_decay_rate=0.0,
                include_in_weight_decay=None,
                exclude_from_weight_decay=None,
                get_decay_scale=None,
                name='AdamWeightDecay',
                **kwargs):
        super(AdamWeightDecay, self).__init__(
            learning_rate=learning_rate,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon,
            amsgrad=amsgrad,
            name=name,
            **kwargs)
        
        self.weight_decay_rate = weight_decay_rate
        self.include_in_weight_decay = include_in_weight_decay
        self.exclude_from_weight_decay = exclude_from_weight_decay
        self.get_decay_scale = get_decay_scale

    @tf.function
    def _resource_apply_dense(self, grad, var, apply_state=None):
        var_name = self._get_variable_name(var.name)
        weight_decay_rate, lr_scale = self._get_decay_scale(var_name)
        
        if weight_decay_rate > 0:
            grad = grad + weight_decay_rate * var
        
        # Apply learning rate scale if needed
        if lr_scale != 1.0:
            grad = grad * lr_scale
            
        return super(AdamWeightDecay, self)._resource_apply_dense(
            grad, var, apply_state=apply_state)

    @tf.function
    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        var_name = self._get_variable_name(var.name)
        weight_decay_rate, lr_scale = self._get_decay_scale(var_name)
        
        if weight_decay_rate > 0:
            grad = grad + weight_decay_rate * tf.gather(var, indices)
            
        # Apply learning rate scale if needed
        if lr_scale != 1.0:
            grad = grad * lr_scale
            
        return super(AdamWeightDecay, self)._resource_apply_sparse(
            grad, var, indices, apply_state=apply_state)

    def _get_variable_name(self, param_name):
        """Get the variable name from the tensor name."""
        m = re.match("^(.*):\\d+$", param_name)
        if m is not None:
            param_name = m.group(1)
        return param_name

    def _get_decay_scale(self, param_name):
        """Return weight decay rate and learning rate scale for the parameter."""
        if self.get_decay_scale:
            return self.get_decay_scale(param_name)
            
        if self.exclude_from_weight_decay:
            for r in self.exclude_from_weight_decay:
                if re.search(r, param_name) is not None:
                    return 0.0, 1.0
                    
        if self.include_in_weight_decay:
            for r in self.include_in_weight_decay:
                if re.search(r, param_name) is not None:
                    return self.weight_decay_rate, 1.0
                    
        return 0.0, 1.0


#----------------------------------------------------------------

import tensorflow as tf
import numpy as np
import json
import csv
import os
import random
from copy import deepcopy

class TFPretrainDataset:
    """TensorFlow dataset for UniVL pretraining."""
    
    def __init__(self, args, tokenizer, stage_one=True):
        self.args = args
        self.tokenizer = tokenizer
        self.stage_one = stage_one
        self.max_words = args.max_words
        self.max_frames = args.max_frames
        self.feature_framerate = args.feature_framerate
        self.use_mil = args.use_mil
        self.pretrain_enhance_vmodal = args.pretrain_enhance_vmodal
        
        # Load data dict
        print(f"Loading data from {args.data_path}")
        with open(args.data_path, 'r') as f:
            self.data_dict = json.load(f)
        
        print(f"Dataset loaded with {len(self.data_dict)} entries")
        
        # Get video lists
        self.video_list = list(self.data_dict.keys())
        
    def __len__(self):
        return len(self.video_list)
    
    def _mask_tokens(self, inputs, token_labels):
        """Mask tokens for masked language modeling prediction"""
        mask_vocab_size = self.tokenizer.vocab_size
        mask_token_id = self.tokenizer.mask_token_id
        
        indices_replaced = tf.random.uniform(shape=tf.shape(inputs)) < 0.8
        inputs = tf.where(
            tf.logical_and(indices_replaced, token_labels > 0),
            mask_token_id * tf.ones_like(inputs),
            inputs
        )
        
        # 10% randomly change token to random token
        indices_random = tf.random.uniform(shape=tf.shape(inputs)) < 0.1
        random_words = tf.random.uniform(
            shape=tf.shape(inputs), minval=0, maxval=mask_vocab_size, dtype=tf.int64)
        inputs = tf.where(
            tf.logical_and(tf.logical_and(indices_random, token_labels > 0), tf.logical_not(indices_replaced)),
            random_words,
            inputs
        )
        
        return inputs
    
    def _mask_frames(self, video_feature, video_mask):
        """Mask frames for masked frame modeling prediction"""
        probability_matrix = tf.cast(video_mask, tf.float32)
        indices_mask = tf.random.uniform(shape=tf.shape(video_mask)) < 0.15
        
        # Need to get the indices where to mask
        video_labels_index = tf.where(
            tf.logical_and(indices_mask, tf.cast(video_mask, tf.bool)))
        
        # Create the mask
        masked_video = tf.identity(video_feature)
        # Replace masked indices with zero vector
        batch_indices = video_labels_index[:, 0]
        frame_indices = video_labels_index[:, 1]
        
        # Here we would have used scatter to update masked values, but TF doesn't have
        # a nice scatter operation like PyTorch. Instead we can use a loop or tf.tensor_scatter_nd_update
        # For now, let's use a dummy approach to show the idea:
        
        # Convert indices to a sparse mask
        updates = tf.zeros_like(tf.gather_nd(video_feature, video_labels_index))
        masked_video = tf.tensor_scatter_nd_update(
            masked_video, video_labels_index, updates)
            
        return masked_video, video_labels_index
    
    def process_raw_data(self, video_id):
        """Process a single video data entry"""
        video_feature_path = os.path.join(self.args.features_path, f"{video_id}.npy")
        
        # Load video features
        video_feature = np.load(video_feature_path)
        
        # Filter and sample frames
        video_feature = video_feature[::self.feature_framerate]
        if self.max_frames < video_feature.shape[0]:
            video_feature = video_feature[:self.max_frames]
        
        # Get video mask
        video_mask = np.ones(video_feature.shape[0], dtype=np.int64)
        
        # Pad if needed
        if self.max_frames > video_feature.shape[0]:
            pad_len = self.max_frames - video_feature.shape[0]
            video_padding = np.zeros((pad_len, video_feature.shape[1]), dtype=np.float32)
            video_feature = np.concatenate((video_feature, video_padding), axis=0)
            video_mask = np.concatenate((video_mask, np.zeros(pad_len, dtype=np.int64)), axis=0)
        
        # Get caption
        caption = self.data_dict[video_id]["caption"]
        if isinstance(caption, list):
            # Select a random caption if there are multiple
            caption = random.choice(caption)
        
        # Tokenize caption
        input_caption_tokenized = self.tokenizer(
            caption,
            max_length=self.max_words,
            padding='max_length',
            truncation=True,
            return_tensors="tf"
        )
        
        input_ids = input_caption_tokenized["input_ids"][0]
        input_mask = input_caption_tokenized["attention_mask"][0]
        segment_ids = tf.zeros_like(input_mask)
        
        # For MLM, copy the input_ids to prepare for masking
        input_ids_masked = tf.identity(input_ids)
        token_labels = tf.cast(input_mask, tf.int64)
        # Ignore special tokens
        special_tokens_mask = tf.cast(
            tf.math.equal(input_ids, self.tokenizer.cls_token_id) |
            tf.math.equal(input_ids, self.tokenizer.sep_token_id) |
            tf.math.equal(input_ids, self.tokenizer.pad_token_id),
            tf.int64
        )
        token_labels = token_labels * (1 - special_tokens_mask)
        
        # Apply masking for MLM
        masked_input_ids = self._mask_tokens(input_ids_masked, token_labels)
        
        # For MFM, copy the video feature to prepare for masking
        video_feature_masked, video_labels_index = self._mask_frames(video_feature, video_mask)
        
        # If using MIL, need captions for decoder
        if not self.stage_one:
            # For decoder captioning task
            decoder_input_ids = tf.identity(input_ids)
            decoder_mask = tf.identity(input_mask)
            decoder_output_ids = tf.identity(input_ids)
        else:
            # Stage one doesn't use decoder
            decoder_input_ids = tf.zeros_like(input_ids)
            decoder_mask = tf.zeros_like(input_mask)
            decoder_output_ids = tf.zeros_like(input_ids)
        
        return (input_ids, input_mask, segment_ids, video_feature, video_mask,
                masked_input_ids, token_labels, video_feature_masked, video_labels_index,
                decoder_input_ids, decoder_mask, decoder_output_ids)
    
    def create_tfrecord_dataset(self, batch_size, is_training=True):
        """Create a TF Dataset from the raw data"""
        
        def generator():
            for video_id in self.video_list:
                try:
                    yield self.process_raw_data(video_id)
                except Exception as e:
                    print(f"Error processing {video_id}: {e}")
                    continue
        
        # Define the output types for the generator
        output_types = (
            tf.int64, tf.int64, tf.int64,  # input_ids, input_mask, segment_ids
            tf.float32, tf.int64,  # video_feature, video_mask
            tf.int64, tf.int64,  # masked_input_ids, token_labels
            tf.float32, tf.int64,  # video_feature_masked, video_labels_index
            tf.int64, tf.int64, tf.int64  # decoder_input_ids, decoder_mask, decoder_output_ids
        )
        
        # Create dataset from generator
        dataset = tf.data.Dataset.from_generator(
            generator,
            output_types=output_types
        )
        
        if is_training:
            dataset = dataset.shuffle(buffer_size=1000)
        
        # Batch and prefetch
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        
        return dataset


def create_pretrain_dataset(args, tokenizer, _strategy=None):
    """Create the final distributed dataset for training"""
    
    dataset_creator = TFPretrainDataset(
        args=args,
        tokenizer=tokenizer,
        stage_one=not args.stage_two
    )
    
    per_replica_batch_size = args.batch_size
    
    dataset = dataset_creator.create_tfrecord_dataset(
        batch_size=per_replica_batch_size,
        is_training=True
    )
    
    if _strategy is not None:
        # Distribute dataset
        dataset = _strategy.experimental_distribute_dataset(dataset)
    
    return dataset

#--------------------------------------------------------------
from transformers import BertTokenizer as HFBertTokenizer
import tensorflow as tf

class BertTokenizer:
    """Wrapper for Huggingface's BertTokenizer to match the original interface"""
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, do_lower_case=True):
        tokenizer = HFBertTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            do_lower_case=do_lower_case)
        
        # Wrap the tokenizer in our class
        return cls(tokenizer)
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
        # Add vocab properties that the original code might expect
        self.vocab_size = tokenizer.vocab_size
        self.cls_token_id = tokenizer.cls_token_id
        self.sep_token_id = tokenizer.sep_token_id
        self.pad_token_id = tokenizer.pad_token_id
        self.mask_token_id = tokenizer.mask_token_id
        
    def __call__(self, *args, **kwargs):
        return self.tokenizer(*args, **kwargs)


#------------------------------------------------

import logging
import sys
import os
import time

def get_logger(filename=None):
    """Get a logger that writes to both stdout and optionally to a file"""
    
    # Create logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s', 
        datefmt='%Y-%m-%d %H:%M:%S')
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    
    # Optionally create file handler
    if filename is not None:
        file_handler = logging.FileHandler(filename, mode='a')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Set propagation to False to avoid duplicate logs
    logger.propagate = False
    
    return logger
import tensorflow as tf
import numpy as np
from keras.optimizers import Adam
from keras.optimizers.schedules import PolynomialDecay
from keras.metrics import Mean, SparseCategoricalAccuracy, CategoricalAccuracy
import os
import time
import json
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge

class SportsCaptioningTrainer:
    """
    Trainer for the Sports Video Understanding Model
    """
    def __init__(
        self,
        model,
        train_dataset,
        val_dataset=None,
        learning_rate=1e-4,
        warmup_steps=10000,
        weight_decay=0.01,
        max_grad_norm=1.0,
        checkpoint_path='checkpoints',
        log_freq=100
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.checkpoint_path = checkpoint_path
        self.log_freq = log_freq
        
        os.makedirs(checkpoint_path, exist_ok=True)
        
        self.lr_schedule = PolynomialDecay(
            initial_learning_rate=learning_rate,
            decay_steps=warmup_steps,
            end_learning_rate=learning_rate * 0.1,
            power=1.0
        )
        
        self.optimizer = Adam(learning_rate=self.lr_schedule, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
        
        self.caption_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='none')
        self.action_loss_fn = tf.keras.losses.CategoricalCrossentropy(
            from_logits=True, reduction='none')
        self.player_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='none')
        
        self.train_caption_loss = Mean(name='train_caption_loss')
        self.train_action_coarse_loss = Mean(name='train_action_coarse_loss')
        self.train_action_fine_loss = Mean(name='train_action_fine_loss')
        self.train_action_event_loss = Mean(name='train_action_event_loss')
        self.train_player_loss = Mean(name='train_player_loss')
        self.train_total_loss = Mean(name='train_total_loss')
        
        self.val_caption_loss = Mean(name='val_caption_loss')
        self.val_action_coarse_loss = Mean(name='val_action_coarse_loss')
        self.val_action_fine_loss = Mean(name='val_action_fine_loss')
        self.val_action_event_loss = Mean(name='val_action_event_loss')
        self.val_player_loss = Mean(name='val_player_loss')
        self.val_total_loss = Mean(name='val_total_loss')
        
        self.caption_accuracy = SparseCategoricalAccuracy(name='caption_accuracy')
        self.action_coarse_accuracy = CategoricalAccuracy(name='action_coarse_accuracy')
        self.action_fine_accuracy = CategoricalAccuracy(name='action_fine_accuracy')
        self.action_event_accuracy = CategoricalAccuracy(name='action_event_accuracy')
        self.player_accuracy = SparseCategoricalAccuracy(name='player_accuracy')
        
        self.checkpoint = tf.train.Checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            step=tf.Variable(0),
            epoch=tf.Variable(0)
        )
        self.checkpoint_manager = tf.train.CheckpointManager(
            self.checkpoint, self.checkpoint_path, max_to_keep=5)
        
        self.restore_checkpoint()
        
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        
    def restore_checkpoint(self):
        """Restore from latest checkpoint if available"""
        latest_checkpoint = self.checkpoint_manager.latest_checkpoint
        if latest_checkpoint:
            self.checkpoint.restore(latest_checkpoint)
            print(f"Restored from checkpoint: {latest_checkpoint}")
            print(f"Starting from epoch {int(self.checkpoint.epoch)}, step {int(self.checkpoint.step)}")
        else:
            print("Initializing from scratch")
    
    def reset_metrics(self):
        """Reset all metrics at the start of an epoch"""
        metrics = [
            self.train_caption_loss, self.train_action_coarse_loss,
            self.train_action_fine_loss, self.train_action_event_loss,
            self.train_player_loss, self.train_total_loss,
            self.val_caption_loss, self.val_action_coarse_loss,
            self.val_action_fine_loss, self.val_action_event_loss,
            self.val_player_loss, self.val_total_loss,
            self.caption_accuracy, self.action_coarse_accuracy,
            self.action_fine_accuracy, self.action_event_accuracy,
            self.player_accuracy
        ]
        
        for metric in metrics:
            metric.reset_states()
    
    @tf.function
    def compute_caption_loss(self, caption_logits, caption_labels, caption_mask):
        """Compute loss for caption generation"""
        loss = self.caption_loss_fn(caption_labels, caption_logits)
        
        mask = tf.cast(caption_mask, dtype=loss.dtype)
        loss *= mask
        
        token_count = tf.reduce_sum(mask)
        total_loss = tf.reduce_sum(loss) / (token_count + 1e-8)
        
        return total_loss
    
    @tf.function
    def compute_action_loss(self, action_logits, action_labels, task_type):
        """Compute loss for action recognition"""
        one_hot_labels = tf.one_hot(action_labels, depth=tf.shape(action_logits)[-1])
        
        loss = self.action_loss_fn(one_hot_labels, action_logits)
        
        mask = tf.cast(task_type == 1, dtype=loss.dtype)
        loss *= mask
        
        sample_count = tf.reduce_sum(mask)
        total_loss = tf.reduce_sum(loss) / (sample_count + 1e-8)
        
        return total_loss
    
    @tf.function
    def compute_player_loss(self, player_logits, player_labels, task_type):
        """Compute loss for player identification"""
        loss = self.player_loss_fn(player_labels, player_logits)
        
        mask = tf.cast(task_type == 2, dtype=loss.dtype)
        loss *= mask
        
        sample_count = tf.reduce_sum(mask)
        total_loss = tf.reduce_sum(loss) / (sample_count + 1e-8)
        
        return total_loss
    
    @tf.function
    def train_step(self, inputs):
        """Single training step"""
        (
            video, video_mask, ball_crops, player_crops, basket_crops,
            court_images, player_positions, ball_positions, basket_positions,
            input_caption_ids, decoder_mask, caption_labels,
            action_coarse_labels, action_fine_labels, action_event_labels,
            player_labels, task_type
        ) = inputs
        
        with tf.GradientTape() as tape:
            model_outputs = self.model(
                (
                    video, video_mask, ball_crops, player_crops, basket_crops,
                    court_images, player_positions, ball_positions, basket_positions,
                    input_caption_ids, decoder_mask
                ),
                training=True
            )
            
            caption_loss = self.compute_caption_loss(
                model_outputs['caption_logits'],
                caption_labels,
                decoder_mask
            )
            
            action_coarse_loss = self.compute_action_loss(
                model_outputs['coarse_action_logits'],
                action_coarse_labels,
                task_type
            )
            
            action_fine_loss = self.compute_action_loss(
                model_outputs['fine_action_logits'],
                action_fine_labels,
                task_type
            )
            
            action_event_loss = self.compute_action_loss(
                model_outputs['event_action_logits'],
                action_event_labels,
                task_type
            )
            
            player_loss = self.compute_player_loss(
                model_outputs['player_logits'],
                player_labels,
                task_type
            )
            
            total_loss = caption_loss + action_coarse_loss + action_fine_loss + action_event_loss + player_loss
            
            if self.weight_decay > 0:
                for var in self.model.trainable_variables:
                    if 'bias' not in var.name and 'layer_norm' not in var.name:
                        total_loss += self.weight_decay * tf.nn.l2_loss(var)
        
        gradients = tape.gradient(total_loss, self.model.trainable_variables)
        
        if self.max_grad_norm > 0:
            gradients, _ = tf.clip_by_global_norm(gradients, self.max_grad_norm)
        
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        self.train_caption_loss.update_state(caption_loss)
        self.train_action_coarse_loss.update_state(action_coarse_loss)
        self.train_action_fine_loss.update_state(action_fine_loss)
        self.train_action_event_loss.update_state(action_event_loss)
        self.train_player_loss.update_state(player_loss)
        self.train_total_loss.update_state(total_loss)
        
        self.caption_accuracy.update_state(
            caption_labels, model_outputs['caption_logits'], sample_weight=decoder_mask)
        
        action_mask = tf.cast(task_type == 1, dtype=tf.bool)
        if tf.reduce_any(action_mask):
            self.action_coarse_accuracy.update_state(
                tf.one_hot(action_coarse_labels, depth=tf.shape(model_outputs['coarse_action_logits'])[-1]),
                tf.boolean_mask(model_outputs['coarse_action_logits'], action_mask))
            
            self.action_fine_accuracy.update_state(
                tf.one_hot(action_fine_labels, depth=tf.shape(model_outputs['fine_action_logits'])[-1]),
                tf.boolean_mask(model_outputs['fine_action_logits'], action_mask))
            
            self.action_event_accuracy.update_state(
                tf.one_hot(action_event_labels, depth=tf.shape(model_outputs['event_action_logits'])[-1]),
                tf.boolean_mask(model_outputs['event_action_logits'], action_mask))
        
        player_mask = tf.cast(task_type == 2, dtype=tf.bool)
        if tf.reduce_any(player_mask):
            self.player_accuracy.update_state(
                tf.boolean_mask(player_labels, player_mask),
                tf.boolean_mask(model_outputs['player_logits'], player_mask))
        
        return total_loss
    
    @tf.function
    def val_step(self, inputs):
        """Single validation step"""
        (
            video, video_mask, ball_crops, player_crops, basket_crops,
            court_images, player_positions, ball_positions, basket_positions,
            input_caption_ids, decoder_mask, caption_labels,
            action_coarse_labels, action_fine_labels, action_event_labels,
            player_labels, task_type
        ) = inputs
        
        model_outputs = self.model(
            (
                video, video_mask, ball_crops, player_crops, basket_crops,
                court_images, player_positions, ball_positions, basket_positions,
                input_caption_ids, decoder_mask
            ),
            training=False
        )
        
        caption_loss = self.compute_caption_loss(
            model_outputs['caption_logits'],
            caption_labels,
            decoder_mask
        )
        
        action_coarse_loss = self.compute_action_loss(
            model_outputs['coarse_action_logits'],
            action_coarse_labels,
            task_type
        )
        
        action_fine_loss = self.compute_action_loss(
            model_outputs['fine_action_logits'],
            action_fine_labels,
            task_type
        )
        
        action_event_loss = self.compute_action_loss(
            model_outputs['event_action_logits'],
            action_event_labels,
            task_type
        )
        
        player_loss = self.compute_player_loss(
            model_outputs['player_logits'],
            player_labels,
            task_type
        )
        
        total_loss = caption_loss + action_coarse_loss + action_fine_loss + action_event_loss + player_loss
        
        self.val_caption_loss.update_state(caption_loss)
        self.val_action_coarse_loss.update_state(action_coarse_loss)
        self.val_action_fine_loss.update_state(action_fine_loss)
        self.val_action_event_loss.update_state(action_event_loss)
        self.val_player_loss.update_state(player_loss)
        self.val_total_loss.update_state(total_loss)
        
        return model_outputs
    
    def train(self, epochs):
        """Train the model for specified number of epochs"""
        start_epoch = int(self.checkpoint.epoch)
        
        for epoch in range(start_epoch, start_epoch + epochs):
            print(f"\nEpoch {epoch + 1}/{start_epoch + epochs}")
            self.reset_metrics()
            
            start_time = time.time()
            step = 0
            
            for batch in self.train_dataset:
                loss = self.train_step(batch)
                step += 1
                
                self.checkpoint.step.assign_add(1)
                
                if step % self.log_freq == 0:
                    print(f"Step {step}: Loss = {loss:.4f}, "
                          f"LR = {self.optimizer.learning_rate.numpy():.6f}, "
                          f"Caption Loss = {self.train_caption_loss.result():.4f}, "
                          f"Action Coarse Loss = {self.train_action_coarse_loss.result():.4f}, "
                          f"Action Fine Loss = {self.train_action_fine_loss.result():.4f}, "
                          f"Action Event Loss = {self.train_action_event_loss.result():.4f}, "
                          f"Player Loss = {self.train_player_loss.result():.4f}, "
                          f"Caption Acc = {self.caption_accuracy.result():.4f}, "
                          f"Action Coarse Acc = {self.action_coarse_accuracy.result():.4f}, "
                          f"Action Fine Acc = {self.action_fine_accuracy.result():.4f}, "
                          f"Action Event Acc = {self.action_event_accuracy.result():.4f}, "
                          f"Player Acc = {self.player_accuracy.result():.4f}")
            
            train_time = time.time() - start_time
            print(f"Training time: {train_time:.2f}s")
            
            print(f"Train Caption Loss: {self.train_caption_loss.result():.4f}")
            print(f"Train Action Coarse Loss: {self.train_action_coarse_loss.result():.4f}")
            print(f"Train Action Fine Loss: {self.train_action_fine_loss.result():.4f}")
            print(f"Train Action Event Loss: {self.train_action_event_loss.result():.4f}")
            print(f"Train Player Loss: {self.train_player_loss.result():.4f}")
            print(f"Train Total Loss: {self.train_total_loss.result():.4f}")
            print(f"Train Caption Accuracy: {self.caption_accuracy.result():.4f}")
            print(f"Train Action Coarse Accuracy: {self.action_coarse_accuracy.result():.4f}")
            print(f"Train Action Fine Accuracy: {self.action_fine_accuracy.result():.4f}")
            print(f"Train Action Event Accuracy: {self.action_event_accuracy.result():.4f}")
            print(f"Train Player Accuracy: {self.player_accuracy.result():.4f}")
            
            if self.val_dataset is not None:
                self.evaluate()
            
            self.checkpoint.epoch.assign_add(1)
            
            self.checkpoint_manager.save()
            
            model_path = os.path.join(self.checkpoint_path, f"model_epoch_{epoch + 1}")
            self.model.save_weights(model_path)
            print(f"Saved model weights to {model_path}")
    
    def evaluate(self):
        """Evaluate the model on validation data"""
        print("\nRunning validation...")
        start_time = time.time()
        
        all_pred_captions = []
        all_true_captions = []
        
        for batch in self.val_dataset:
            _, _, _, _, _, _, _, _, _, _, decoder_mask, caption_labels, _, _, _, _, _ = batch
            
            model_outputs = self.val_step(batch)
            
            caption_preds = tf.argmax(model_outputs['caption_logits'], axis=-1)
            
            caption_preds_np = caption_preds.numpy()
            caption_labels_np = caption_labels.numpy()
            decoder_mask_np = decoder_mask.numpy()

            for i in range(caption_preds_np.shape[0]):
                pred_caption = []
                true_caption = []
                for j in range(caption_preds_np.shape[1]):
                    if decoder_mask_np[i, j] == 1:
                        pred_caption.append(caption_preds_np[i, j])
                        true_caption.append(caption_labels_np[i, j])
                
                all_pred_captions.append(pred_caption)
                all_true_captions.append([true_caption])
        
        print(f"Val Caption Loss: {self.val_caption_loss.result():.4f}")
        print(f"Val Action Coarse Loss: {self.val_action_coarse_loss.result():.4f}")
        print(f"Val Action Fine Loss: {self.val_action_fine_loss.result():.4f}")
        print(f"Val Action Event Loss: {self.val_action_event_loss.result():.4f}")
        print(f"Val Player Loss: {self.val_player_loss.result():.4f}")
        print(f"Val Total Loss: {self.val_total_loss.result():.4f}")
        
        bleu1 = corpus_bleu(all_true_captions, all_pred_captions, weights=(1, 0, 0, 0))
        bleu2 = corpus_bleu(all_true_captions, all_pred_captions, weights=(0.5, 0.5, 0, 0))
        bleu3 = corpus_bleu(all_true_captions, all_pred_captions, weights=(0.33, 0.33, 0.33, 0))
        bleu4 = corpus_bleu(all_true_captions, all_pred_captions, weights=(0.25, 0.25, 0.25, 0.25))
        
        print(f"BLEU-1: {bleu1:.4f}")
        print(f"BLEU-2: {bleu2:.4f}")
        print(f"BLEU-3: {bleu3:.4f}")
        print(f"BLEU-4: {bleu4:.4f}")
        
        val_time = time.time() - start_time
        print(f"Validation time: {val_time:.2f}s")
        
        return {
            'caption_loss': self.val_caption_loss.result(),
            'action_coarse_loss': self.val_action_coarse_loss.result(),
            'action_fine_loss': self.val_action_fine_loss.result(),
            'action_event_loss': self.val_action_event_loss.result(),
            'player_loss': self.val_player_loss.result(),
            'total_loss': self.val_total_loss.result(),
            'bleu1': bleu1,
            'bleu2': bleu2,
            'bleu3': bleu3,
            'bleu4': bleu4
        }

    def generate_caption(self, video, tokenizer, max_length=30):
        """
        Generate caption for a video using beam search
        
        Args:
            video: Video tensor with shape [1, T, H, W, C]
            tokenizer: Tokenizer for decoding
            max_length: Maximum caption length
            
        Returns:
            Generated caption string
        """
        batch_size = tf.shape(video)[0]
        seq_len = tf.shape(video)[1]
        
        dummy_crops = tf.zeros([batch_size, seq_len, 224, 224, 3])
        dummy_positions = tf.zeros([batch_size, seq_len, 2])
        
        start_token = tokenizer.cls_token_id
        decoder_input = tf.expand_dims(tf.constant([start_token]), 0)
        
        decoder_mask = tf.ones_like(decoder_input)
        
        video_mask = tf.ones([batch_size, seq_len], dtype=tf.int32)
        
        output_caption = self._beam_search(
            video, video_mask, dummy_crops, dummy_crops, dummy_crops,
            dummy_crops, dummy_positions, dummy_positions, dummy_positions,
            decoder_input, decoder_mask, beam_size=5, max_length=max_length, tokenizer=tokenizer
        )
        
        return output_caption
    
    def _beam_search(self, video, video_mask, ball_crops, player_crops, basket_crops,
                  court_images, player_positions, ball_positions, basket_positions,
                  decoder_input, decoder_mask, beam_size=5, max_length=30, tokenizer=None):
        """
        Perform beam search for caption generation
        
        Returns:
            Generated caption string
        """
        model_outputs = self.model(
            (
                video, video_mask, ball_crops, player_crops, basket_crops,
                court_images, player_positions, ball_positions, basket_positions,
                decoder_input, decoder_mask
            ),
            training=False
        )
        
        encoder_output = model_outputs['fused_features']
    
        batch_size = 1
        
        start_token = tokenizer.cls_token_id
        end_token = tokenizer.sep_token_id

        initial_ids = tf.constant([[start_token]], dtype=tf.int32)
        initial_mask = tf.ones_like(initial_ids, dtype=tf.int32)

        beam_scores = tf.zeros([batch_size, beam_size], dtype=tf.float32)
        beam_seqs = tf.tile(initial_ids, [batch_size, beam_size])
        beam_masks = tf.tile(initial_mask, [batch_size, beam_size])

        finished_seqs = tf.zeros([batch_size, beam_size, 0], dtype=tf.int32)
        finished_scores = tf.fill([batch_size, beam_size], -float('inf'))

        repeated_encoder_output = tf.tile(
            encoder_output[:, tf.newaxis, :, :],
            [1, beam_size, 1, 1]
        )
        repeated_encoder_output = tf.reshape(
            repeated_encoder_output,
            [batch_size * beam_size, tf.shape(encoder_output)[1], tf.shape(encoder_output)[2]]
        )

        for step in range(max_length):
            decoder_input_ids = beam_seqs
            decoder_input_mask = beam_masks
            
            flat_decoder_input_ids = tf.reshape(decoder_input_ids, [batch_size * beam_size, -1])
            flat_decoder_input_mask = tf.reshape(decoder_input_mask, [batch_size * beam_size, -1])
            
            caption_logits = self.model.decoder(
                flat_decoder_input_ids,
                repeated_encoder_output,
                input_mask=flat_decoder_input_mask,
                training=False
            )

            next_token_logits = caption_logits[:, -1, :]

            next_token_probs = tf.nn.softmax(next_token_logits, axis=-1)

            vocabulary_size = tf.shape(next_token_probs)[1]
            next_token_probs = tf.reshape(next_token_probs, [batch_size, beam_size, -1])

            beam_scores_for_next_step = tf.expand_dims(beam_scores, -1) + tf.math.log(next_token_probs + 1e-10)
 
            flattened_beam_scores = tf.reshape(beam_scores_for_next_step, [batch_size, -1])

            top_k_scores, top_k_indices = tf.math.top_k(flattened_beam_scores, k=beam_size * 2)

            beam_indices = top_k_indices // vocabulary_size
            token_indices = top_k_indices % vocabulary_size
            
            new_beam_seqs = []
            new_beam_masks = []
            new_beam_scores = []
            
            for b in range(batch_size):
                for i in range(beam_size * 2):
                    beam_idx = beam_indices[b, i]
                    token_idx = token_indices[b, i]
                    score = top_k_scores[b, i]
                    
                    curr_seq = beam_seqs[b, beam_idx]
                    curr_mask = beam_masks[b, beam_idx]
                    
                    if step > 0 and curr_seq[-1] == end_token:
                        finished_seqs = tf.concat([
                            finished_seqs,
                            tf.expand_dims(tf.expand_dims(curr_seq, 0), 0)
                        ], axis=2)
                        finished_scores = tf.concat([
                            finished_scores,
                            tf.expand_dims(tf.expand_dims(score, 0), 0)
                        ], axis=2)
                        continue

                    new_seq = tf.concat([curr_seq, [token_idx]], axis=0)
                    new_mask = tf.concat([curr_mask, [1]], axis=0)
                    
                    new_beam_seqs.append(new_seq)
                    new_beam_masks.append(new_mask)
                    new_beam_scores.append(score)

                    if token_idx == end_token:
                        finished_seqs = tf.concat([
                            finished_seqs,
                            tf.expand_dims(tf.expand_dims(new_seq, 0), 0)
                        ], axis=2)
                        finished_scores = tf.concat([
                            finished_scores,
                            tf.expand_dims(tf.expand_dims(score, 0), 0)
                        ], axis=2)
            
            if len(new_beam_seqs) <= beam_size:
                beam_seqs = tf.stack(new_beam_seqs)
                beam_masks = tf.stack(new_beam_masks)
                beam_scores = tf.stack(new_beam_scores)
            else:
                top_indices = tf.argsort(new_beam_scores, direction='DESCENDING')[:beam_size]
                selected_seqs = tf.gather(new_beam_seqs, top_indices)
                selected_masks = tf.gather(new_beam_masks, top_indices)
                selected_scores = tf.gather(new_beam_scores, top_indices)
                
                beam_seqs = selected_seqs
                beam_masks = selected_masks
                beam_scores = selected_scores
            
            if tf.reduce_all(beam_seqs[:, -1] == end_token):
                break
        
        best_score_idx = tf.argmax(beam_scores, axis=-1)
        best_seq = tf.gather(beam_seqs, best_score_idx, batch_dims=1)
        
        caption = tokenizer.decode(best_seq[0].numpy())
        
        return caption
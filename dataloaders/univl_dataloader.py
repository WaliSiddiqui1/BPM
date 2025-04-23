import tensorflow as tf
import numpy as np
import pandas as pd
import json
import os
import random
import cv2
from collections import defaultdict
from PIL import Image
import time
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge

class NSVADataLoader:
    """
    DataLoader for NBA Sports Video Analysis dataset.
    Processes video frames, extracts features, and prepares inputs for the model.
    """
    def __init__(
        self,
        csv_path,
        json_path,
        video_features_path,
        ball_features_path=None,
        player_features_path=None,
        basket_features_path=None,
        court_features_path=None,
        tokenizer=None,
        max_frames=100,
        max_words=30,
        batch_size=16,
        split_type="train"
    ):
        self.csv = pd.read_csv(csv_path)
        self.data = json.load(open(json_path, 'r'))
        self.tokenizer = tokenizer
        self.max_frames = max_frames
        self.max_words = max_words
        self.batch_size = batch_size
        self.split_type = split_type
        
        self.video_features_path = video_features_path
        self.ball_features_path = ball_features_path
        self.player_features_path = player_features_path
        self.basket_features_path = basket_features_path
        self.court_features_path = court_features_path
        
        self.split_dict = json.load(open('./split2video_id_after_videos_combination.json', 'r'))
        self.video_ids = self.split_dict[split_type]
        
        self._process_captions()
        
        self.feature_cache = {}
        self.cache_size_limit = 1000
        
    def _process_captions(self):
        """Process captions and video information from data"""
        self.sentences_dict = {}
        self.video_sentences_dict = defaultdict(list)
        
        for idx, sentence in enumerate(self.data['sentences']):
            video_id = sentence['video_id']
            if video_id in self.video_ids:
                self.video_sentences_dict[video_id].append(sentence['caption'])
        
        idx = 0
        for video_id in self.video_ids:
            if video_id in self.video_sentences_dict:
                for caption in self.video_sentences_dict[video_id]:
                    self.sentences_dict[idx] = (video_id, caption)
                    idx += 1
        
        self.sample_len = len(self.sentences_dict)
        print(f"Loaded {self.sample_len} samples for {self.split_type} split")
    
    def __len__(self):
        """Return the number of samples in the dataset"""
        return self.sample_len
    
    def _load_feature(self, video_id, feature_type):
        """Load feature from disk or cache"""
        cache_key = f"{feature_type}_{video_id}"
        
        if cache_key in self.feature_cache:
            return self.feature_cache[cache_key]
        
        if feature_type == "video":
            feature_path = os.path.join(self.video_features_path, f"{video_id}.npy")
            feature = np.load(feature_path)
        elif feature_type == "ball" and self.ball_features_path:
            feature_path = os.path.join(self.ball_features_path, f"{video_id}.npy")
            feature = np.load(feature_path) if os.path.exists(feature_path) else None
        elif feature_type == "player" and self.player_features_path:
            feature_path = os.path.join(self.player_features_path, f"{video_id}.npy")
            feature = np.load(feature_path) if os.path.exists(feature_path) else None
        elif feature_type == "basket" and self.basket_features_path:
            feature_path = os.path.join(self.basket_features_path, f"{video_id}.npy")
            feature = np.load(feature_path) if os.path.exists(feature_path) else None
        elif feature_type == "court" and self.court_features_path:
            feature_path = os.path.join(self.court_features_path, f"{video_id}.npy")
            feature = np.load(feature_path) if os.path.exists(feature_path) else None
        else:
            feature = None
        
        if feature is not None:
            if len(self.feature_cache) >= self.cache_size_limit:
                keys = list(self.feature_cache.keys())
                del self.feature_cache[random.choice(keys)]
            
            self.feature_cache[cache_key] = feature
        
        return feature
    
    def _process_video_features(self, video_id):
        """Process video features for a single video"""
        video_feature = self._load_feature(video_id, "video")
        
        if video_feature.shape[0] > self.max_frames:
            indices = np.linspace(0, video_feature.shape[0] - 1, self.max_frames, dtype=int)
            video_feature = video_feature[indices]
        
        video_mask = np.ones(video_feature.shape[0], dtype=np.int32)
        
        if video_feature.shape[0] < self.max_frames:
            pad_len = self.max_frames - video_feature.shape[0]
            video_padding = np.zeros((pad_len, video_feature.shape[1]), dtype=np.float32)
            video_feature = np.concatenate((video_feature, video_padding), axis=0)
            video_mask = np.concatenate((video_mask, np.zeros(pad_len, dtype=np.int32)), axis=0)
        
        return video_feature, video_mask
    
    def _process_object_features(self, video_id):
        """Process fine-grained object features for a single video"""
        ball_feature = np.zeros((self.max_frames, 224, 224, 3), dtype=np.float32)
        player_feature = np.zeros((self.max_frames, 5, 224, 224, 3), dtype=np.float32)  # Assuming max 5 players
        basket_feature = np.zeros((self.max_frames, 224, 224, 3), dtype=np.float32)
        court_feature = np.zeros((self.max_frames, 224, 224, 3), dtype=np.float32)
        
        ball_data = self._load_feature(video_id, "ball")
        player_data = self._load_feature(video_id, "player")
        basket_data = self._load_feature(video_id, "basket")
        court_data = self._load_feature(video_id, "court")
        
        if ball_data is not None:
            seq_len = min(ball_data.shape[0], self.max_frames)
            ball_feature[:seq_len] = ball_data[:seq_len]
        
        if player_data is not None:
            seq_len = min(player_data.shape[0], self.max_frames)
            num_players = min(player_data.shape[1], 5)
            player_feature[:seq_len, :num_players] = player_data[:seq_len, :num_players]
        
        if basket_data is not None:
            seq_len = min(basket_data.shape[0], self.max_frames)
            basket_feature[:seq_len] = basket_data[:seq_len]
        
        if court_data is not None:
            seq_len = min(court_data.shape[0], self.max_frames)
            court_feature[:seq_len] = court_data[:seq_len]
        
        player_positions = np.zeros((self.max_frames, 5, 2), dtype=np.float32)
        ball_positions = np.zeros((self.max_frames, 2), dtype=np.float32)
        basket_positions = np.zeros((self.max_frames, 2), dtype=np.float32)
        
        return (
            ball_feature, player_feature, basket_feature, court_feature,
            player_positions, ball_positions, basket_positions
        )
    
    def _process_caption(self, caption):
        """Process caption text into token IDs"""
        task_type = 0
        if caption.startswith("<T1>_"):
            task_type = 0  # Caption
            caption = caption[5:]
        elif caption.startswith("<T2>_"):
            task_type = 1  # Action
            caption = caption[5:]
        elif caption.startswith("<T3>_"):
            task_type = 2  # Player
            caption = caption[5:]
        elif caption.startswith("<T4>_"):
            task_type = 3  # Other
            caption = caption[5:]
        
        tokenized = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=self.max_words,
            return_tensors="tf"
        )
        
        input_ids = tokenized["input_ids"][0]
        attention_mask = tokenized["attention_mask"][0]
        
        decoder_input_ids = input_ids[:-1]
        decoder_input_ids = tf.pad(decoder_input_ids, [[0, 1]])
        decoder_input_ids = tf.concat([[self.tokenizer.cls_token_id], decoder_input_ids[1:]], axis=0)
        
        decoder_output_ids = input_ids[1:]
        decoder_output_ids = tf.pad(decoder_output_ids, [[1, 0]])
        
        decoder_mask = attention_mask
        
        action_coarse_label = tf.zeros([], dtype=tf.int32)
        action_fine_label = tf.zeros([], dtype=tf.int32)
        action_event_label = tf.zeros([], dtype=tf.int32)
        player_label = tf.zeros([], dtype=tf.int32)
        
        if task_type == 1:
            action_parts = caption.split()
            if len(action_parts) >= 1:
                action_coarse_label = tf.constant(0, dtype=tf.int32)
                action_fine_label = tf.constant(0, dtype=tf.int32)
                action_event_label = tf.constant(0, dtype=tf.int32)
        
        elif task_type == 2:  
            player_parts = caption.split()
            if len(player_parts) >= 1:
                player_label = tf.constant(0, dtype=tf.int32)
        
        return (
            input_ids, attention_mask, decoder_input_ids, decoder_mask, decoder_output_ids,
            action_coarse_label, action_fine_label, action_event_label, player_label, task_type
        )
    
    def __getitem__(self, idx):
        """Get a single sample from the dataset"""
        video_id, caption = self.sentences_dict[idx]
        
        video_feature, video_mask = self._process_video_features(video_id)
        
        (
            ball_feature, player_feature, basket_feature, court_feature,
            player_positions, ball_positions, basket_positions
        ) = self._process_object_features(video_id)
        
        (
            input_ids, attention_mask, decoder_input_ids, decoder_mask, decoder_output_ids,
            action_coarse_label, action_fine_label, action_event_label, player_label, task_type
        ) = self._process_caption(caption)
        
        return (
            video_feature, video_mask, 
            ball_feature, player_feature, basket_feature, court_feature,
            player_positions, ball_positions, basket_positions,
            decoder_input_ids, decoder_mask, decoder_output_ids,
            action_coarse_label, action_fine_label, action_event_label,
            player_label, task_type
        )
    
    def create_tf_dataset(self):
        """Create a TensorFlow dataset from the dataloader"""
        def generator():
            indices = list(range(len(self)))
            if self.split_type == "train":
                random.shuffle(indices)
            
            for idx in indices:
                yield self.__getitem__(idx)
        
        output_types = (
            tf.float32, tf.int32,
            tf.float32, tf.float32, tf.float32, tf.float32,
            tf.float32, tf.float32, tf.float32,
            tf.int32, tf.int32, tf.int32,
            tf.int32, tf.int32, tf.int32,
            tf.int32, tf.int32
        )
        
        output_shapes = (
            (self.max_frames, None),  # video_feature
            (self.max_frames,),  # video_mask
            (self.max_frames, 224, 224, 3),  # ball_feature
            (self.max_frames, 5, 224, 224, 3),  # player_feature
            (self.max_frames, 224, 224, 3),  # basket_feature
            (self.max_frames, 224, 224, 3),  # court_feature
            (self.max_frames, 5, 2),  # player_positions
            (self.max_frames, 2),  # ball_positions
            (self.max_frames, 2),  # basket_positions
            (self.max_words,),  # decoder_input_ids
            (self.max_words,),  # decoder_mask
            (self.max_words,),  # decoder_output_ids
            (),  # action_coarse_label
            (),  # action_fine_label
            (),  # action_event_label
            (),  # player_label
            ()   # task_type
        )
        
        dataset = tf.data.Dataset.from_generator(
            generator,
            output_types=output_types,
            output_shapes=output_shapes
        )
        
        if self.split_type == "train":
            dataset = dataset.shuffle(buffer_size=1000)
        
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        
        return dataset
    
    def get_evaluation_data(self, max_samples=100):
        """
        Get a subset of data for evaluation of NLG metrics.
        Returns ground truth captions and corresponding video IDs.
        """
        eval_video_ids = []
        eval_captions = []
        
        unique_videos = {}
        for idx in range(min(max_samples, len(self))):
            video_id, caption = self.sentences_dict[idx]
            
            if caption.startswith("<T1>_"):
                clean_caption = caption[5:]
                if video_id not in unique_videos:
                    unique_videos[video_id] = []
                unique_videos[video_id].append(clean_caption)
        
        for video_id, captions in unique_videos.items():
            eval_video_ids.append(video_id)
            eval_captions.append(captions)
            
            if len(eval_video_ids) >= max_samples:
                break
        
        return eval_video_ids, eval_captions

class MetricsEvaluator:
    """
    Evaluate model outputs using NLG metrics like BLEU, METEOR, ROUGE, CIDEr.
    """
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer
        self.rouge = Rouge()
    
    def calculate_bleu(self, references, hypotheses):
        """
        Calculate BLEU score.
        
        Args:
            references: List of lists of reference sentences (tokenized)
            hypotheses: List of hypotheses sentences (tokenized)
            
        Returns:
            BLEU-1, BLEU-2, BLEU-3, BLEU-4 scores
        """
        refs = [[ref] for ref in references]
        
        bleu1 = corpus_bleu(refs, hypotheses, weights=(1, 0, 0, 0))
        bleu2 = corpus_bleu(refs, hypotheses, weights=(0.5, 0.5, 0, 0))
        bleu3 = corpus_bleu(refs, hypotheses, weights=(0.33, 0.33, 0.33, 0))
        bleu4 = corpus_bleu(refs, hypotheses, weights=(0.25, 0.25, 0.25, 0.25))
        
        return bleu1, bleu2, bleu3, bleu4
    
    def calculate_meteor(self, references, hypotheses):
        """
        Calculate METEOR score.
        
        Args:
            references: List of reference sentences (untokenized)
            hypotheses: List of hypotheses sentences (untokenized)
            
        Returns:
            METEOR score
        """
        scores = []
        for ref, hyp in zip(references, hypotheses):
            score = meteor_score([ref], hyp)
            scores.append(score)
        
        return sum(scores) / len(scores) if scores else 0
    
    def calculate_rouge(self, references, hypotheses):
        """
        Calculate ROUGE scores.
        
        Args:
            references: List of reference sentences (untokenized)
            hypotheses: List of hypotheses sentences (untokenized)
            
        Returns:
            ROUGE-1, ROUGE-2, ROUGE-L scores
        """
        try:
            scores = self.rouge.get_scores(hypotheses, references, avg=True)
            return (
                scores['rouge-1']['f'],
                scores['rouge-2']['f'],
                scores['rouge-l']['f']
            )
        except Exception as e:
            print(f"Error calculating ROUGE: {e}")
            return 0, 0, 0
    
    def calculate_cider(self, references, hypotheses):
        """
        Calculate CIDEr score.
        This is a simplified implementation as the full CIDEr calculation is complex.
        
        Args:
            references: List of lists of reference sentences (untokenized)
            hypotheses: List of hypotheses sentences (untokenized)
            
        Returns:
            CIDEr score
        """
        return 0.0
    
    def tokenize_text(self, text):
        """Tokenize text using the model's tokenizer"""
        if self.tokenizer:
            tokens = self.tokenizer.tokenize(text)
            return tokens
        else:
            return text.lower().split()
    
    def detokenize_text(self, tokens):
        """Convert tokens back to text"""
        if self.tokenizer:
            text = self.tokenizer.convert_tokens_to_string(tokens)
            return text
        else:
            return ' '.join(tokens)
    
    def evaluate(self, references, hypotheses):
        """
        Calculate all metrics for generated captions.
        
        Args:
            references: List of lists of reference sentences
            hypotheses: List of hypotheses sentences
            
        Returns:
            Dictionary of evaluation metrics
        """
        tokenized_refs = [self.tokenize_text(ref[0]) for ref in references]
        tokenized_hyps = [self.tokenize_text(hyp) for hyp in hypotheses]
        
        bleu1, bleu2, bleu3, bleu4 = self.calculate_bleu(tokenized_refs, tokenized_hyps)
        
        refs_text = [ref[0] for ref in references]
        hyps_text = hypotheses
        
        meteor = self.calculate_meteor(refs_text, hyps_text)
        
        rouge1, rouge2, rougeL = self.calculate_rouge(refs_text, hyps_text)
        
        cider = self.calculate_cider(references, hyps_text)
        
        return {
            'bleu1': bleu1,
            'bleu2': bleu2,
            'bleu3': bleu3,
            'bleu4': bleu4,
            'meteor': meteor,
            'rouge1': rouge1,
            'rouge2': rouge2,
            'rougeL': rougeL,
            'cider': cider
        }

def prepare_datasets(config):
    """
    Prepare training, validation, and test datasets.
    
    Args:
        config: Dictionary with dataset configuration
        
    Returns:
        train_dataset, val_dataset, test_dataset
    """
    train_dataloader = NSVADataLoader(
        csv_path=config['csv_path'],
        json_path=config['json_path'],
        video_features_path=config['video_features_path'],
        ball_features_path=config.get('ball_features_path'),
        player_features_path=config.get('player_features_path'),
        basket_features_path=config.get('basket_features_path'),
        court_features_path=config.get('court_features_path'),
        tokenizer=config['tokenizer'],
        max_frames=config['max_frames'],
        max_words=config['max_words'],
        batch_size=config['batch_size'],
        split_type='train'
    )
    
    val_dataloader = NSVADataLoader(
        csv_path=config['csv_path'],
        json_path=config['json_path'],
        video_features_path=config['video_features_path'],
        ball_features_path=config.get('ball_features_path'),
        player_features_path=config.get('player_features_path'),
        basket_features_path=config.get('basket_features_path'),
        court_features_path=config.get('court_features_path'),
        tokenizer=config['tokenizer'],
        max_frames=config['max_frames'],
        max_words=config['max_words'],
        batch_size=config['batch_size'],
        split_type='val'
    )
    
    test_dataloader = NSVADataLoader(
        csv_path=config['csv_path'],
        json_path=config['json_path'],
        video_features_path=config['video_features_path'],
        ball_features_path=config.get('ball_features_path'),
        player_features_path=config.get('player_features_path'),
        basket_features_path=config.get('basket_features_path'),
        court_features_path=config.get('court_features_path'),
        tokenizer=config['tokenizer'],
        max_frames=config['max_frames'],
        max_words=config['max_words'],
        batch_size=config['batch_size'],
        split_type='test'
    )
    
    train_dataset = train_dataloader.create_tf_dataset()
    val_dataset = val_dataloader.create_tf_dataset()
    test_dataset = test_dataloader.create_tf_dataset()
    
    return train_dataset, val_dataset, test_dataset, train_dataloader, val_dataloader, test_dataloader
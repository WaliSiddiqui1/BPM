import tensorflow as tf
import argparse
import os
import json
import time
import numpy as np
from transformers import BertTokenizer
from keras.mixed_precision import set_global_policy

from models.univl_model import SportsVideoUnderstandingModel
from training.univl_training import SportsCaptioningTrainer
from dataloaders.univl_dataloader import prepare_datasets, MetricsEvaluator

def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate NBA Sports Video Understanding model")
    
    parser.add_argument("--csv_path", type=str, required=True, help="Path to csv file with video metadata")
    parser.add_argument("--json_path", type=str, required=True, help="Path to json file with caption data")
    parser.add_argument("--video_features_path", type=str, required=True, help="Path to video features")
    parser.add_argument("--ball_features_path", type=str, default=None, help="Path to ball features")
    parser.add_argument("--player_features_path", type=str, default=None, help="Path to player features")
    parser.add_argument("--basket_features_path", type=str, default=None, help="Path to basket features")
    parser.add_argument("--court_features_path", type=str, default=None, help="Path to court features")
    
    parser.add_argument("--embed_dim", type=int, default=768, help="Embedding dimension")
    parser.add_argument("--max_frames", type=int, default=100, help="Maximum number of frames")
    parser.add_argument("--max_words", type=int, default=30, help="Maximum number of words")
    parser.add_argument("--timesformer_depth", type=int, default=12, help="Depth of TimeSformer")
    parser.add_argument("--visual_encoder_layers", type=int, default=6, help="Number of visual encoder layers")
    parser.add_argument("--cross_encoder_layers", type=int, default=2, help="Number of cross encoder layers")
    parser.add_argument("--decoder_layers", type=int, default=3, help="Number of decoder layers")
    parser.add_argument("--num_heads", type=int, default=12, help="Number of attention heads")
    
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=10000, help="Number of warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm")
    parser.add_argument("--mixed_precision", action="store_true", help="Use mixed precision training")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints", help="Path to save checkpoints")
    parser.add_argument("--log_freq", type=int, default=100, help="Logging frequency in steps")
    
    parser.add_argument("--mode", type=str, choices=["train", "evaluate", "predict"], required=True, 
                        help="Mode: train, evaluate, or predict")
    parser.add_argument("--load_checkpoint", action="store_true", help="Load from checkpoint if available")
    parser.add_argument("--use_tpu", action="store_true", help="Use TPU for training")
    parser.add_argument("--tpu_name", type=str, default=None, help="Name of TPU to use")
    parser.add_argument("--devices", type=int, default=1, help="Number of devices to use")
    
    args = parser.parse_args()
    return args

def init_model(args, strategy, tokenizer):
    """Initialize model with distribution strategy"""
    with strategy.scope():
        model = SportsVideoUnderstandingModel(
            vocab_size=tokenizer.vocab_size,
            max_frames=args.max_frames,
            max_words=args.max_words,
            embed_dim=args.embed_dim,
            timesformer_depth=args.timesformer_depth,
            visual_encoder_layers=args.visual_encoder_layers,
            cross_encoder_layers=args.cross_encoder_layers,
            decoder_layers=args.decoder_layers,
            num_heads=args.num_heads,
            dropout_rate=0.1
        )
        
        dummy_video = tf.zeros((1, args.max_frames, 224, 224, 3), dtype=tf.float32)
        dummy_video_mask = tf.ones((1, args.max_frames), dtype=tf.int32)
        dummy_ball = tf.zeros((1, args.max_frames, 224, 224, 3), dtype=tf.float32)
        dummy_player = tf.zeros((1, args.max_frames, 5, 224, 224, 3), dtype=tf.float32)
        dummy_basket = tf.zeros((1, args.max_frames, 224, 224, 3), dtype=tf.float32)
        dummy_court = tf.zeros((1, args.max_frames, 224, 224, 3), dtype=tf.float32)
        dummy_player_pos = tf.zeros((1, args.max_frames, 5, 2), dtype=tf.float32)
        dummy_ball_pos = tf.zeros((1, args.max_frames, 2), dtype=tf.float32)
        dummy_basket_pos = tf.zeros((1, args.max_frames, 2), dtype=tf.float32)
        dummy_decoder_input = tf.zeros((1, args.max_words), dtype=tf.int32)
        dummy_decoder_mask = tf.zeros((1, args.max_words), dtype=tf.int32)
        
        _ = model((
            dummy_video, dummy_video_mask, 
            dummy_ball, dummy_player, dummy_basket, dummy_court,
            dummy_player_pos, dummy_ball_pos, dummy_basket_pos,
            dummy_decoder_input, dummy_decoder_mask
        ), training=False)
        
        print(f"Model initialized with {sum(np.prod(v.get_shape()) for v in model.trainable_variables):,} parameters")
        
    return model

def train(args, strategy, model, tokenizer):
    """Train the model"""
    # Prepare datasets
    config = {
        'csv_path': args.csv_path,
        'json_path': args.json_path,
        'video_features_path': args.video_features_path,
        'ball_features_path': args.ball_features_path,
        'player_features_path': args.player_features_path,
        'basket_features_path': args.basket_features_path,
        'court_features_path': args.court_features_path,
        'tokenizer': tokenizer,
        'max_frames': args.max_frames,
        'max_words': args.max_words,
        'batch_size': args.batch_size
    }
    
    train_dataset, val_dataset, _, train_loader, val_loader, _ = prepare_datasets(config)
    
    # Create trainer
    trainer = SportsCaptioningTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        checkpoint_path=args.checkpoint_path,
        log_freq=args.log_freq
    )
    
    # Load checkpoint if requested
    if args.load_checkpoint:
        trainer.restore_checkpoint()
    
    # Train the model
    trainer.train(args.epochs)
    
    # Final evaluation
    results = trainer.evaluate()
    
    # Print results
    print("\nFinal Evaluation Results:")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")
    
    # Save results
    with open(os.path.join(args.checkpoint_path, "final_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    return results

def evaluate(args, model, tokenizer):
    """Evaluate the model"""
    # Prepare datasets
    config = {
        'csv_path': args.csv_path,
        'json_path': args.json_path,
        'video_features_path': args.video_features_path,
        'ball_features_path': args.ball_features_path,
        'player_features_path': args.player_features_path,
        'basket_features_path': args.basket_features_path,
        'court_features_path': args.court_features_path,
        'tokenizer': tokenizer,
        'max_frames': args.max_frames,
        'max_words': args.max_words,
        'batch_size': args.batch_size
    }
    
    _, _, test_dataset, _, _, test_loader = prepare_datasets(config)
    
    trainer = SportsCaptioningTrainer(
        model=model,
        train_dataset=None,
        val_dataset=test_dataset,
        learning_rate=args.learning_rate,
        checkpoint_path=args.checkpoint_path,
        log_freq=args.log_freq
    )
    
    trainer.restore_checkpoint()
    
    results = trainer.evaluate()
    
    print("\nGenerating captions for NLG metrics evaluation...")
    eval_video_ids, eval_references = test_loader.get_evaluation_data(max_samples=100)
    
    hypotheses = []
    for video_id in eval_video_ids:
        video_feature, video_mask = test_loader._process_video_features(video_id)
        video_feature = tf.expand_dims(tf.convert_to_tensor(video_feature), 0)
        video_mask = tf.expand_dims(tf.convert_to_tensor(video_mask), 0)
        
        caption = trainer.generate_caption(video_feature, tokenizer)
        hypotheses.append(caption)
    
    evaluator = MetricsEvaluator(tokenizer)
    nlg_results = evaluator.evaluate(eval_references, hypotheses)
    
    print("\nTest Evaluation Results:")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")
    
    print("\nNLG Metrics:")
    for k, v in nlg_results.items():
        print(f"{k}: {v:.4f}")
    
    all_results = {**results, **nlg_results}
    with open(os.path.join(args.checkpoint_path, "test_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    
    return all_results

def predict(args, model, tokenizer):
    """Run prediction on a single video"""
    config = {
        'csv_path': args.csv_path,
        'json_path': args.json_path,
        'video_features_path': args.video_features_path,
        'ball_features_path': args.ball_features_path,
        'player_features_path': args.player_features_path,
        'basket_features_path': args.basket_features_path,
        'court_features_path': args.court_features_path,
        'tokenizer': tokenizer,
        'max_frames': args.max_frames,
        'max_words': args.max_words,
        'batch_size': 1,
        'split_type': 'test'
    }
    
    _, _, _, _, _, test_loader = prepare_datasets(config)
    
    trainer = SportsCaptioningTrainer(
        model=model,
        train_dataset=None,
        val_dataset=None,
        learning_rate=args.learning_rate,
        checkpoint_path=args.checkpoint_path
    )
    
    trainer.restore_checkpoint()
    
    eval_video_ids, eval_references = test_loader.get_evaluation_data(max_samples=5)
    
    print("\nGenerating captions for sample videos:")
    for i, video_id in enumerate(eval_video_ids):
        video_feature, video_mask = test_loader._process_video_features(video_id)
        video_feature = tf.expand_dims(tf.convert_to_tensor(video_feature), 0)
        video_mask = tf.expand_dims(tf.convert_to_tensor(video_mask), 0)
        
        start_time = time.time()
        caption = trainer.generate_caption(video_feature, tokenizer)
        generation_time = time.time() - start_time
        
        print(f"\nVideo {i+1} (ID: {video_id}):")
        print(f"Generated caption: {caption}")
        print(f"Reference caption: {eval_references[i][0]}")
        print(f"Generation time: {generation_time:.2f}s")
    
    print("\nPrediction completed.")

def main():
    args = parse_args()
    
    if args.mixed_precision:
        set_global_policy('mixed_float16')
        print("Using mixed precision training")
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    model = init_model(args, tokenizer)
    
    if args.mode == "train":
        train(args, model, tokenizer)
    elif args.mode == "evaluate":
        evaluate(args, model, tokenizer)
    elif args.mode == "predict":
        predict(args, model, tokenizer)

if __name__ == "__main__":
    main()
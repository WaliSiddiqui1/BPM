import tensorflow as tf
from tensorflow import keras
import numpy as np
import math

class TimeSformerModule(keras.layers.Layer):
    """
    TensorFlow implementation of TimeSformer for video understanding.
    """
    def __init__(self, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, 
                 max_frames=100, patch_size=16, img_size=224, drop_rate=0.1, **kwargs):
        super(TimeSformerModule, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.max_frames = max_frames
        self.patch_size = patch_size
        self.img_size = img_size
        self.drop_rate = drop_rate
        
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.patch_embed = keras.layers.Conv2D(
            filters=embed_dim, 
            kernel_size=patch_size, 
            strides=patch_size,
            padding='valid',
            name='patch_embed'
        )
        
        # Position embeddings
        self.pos_embed = self.add_weight(
            shape=(1, self.num_patches + 1, embed_dim),
            initializer=tf.random_normal_initializer(stddev=0.02),
            trainable=True,
            name='pos_embed'
        )
        
        # Time embeddings for temporal attention
        self.time_embed = self.add_weight(
            shape=(1, max_frames, embed_dim),
            initializer=tf.random_normal_initializer(stddev=0.02),
            trainable=True,
            name='time_embed'
        )
        
        # CLS token embedding
        self.cls_token = self.add_weight(
            shape=(1, 1, embed_dim),
            initializer=tf.random_normal_initializer(stddev=0.02),
            trainable=True,
            name='cls_token'
        )
        
        # Dropout
        self.pos_drop = keras.layers.Dropout(drop_rate)
        
        # Transformer blocks with divided space-time attention
        self.blocks = [
            DividedSpaceTimeBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                name=f'block_{i}'
            ) for i in range(depth)
        ]
        
        # Layer norm
        self.norm = keras.layers.LayerNormalization(epsilon=1e-6, name='encoder_norm')
        
    def call(self, x, training=False):
        """
        Input: x - shape [B, T, H, W, C]
        Output: shape [B, T, D] where D is embed_dim
        """
        batch_size, num_frames = tf.shape(x)[0], tf.shape(x)[1]
        
        # Reshape to process each frame with patch embedding
        x = tf.reshape(x, [-1, self.img_size, self.img_size, 3])
        x = self.patch_embed(x)  # [B*T, H/P, W/P, D]
        
        # Reshape to sequence
        x = tf.reshape(x, [-1, self.num_patches, self.embed_dim])  # [B*T, N, D]
        
        # Add cls token to each frame sequence
        cls_tokens = tf.broadcast_to(
            self.cls_token, [batch_size * num_frames, 1, self.embed_dim]
        )
        x = tf.concat([cls_tokens, x], axis=1)  # [B*T, N+1, D]
        
        # Add position embeddings
        x = x + self.pos_embed
        
        # Reshape to separate time dimension for space-time processing
        x = tf.reshape(x, [batch_size, num_frames, self.num_patches + 1, self.embed_dim])
        
        # Add time embeddings
        time_embed = self.time_embed[:, :num_frames, :]
        time_embed = tf.broadcast_to(time_embed[:, :, None, :], 
                                     [batch_size, num_frames, self.num_patches + 1, self.embed_dim])
        x = x + time_embed
        
        # Reshape back to sequence for processing
        x = tf.reshape(x, [batch_size * num_frames, self.num_patches + 1, self.embed_dim])
        
        # Apply dropout
        x = self.pos_drop(x, training=training)
        
        # Process through transformer blocks
        for block in self.blocks:
            x = block(x, batch_size, num_frames, training=training)
        
        # Apply final norm
        x = self.norm(x)
        
        # Extract CLS tokens and reshape to [B, T, D]
        cls_output = x[:, 0]
        cls_output = tf.reshape(cls_output, [batch_size, num_frames, self.embed_dim])
        
        return cls_output

class DividedSpaceTimeBlock(keras.layers.Layer):
    """
    Transformer block with divided space-time attention as described in TimeSformer.
    First applies temporal attention, then spatial attention.
    """
    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop=0.0, **kwargs):
        super(DividedSpaceTimeBlock, self).__init__(**kwargs)
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        
        # Temporal attention
        self.temporal_norm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.temporal_attn = MultiHeadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout_rate=drop
        )
        
        # Spatial attention
        self.spatial_norm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.spatial_attn = MultiHeadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout_rate=drop
        )
        
        # MLP
        self.norm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = keras.Sequential([
            keras.layers.Dense(mlp_hidden_dim, activation=tf.nn.gelu),
            keras.layers.Dropout(drop),
            keras.layers.Dense(dim),
            keras.layers.Dropout(drop)
        ])
        
    def call(self, x, batch_size, num_frames, training=False):
        # Input: [B*T, N, D]
        # First process temporal attention (across frames)
        num_patches = tf.shape(x)[1]
        
        # Reshape to [B, T, N, D] to process temporal dimension
        x_t = tf.reshape(x, [batch_size, num_frames, num_patches, self.dim])
        
        # Transpose to [B, N, T, D] to do attention on temporal dim
        x_t = tf.transpose(x_t, [0, 2, 1, 3])
        x_t = tf.reshape(x_t, [batch_size * num_patches, num_frames, self.dim])
        
        # Apply temporal attention
        x_t_norm = self.temporal_norm1(x_t)
        x_t = x_t + self.temporal_attn(x_t_norm, x_t_norm, x_t_norm, training=training)
        
        # Reshape back to [B*T, N, D]
        x_t = tf.reshape(x_t, [batch_size, num_patches, num_frames, self.dim])
        x_t = tf.transpose(x_t, [0, 2, 1, 3])
        x_t = tf.reshape(x_t, [batch_size * num_frames, num_patches, self.dim])
        
        # Process spatial attention
        x_s_norm = self.spatial_norm1(x_t)
        x_s = x_t + self.spatial_attn(x_s_norm, x_s_norm, x_s_norm, training=training)
        
        # Apply MLP
        x_norm = self.norm2(x_s)
        x_mlp = self.mlp(x_norm, training=training)
        x_out = x_s + x_mlp
        
        return x_out

class MultiHeadAttention(keras.layers.Layer):
    """
    Custom implementation of multi-head attention with explicit q, k, v projections.
    """
    def __init__(self, embed_dim, num_heads, dropout_rate=0.0, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        if self.head_dim * num_heads != embed_dim:
            raise ValueError(f"embed_dim {embed_dim} not divisible by num_heads {num_heads}")
        
        # Projections for query, key, value
        self.q_proj = keras.layers.Dense(embed_dim)
        self.k_proj = keras.layers.Dense(embed_dim)
        self.v_proj = keras.layers.Dense(embed_dim)
        
        # Output projection
        self.out_proj = keras.layers.Dense(embed_dim)
        
        # Dropout
        self.dropout = keras.layers.Dropout(dropout_rate)
        
    def call(self, query, key, value, mask=None, training=False):
        batch_size = tf.shape(query)[0]
        seq_len = tf.shape(query)[1]
        
        # Linear projections and reshape for multi-head attention
        q = self.q_proj(query)  # [B, L, D]
        k = self.k_proj(key)    # [B, L, D]
        v = self.v_proj(value)  # [B, L, D]
        
        # Reshape to [B, L, H, D/H]
        q = tf.reshape(q, [batch_size, seq_len, self.num_heads, self.head_dim])
        k = tf.reshape(k, [batch_size, seq_len, self.num_heads, self.head_dim])
        v = tf.reshape(v, [batch_size, seq_len, self.num_heads, self.head_dim])
        
        # Transpose to [B, H, L, D/H]
        q = tf.transpose(q, [0, 2, 1, 3])
        k = tf.transpose(k, [0, 2, 1, 3])
        v = tf.transpose(v, [0, 2, 1, 3])
        
        # Compute attention scores
        scores = tf.matmul(q, k, transpose_b=True)  # [B, H, L, L]
        scores = scores / tf.math.sqrt(tf.cast(self.head_dim, tf.float32))
        
        # Apply mask if provided
        if mask is not None:
            scores += (mask * -1e9)
        
        # Apply softmax and dropout
        attention_weights = tf.nn.softmax(scores, axis=-1)
        attention_weights = self.dropout(attention_weights, training=training)
        
        # Compute output
        output = tf.matmul(attention_weights, v)  # [B, H, L, D/H]
        output = tf.transpose(output, [0, 2, 1, 3])  # [B, L, H, D/H]
        output = tf.reshape(output, [batch_size, seq_len, self.embed_dim])  # [B, L, D]
        
        # Apply output projection
        output = self.out_proj(output)
        
        return output

class ObjectDetectionModule(keras.layers.Layer):
    """
    Module for processing detected objects (ball, basket, player with ball).
    Uses a Vision Transformer (ViT) to extract features from object crops.
    """
    def __init__(self, embed_dim=768, **kwargs):
        super(ObjectDetectionModule, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        
        # We'll use a simplified ViT here - in production you'd use a pre-trained one
        self.vit = keras.Sequential([
            keras.layers.LayerNormalization(epsilon=1e-6),
            keras.layers.Dense(embed_dim * 4, activation=tf.nn.gelu),
            keras.layers.Dense(embed_dim),
            keras.layers.LayerNormalization(epsilon=1e-6)
        ])
    
    def call(self, object_crops, training=False):
        """
        Process detected object crops with ViT
        Input: object_crops - shape [B, N, H, W, C] where N is number of detected objects
        Output: shape [B, N, D] where D is embed_dim
        """
        batch_size = tf.shape(object_crops)[0]
        num_objects = tf.shape(object_crops)[1]
        
        # Reshape to process each crop
        object_crops = tf.reshape(object_crops, [-1, tf.shape(object_crops)[2], 
                                             tf.shape(object_crops)[3], 
                                             tf.shape(object_crops)[4]])
        
        # Apply global average pooling to get a fixed-size representation
        pooled_crops = tf.reduce_mean(object_crops, axis=[1, 2])  # [B*N, C]
        
        # Process with ViT
        features = self.vit(pooled_crops, training=training)  # [B*N, D]
        
        # Reshape back to batch
        features = tf.reshape(features, [batch_size, num_objects, self.embed_dim])
        
        return features

class PositionAwareModule(keras.layers.Layer):
    """
    Module for position awareness using court-line segmentation.
    Combines detected players, ball and basket with court-line segmentation.
    """
    def __init__(self, embed_dim=768, **kwargs):
        super(PositionAwareModule, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        
        # Court-line feature extractor
        self.court_encoder = keras.Sequential([
            keras.layers.Conv2D(64, kernel_size=3, strides=2, padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.Activation('relu'),
            keras.layers.Conv2D(128, kernel_size=3, strides=2, padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.Activation('relu'),
            keras.layers.Conv2D(256, kernel_size=3, strides=2, padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.Activation('relu'),
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dense(embed_dim)
        ])
        
        # Fusion layer to combine court features with object positions
        self.fusion = keras.layers.Dense(embed_dim)
    
    def call(self, court_images, player_pos, ball_pos, basket_pos, training=False):
        """
        Process court-line segmentation and object positions
        Inputs:
            court_images - [B, H, W, C]
            player_pos - [B, N, 2] (x,y coordinates of players)
            ball_pos - [B, 2] (x,y coordinates of ball)
            basket_pos - [B, 2] (x,y coordinates of basket)
        Output: [B, D]
        """
        # Extract court features
        court_features = self.court_encoder(court_images, training=training)  # [B, D]
        
        # Create position encoding for objects
        batch_size = tf.shape(player_pos)[0]
        num_players = tf.shape(player_pos)[1]
        
        # Concat all positions
        player_pos_flat = tf.reshape(player_pos, [batch_size, -1])  # [B, N*2]
        positions = tf.concat([player_pos_flat, ball_pos, basket_pos], axis=1)  # [B, N*2+4]
        
        # Project to embedding space
        pos_embedding = keras.layers.Dense(self.embed_dim)(positions)  # [B, D]
        
        # Fuse court features with position embedding
        fused_features = court_features + pos_embedding
        output = self.fusion(fused_features)  # [B, D]
        
        return output

class VisualEncoder(keras.layers.Layer):
    """
    Transformer encoder for fine-grained visual features.
    Takes features from ball, player with ball, basket, and position-aware module.
    """
    def __init__(self, embed_dim=768, num_layers=6, num_heads=12, **kwargs):
        super(VisualEncoder, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        
        # Create transformer blocks
        self.transformer_blocks = [
            keras.layers.TransformerBlock(
                num_heads=num_heads,
                embed_dim=embed_dim,
                ff_dim=embed_dim * 4,
                dropout=0.1
            ) for _ in range(num_layers)
        ]
        
        # Final layer norm
        self.layer_norm = keras.layers.LayerNormalization(epsilon=1e-6)
    
    def call(self, features, mask=None, training=False):
        """
        Process sequence of fine-grained features
        Input: features - shape [B, L, D] where L is sequence length
              mask - shape [B, L] attention mask
        Output: shape [B, L, D]
        """
        x = features
        
        # Process through transformer blocks
        for block in self.transformer_blocks:
            x = block(x, mask=mask, training=training)
        
        # Apply final norm
        x = self.layer_norm(x)
        
        return x

class CrossEncoder(keras.layers.Layer):
    """
    Cross-modal encoder for attending between coarse and fine-grained features.
    """
    def __init__(self, embed_dim=768, num_layers=3, num_heads=12, **kwargs):
        super(CrossEncoder, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        
        # Cross-attention transformer blocks
        self.cross_blocks = [
            CrossAttentionBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ff_dim=embed_dim * 4,
                dropout=0.1
            ) for _ in range(num_layers)
        ]
        
        # Final layer norm
        self.layer_norm = keras.layers.LayerNormalization(epsilon=1e-6)
    
    def call(self, coarse_features, fine_features, coarse_mask=None, fine_mask=None, training=False):
        """
        Cross-attention between coarse and fine-grained features
        Inputs:
            coarse_features - [B, L1, D] from TimeSformer
            fine_features - [B, L2, D] from Visual Encoder
            coarse_mask - [B, L1] mask for coarse features
            fine_mask - [B, L2] mask for fine features
        Output: [B, L1+L2, D] concatenated features with cross-attention
        """
        # Concatenate features
        concat_features = tf.concat([coarse_features, fine_features], axis=1)
        
        # Concatenate masks if provided
        mask = None
        if coarse_mask is not None and fine_mask is not None:
            mask = tf.concat([coarse_mask, fine_mask], axis=1)
        
        # Process through cross-attention blocks
        x = concat_features
        for block in self.cross_blocks:
            x = block(x, mask=mask, training=training)
        
        # Apply final norm
        x = self.layer_norm(x)
        
        return x

class CrossAttentionBlock(keras.layers.Layer):
    """
    Transformer block with cross-attention mechanism.
    """
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1, **kwargs):
        super(CrossAttentionBlock, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        
        # Self-attention
        self.attn = keras.layers.MultiHeadAttention(
            num_heads=num_heads, 
            key_dim=embed_dim // num_heads,
            dropout=dropout
        )
        
        # Feed-forward network
        self.ffn = keras.Sequential([
            keras.layers.Dense(ff_dim, activation=tf.nn.gelu),
            keras.layers.Dropout(dropout),
            keras.layers.Dense(embed_dim),
            keras.layers.Dropout(dropout)
        ])
        
        # Layer norms
        self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        
        # Dropout
        self.dropout1 = keras.layers.Dropout(dropout)
        self.dropout2 = keras.layers.Dropout(dropout)
    
    def call(self, inputs, mask=None, training=False):
        # Self-attention with residual connection
        attn_output = self.attn(
            query=inputs,
            key=inputs,
            value=inputs,
            attention_mask=mask,
            training=training
        )
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        # Feed-forward with residual connection
        ffn_output = self.ffn(out1, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class DecoderModule(keras.layers.Layer):
    """
    Transformer decoder for caption generation.
    """
    def __init__(self, vocab_size, embed_dim=768, num_layers=3, num_heads=12, **kwargs):
        super(DecoderModule, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        
        # Word embeddings
        self.word_embeddings = keras.layers.Embedding(vocab_size, embed_dim)
        
        # Decoder blocks
        self.decoder_blocks = [
            DecoderBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ff_dim=embed_dim * 4,
                dropout=0.1
            ) for _ in range(num_layers)
        ]
        
        # Output projection
        self.output_projection = keras.layers.Dense(vocab_size)
        
        # Final layer norm
        self.layer_norm = keras.layers.LayerNormalization(epsilon=1e-6)
    
    def call(self, input_ids, encoder_output, input_mask=None, encoder_mask=None, training=False):
        """
        Decode caption from encoder output
        Inputs:
            input_ids - [B, T] caption token ids
            encoder_output - [B, L, D] encoder output
            input_mask - [B, T] caption mask
            encoder_mask - [B, L] encoder output mask
        Output: [B, T, V] caption logits
        """
        # Get word embeddings
        seq_len = tf.shape(input_ids)[1]
        x = self.word_embeddings(input_ids)  # [B, T, D]
        
        # Add sinusoidal position embeddings
        positions = tf.range(start=0, limit=seq_len, delta=1)
        position_embeddings = get_sinusoidal_position_embeddings(
            positions, self.embed_dim)
        x = x + position_embeddings
        
        # Process through decoder blocks
        for block in self.decoder_blocks:
            x = block(
                x, 
                encoder_output, 
                self_attention_mask=create_look_ahead_mask(seq_len),
                padding_mask=input_mask,
                encoder_padding_mask=encoder_mask,
                training=training
            )
        
        # Apply final norm
        x = self.layer_norm(x)
        
        # Generate logits
        logits = self.output_projection(x)  # [B, T, V]
        
        return logits

class DecoderBlock(keras.layers.Layer):
    """
    Transformer decoder block with self-attention and cross-attention.
    """
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1, **kwargs):
        super(DecoderBlock, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        
        # Self-attention
        self.self_attn = keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads,
            dropout=dropout
        )
        
        # Cross-attention for encoder output
        self.cross_attn = keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads,
            dropout=dropout
        )
        
        # Feed-forward network
        self.ffn = keras.Sequential([
            keras.layers.Dense(ff_dim, activation=tf.nn.gelu),
            keras.layers.Dropout(dropout),
            keras.layers.Dense(embed_dim),
            keras.layers.Dropout(dropout)
        ])
        
        # Layer norms
        self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = keras.layers.LayerNormalization(epsilon=1e-6)
        
        # Dropout
        self.dropout1 = keras.layers.Dropout(dropout)
        self.dropout2 = keras.layers.Dropout(dropout)
        self.dropout3 = keras.layers.Dropout(dropout)
    
    def call(self, x, encoder_output, self_attention_mask=None, 
             padding_mask=None, encoder_padding_mask=None, training=False):
        """
        Process inputs through self-attention, cross-attention and feed-forward
        """
        # Self-attention with causal mask and padding mask
        combined_mask = None
        if self_attention_mask is not None:
            if padding_mask is not None:
                padding_mask = tf.cast(padding_mask[:, tf.newaxis, tf.newaxis, :], tf.float32)
                combined_mask = tf.minimum(self_attention_mask, padding_mask)
            else:
                combined_mask = self_attention_mask
                
        attn1 = self.self_attn(
            query=x,
            key=x,
            value=x,
            attention_mask=combined_mask,
            training=training
        )
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(x + attn1)
        
        # Cross-attention with encoder output
        encoder_padding_mask_expanded = None
        if encoder_padding_mask is not None:
            encoder_padding_mask_expanded = tf.cast(
                encoder_padding_mask[:, tf.newaxis, tf.newaxis, :], tf.float32)
        
        attn2 = self.cross_attn(
            query=out1,
            key=encoder_output,
            value=encoder_output,
            attention_mask=encoder_padding_mask_expanded,
            training=training
        )
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(out1 + attn2)
        
        # Feed-forward
        ffn_output = self.ffn(out2, training=training)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(out2 + ffn_output)
        
        return out3

def get_sinusoidal_position_embeddings(positions, embed_dim):
    """
    Generate sinusoidal position embeddings.
    """
    # Create an array of positions
    positions = tf.cast(positions, tf.float32)
    
    # Get the embedding dimension pairs for sine and cosine
    half_dim = embed_dim // 2
    
    # Apply log space to get the timescales
    log_timescale_increment = tf.math.log(10000.0) / tf.cast(half_dim - 1, tf.float32)
    inv_timescales = tf.exp(tf.range(half_dim, dtype=tf.float32) * -log_timescale_increment)
    
    # Calculate sine and cosine embeddings
    scaled_positions = positions[:, tf.newaxis] * inv_timescales[tf.newaxis, :]
    pos_embeddings = tf.concat([tf.sin(scaled_positions), tf.cos(scaled_positions)], axis=-1)
    
    # Handle odd embedding dimensions
    if embed_dim % 2 == 1:
        pos_embeddings = tf.pad(pos_embeddings, [[0, 0], [0, 1]])
    
    return pos_embeddings

def create_look_ahead_mask(size):
    """
    Create a look-ahead mask for transformer decoder.
    """
    # Create a lower triangular matrix (including the diagonal)
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    
    # Convert to float32 and expand dimensions for broadcasting
    mask = tf.cast(mask, tf.float32)
    
    # Tensor will be [1, 1, size, size]
    return mask[tf.newaxis, tf.newaxis, :, :]

class SportsVideoUnderstandingModel(keras.Model):
    """
    Complete model for sports video understanding based on the paper.
    Integrates TimeSformer, object detection features, position awareness,
    cross-encoding, and decoding for caption generation.
    """
    def __init__(
        self, 
        vocab_size,
        max_frames=100,
        max_words=30,
        embed_dim=768,
        timesformer_depth=12,
        visual_encoder_layers=6,
        cross_encoder_layers=2,
        decoder_layers=3,
        num_heads=12,
        dropout_rate=0.1,
        **kwargs
    ):
        super(SportsVideoUnderstandingModel, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.max_frames = max_frames
        self.max_words = max_words
        self.embed_dim = embed_dim
        
        # TimeSformer for coarse video understanding
        self.timesformer = TimeSformerModule(
            embed_dim=embed_dim,
            depth=timesformer_depth,
            num_heads=num_heads,
            max_frames=max_frames,
            drop_rate=dropout_rate
        )
        
        # Object detection modules for fine-grained features
        self.ball_detector = ObjectDetectionModule(embed_dim=embed_dim)
        self.player_detector = ObjectDetectionModule(embed_dim=embed_dim)
        self.basket_detector = ObjectDetectionModule(embed_dim=embed_dim)
        
        # Position-aware module for court understanding
        self.position_aware = PositionAwareModule(embed_dim=embed_dim)
        
        # Visual encoder for fine-grained features
        self.visual_encoder = VisualEncoder(
            embed_dim=embed_dim,
            num_layers=visual_encoder_layers,
            num_heads=num_heads
        )
        
        # Cross-encoder for feature fusion
        self.cross_encoder = CrossEncoder(
            embed_dim=embed_dim,
            num_layers=cross_encoder_layers,
            num_heads=num_heads
        )
        
        # Decoder for caption generation
        self.decoder = DecoderModule(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_layers=decoder_layers,
            num_heads=num_heads
        )
        
        # For retrieval and contrastive learning
        self.similarity_dense = keras.layers.Dense(embed_dim)
        
        # For action recognition
        self.action_coarse_classifier = keras.layers.Dense(14)  # coarse actions
        self.action_fine_classifier = keras.layers.Dense(124)   # fine actions
        self.action_event_classifier = keras.layers.Dense(24)   # event actions
        
        # For player identification
        self.player_identifier = keras.layers.Dense(184)  # total player identities
        
    def process_video_features(self, video, video_mask, training=False):
        """
        Process video features with TimeSformer
        Input: video - shape [B, T, H, W, C]
              video_mask - shape [B, T]
        Output: shape [B, T, D]
        """
        # Extract coarse features with TimeSformer
        coarse_features = self.timesformer(video, training=training)  # [B, T, D]
        
        return coarse_features
    
    def process_object_features(self, ball_crops, player_crops, basket_crops, 
                                court_images, player_positions, ball_positions, 
                                basket_positions, training=False):
        """
        Process fine-grained object features
        Inputs:
            ball_crops - [B, T, H, W, C]
            player_crops - [B, T, N, H, W, C] where N is number of players
            basket_crops - [B, T, H, W, C]
            court_images - [B, T, H, W, C]
            player_positions - [B, T, N, 2]
            ball_positions - [B, T, 2]
            basket_positions - [B, T, 2]
        Output: shape [B, T, D]
        """
        batch_size = tf.shape(ball_crops)[0]
        seq_len = tf.shape(ball_crops)[1]
        
        # Extract object features frame by frame
        ball_features_list = []
        player_features_list = []
        basket_features_list = []
        position_features_list = []
        
        for t in range(self.max_frames):
            if t < seq_len:
                # Extract ball features
                ball_frame = ball_crops[:, t]
                ball_feat = self.ball_detector(ball_frame, training=training)  # [B, D]
                
                # Extract player features
                player_frame = player_crops[:, t]
                player_feat = self.player_detector(player_frame, training=training)  # [B, N, D]
                player_feat = tf.reduce_mean(player_feat, axis=1)  # [B, D]
                
                # Extract basket features
                basket_frame = basket_crops[:, t]
                basket_feat = self.basket_detector(basket_frame, training=training)  # [B, D]
                
                # Extract position-aware features
                court_frame = court_images[:, t]
                player_pos = player_positions[:, t]
                ball_pos = ball_positions[:, t]
                basket_pos = basket_positions[:, t]
                position_feat = self.position_aware(
                    court_frame, player_pos, ball_pos, basket_pos, training=training)  # [B, D]
                
                ball_features_list.append(ball_feat)
                player_features_list.append(player_feat)
                basket_features_list.append(basket_feat)
                position_features_list.append(position_feat)
            else:
                # Add zero padding for remaining frames
                zeros = tf.zeros([batch_size, self.embed_dim], dtype=tf.float32)
                ball_features_list.append(zeros)
                player_features_list.append(zeros)
                basket_features_list.append(zeros)
                position_features_list.append(zeros)
        
        # Stack along time dimension
        ball_features = tf.stack(ball_features_list, axis=1)  # [B, T, D]
        player_features = tf.stack(player_features_list, axis=1)  # [B, T, D]
        basket_features = tf.stack(basket_features_list, axis=1)  # [B, T, D]
        position_features = tf.stack(position_features_list, axis=1)  # [B, T, D]
        
        # Concatenate object features for each frame
        object_features = tf.concat(
            [ball_features, player_features, basket_features, position_features], 
            axis=1)  # [B, 4*T, D]
        
        # Create mask for object features
        object_mask = tf.ones([batch_size, 4 * seq_len], dtype=tf.int32)
        if seq_len < self.max_frames:
            pad_len = 4 * (self.max_frames - seq_len)
            object_mask = tf.concat(
                [object_mask, tf.zeros([batch_size, pad_len], dtype=tf.int32)], 
                axis=1)
        
        # Process with visual encoder
        fine_features = self.visual_encoder(
            object_features, mask=object_mask, training=training)  # [B, 4*T, D]
        
        return fine_features, object_mask
    
    def fuse_features(self, coarse_features, fine_features, 
                      coarse_mask, fine_mask, training=False):
        """
        Fuse coarse and fine-grained features
        """
        # Apply cross-encoder
        fused_features = self.cross_encoder(
            coarse_features, fine_features, 
            coarse_mask=coarse_mask, fine_mask=fine_mask,
            training=training
        )
        
        return fused_features
    
    def generate_caption(self, fused_features, input_caption_ids, decoder_mask, training=False):
        """
        Generate caption from fused features
        """
        # Split fused features back to coarse and fine
        coarse_len = tf.shape(input_caption_ids)[1]
        encoder_output = fused_features[:, :coarse_len]
        
        # Create encoder mask from features
        encoder_mask = tf.ones([tf.shape(encoder_output)[0], coarse_len], dtype=tf.int32)
        
        # Generate caption logits
        caption_logits = self.decoder(
            input_caption_ids, 
            encoder_output, 
            input_mask=decoder_mask,
            encoder_mask=encoder_mask,
            training=training
        )
        
        return caption_logits
    
    def recognize_action(self, fused_features, training=False):
        """
        Recognize basketball actions from fused features
        Returns predictions at coarse, fine, and event levels
        """
        # Get sequence representation (use first token)
        sequence_representation = fused_features[:, 0]
        
        # Classification for different action levels
        coarse_logits = self.action_coarse_classifier(sequence_representation)
        fine_logits = self.action_fine_classifier(sequence_representation)
        event_logits = self.action_event_classifier(sequence_representation)
        
        return coarse_logits, fine_logits, event_logits
    
    def identify_players(self, fused_features, training=False):
        """
        Identify players from fused features
        """
        # Use sequence representation for player identification
        sequence_representation = fused_features[:, 0]
        
        # Player identification logits
        player_logits = self.player_identifier(sequence_representation)
        
        return player_logits
    
    def compute_similarity(self, video_features, text_features):
        """
        Compute similarity between video and text features for retrieval
        """
        # Normalize features
        video_features = tf.nn.l2_normalize(video_features, axis=-1)
        text_features = tf.nn.l2_normalize(text_features, axis=-1)
        
        # Compute dot product similarity
        similarity = tf.matmul(text_features, video_features, transpose_b=True)
        
        # Scale similarity
        similarity = similarity / 0.07  # Temperature scaling
        
        return similarity
    
    def call(self, inputs, training=False):
        """
        Forward pass through the model
        """
        # Unpack inputs
        (
            video, video_mask, ball_crops, player_crops, basket_crops,
            court_images, player_positions, ball_positions, basket_positions,
            input_caption_ids, decoder_mask
        ) = inputs
        
        # Process coarse video features
        coarse_features = self.process_video_features(video, video_mask, training=training)
        
        # Process fine-grained object features
        fine_features, fine_mask = self.process_object_features(
            ball_crops, player_crops, basket_crops,
            court_images, player_positions, ball_positions, basket_positions,
            training=training
        )
        
        # Fuse features
        fused_features = self.fuse_features(
            coarse_features, fine_features,
            coarse_mask=video_mask, fine_mask=fine_mask,
            training=training
        )
        
        # Generate caption
        caption_logits = self.generate_caption(
            fused_features, input_caption_ids, decoder_mask, training=training
        )
        
        # Recognize action
        coarse_action_logits, fine_action_logits, event_action_logits = self.recognize_action(
            fused_features, training=training
        )
        
        # Identify players
        player_logits = self.identify_players(fused_features, training=training)
        
        return {
            'caption_logits': caption_logits,
            'coarse_action_logits': coarse_action_logits,
            'fine_action_logits': fine_action_logits,
            'event_action_logits': event_action_logits,
            'player_logits': player_logits,
            'fused_features': fused_features
        }
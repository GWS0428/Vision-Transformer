"""Define the model."""

import tensorflow as tf

from MultiheadAttention import MultiHeadAttention


class Patch(tf.keras.layers.Layer):
    """coverts input images to patches"""
    def __init__(self, patch_size, **kwards):
        super(Patch, self).__init__(**kwards)
        self.patch_size = patch_size
    
    def call(self, inputs):
        patches = self.convert_to_patches(inputs, self.patch_size)
        return patches
    
    def convert_to_patches(self, images, patch_size):
        """convert batch of images to batch of flattened patches"""
        # shape of images : (batch_size, width, height, channels)
        # shape of output : (batch_size, no. of flattened patches, patch_size)
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(images=images, 
                                           sizes=[1, patch_size, patch_size, 1], 
                                           strides=[1, patch_size, patch_size, 1], 
                                           rates=[1, 1, 1, 1], 
                                           padding='VALID')
        flattened_size = tf.shape(patches)[-1]
        patches = tf.reshape(patches, shape=[batch_size, -1, flattened_size])
    
        return patches
    
    
class Projection(tf.keras.layers.Layer):
    """linear projection of flattened patches"""
    def __init__(self, d_model, **kwards):
        super(Projection, self).__init__(**kwards)
        self.d_model = d_model
        self.another = tf.keras.layers.Dense(units=d_model)
        self.project = tf.keras.layers.Dense(units=d_model)
        self.cls_token = self.add_weight(name='class token',
                                        shape=(1, 1, d_model),
                                        initializer=tf.initializers.RandomNormal(),
                                        trainable=True)
        
    def call(self, inputs):
        cls_token = tf.tile(self.cls_token, [tf.shape(inputs)[0], 1, 1])
        inputs = self.another(inputs)
        inputs = self.project(inputs)
        return tf.concat([inputs, cls_token], axis=1)
    
    
class Pos_embedding(tf.keras.layers.Layer):
    """add standard 1D positional embedding"""
    def __init__(self, **kwards):
        super(Pos_embedding, self).__init__(**kwards)
        
    def build(self, input_shape):
        self.pos_embedding = self.add_weight(name='pos_embedding',
                                            shape=(1, input_shape[1], input_shape[2]),
                                            initializer=tf.initializers.RandomNormal(),
                                            trainable=True)
        
    def call(self, inputs):
        return inputs + self.pos_embedding
    
    
class MLP(tf.keras.layers.Layer):
    """MLP layer in encoder of the transformer"""
    def __init__(self, d_model, mlp_dim, dropout_rate, **kwards):
        super(MLP, self).__init__(**kwards)
        self.net = tf.keras.Sequential([tf.keras.layers.Dense(mlp_dim, activation='relu'),
                                      tf.keras.layers.Dropout(dropout_rate),
                                      tf.keras.layers.Dense(d_model),
                                      tf.keras.layers.Dropout(dropout_rate)])
    def call(self, inputs):
        return self.net(inputs)
    
    

class EncoderBlock(tf.keras.layers.Layer):
    """Transformer encoder block."""
    def __init__(self, d_model,
                 mlp_dim, num_heads, dropout_rate, use_bias=False,
                 **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.mlp = MLP(d_model=d_model, mlp_dim=mlp_dim, dropout_rate=dropout_rate)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, X, mask=False):
        output = self.layernorm1(X)
        output = self.dropout1(output)
        output = self.attention(v=X, k=X, q=X, mask=None) + X
        
        output2 = self.layernorm2(output)
        output2 = self.dropout2(output2)
        output2 = self.mlp(output2) + output
        
        return output2
    

class ViT(tf.keras.Model):
    """Vision Transformer model"""
    def __init__(self, d_model, mlp_dim,
                 num_heads, dropout_rate, num_layers, 
                 patch_size, num_classes, use_bias=False, **kwards):
        super(ViT, self).__init__(**kwards)
        self.d_model = d_model
        self.patch = Patch(patch_size)
        self.projection = Projection(d_model)
        self.pos_embedding = Pos_embedding()
        self.blocks = []
        for _ in range(num_layers):
            self.blocks.append(EncoderBlock(d_model, mlp_dim, num_heads, dropout_rate, use_bias))
        self.mlp_head = tf.keras.Sequential([tf.keras.layers.LayerNormalization(epsilon=1e-6),
                                            tf.keras.layers.Dense(num_classes)])
    
    def call(self, X):
        X = self.patch(X)
        X = self.projection(X)
        X = self.pos_embedding(X)
        for blk in self.blocks:
            X = blk(X)
        X = X[:, 0]
        X = self.mlp_head(X)
        return X
    
    def model(self):
        x = tf.keras.Input(shape=(256, 256, 3))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))

import re, os
import numpy as np

import pickle

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback, ReduceLROnPlateau
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model

import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.text import Tokenizer

BASE_DIR = 'word_level_generation'
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
UTILS_DIR = os.path.join(BASE_DIR, 'utils')

TOKENIZER_PATH = os.path.join(UTILS_DIR, 'tokenizer.pickle')

WINDOW_SIZE = 64

class CustomModel:
    def __init__(self, name, model):
        self.name = re.sub('[^A-Za-z0-9]', '_', str(name)).lower()
        self.model_weights_dir = os.path.join(TRAIN_DIR, f'{self.name}_weights')
        self.model = model
        self.load()

    def load(self):
        for i, layer in enumerate(self.model.layers):
            npz_file = os.path.join(self.model_weights_dir, f'layer_{i}.npz')
            if os.path.exists(npz_file):
                # Load all arrays from the .npz
                data = np.load(npz_file, allow_pickle=True)
                weights = [data[key] for key in data]
                layer.set_weights(weights)

                print("Load layer weights from", npz_file)


def read_pickle(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def positional_encoding(seq_len, d_model):
    positions = tf.range(seq_len, dtype=tf.float32)[:, tf.newaxis]
    dims = tf.range(d_model, dtype=tf.float32)[tf.newaxis, :]
    angle_rates = 1 / tf.pow(10000., (2 * (dims // 2)) / d_model)
    angle_rads = positions * angle_rates
    sines = tf.sin(angle_rads[:, 0::2])
    cosines = tf.cos(angle_rads[:, 1::2])
    pos_encoding = tf.concat([sines, cosines], axis=-1)
    return pos_encoding[tf.newaxis, ...]


def build_model_1(tokenizer_dict):

    NAME = "Encoder v1"

    vocab_size = len(tokenizer_dict) + 1 # Padding
    d_model = 144
    seq_len = WINDOW_SIZE + 1 # CLS TOKEN

    # 1️⃣ Input
    input_layer = Input(shape=(seq_len,), dtype=tf.int32)

    # 2️⃣ Embedding
    embedding_layer = Embedding(vocab_size, d_model)
    x = embedding_layer(input_layer)
    x = Dropout(0.1)(x)

    # 3️⃣ Positional Encoding
    pos_encoding = tf.cast(positional_encoding(seq_len, d_model), tf.float32)
    x = Lambda(lambda t: t + pos_encoding[:tf.shape(t)[1], :],
               output_shape=(seq_len, d_model))(x)

    # 🔁 Encoder Block
    def encoder_block(x):
        # Multi-head attention
        attn_output = MultiHeadAttention(num_heads=4, key_dim=d_model // 4)(x, x)
        attn_output = Dropout(0.1)(attn_output)
        x = Add()([x, attn_output])
        x = LayerNormalization(epsilon=1e-6)(x)

        # Feed-forward
        ffn = Dense(4 * d_model, activation='relu')(x)
        ffn = Dropout(0.1)(ffn)
        ffn = Dense(d_model)(ffn)
        x = Add()([x, ffn])
        x = LayerNormalization(epsilon=1e-6)(x)
        return x

    # 4️⃣ Two Encoder Blocks
    x = encoder_block(x)
    x = encoder_block(x)

    # 5️⃣ Extract CLS token
    cls_token = Lambda(lambda t: t[:, 0, :], output_shape=(d_model,))(x)

    # 6️⃣ Feed-forward + Concatenation
    x = Dense(256, activation='relu')(cls_token)
    x = Dropout(0.2)(x)
    x = Concatenate()([x, cls_token])  # shape: (batch, 256 + d_model)

    # 7️⃣ Project back to embedding dim for weight tying
    proj = Dense(d_model)(x)

    # 8️⃣ Weight tying output (wrap tf.matmul in Lambda)
    def tie_weights(t):
        return tf.matmul(t, embedding_layer.embeddings, transpose_b=True)

    logits = Lambda(tie_weights, output_shape=(vocab_size,))(proj)
    output_layer = Activation('softmax')(logits)

    # ✅ Build & Compile
    model = Model(inputs=input_layer, outputs=output_layer)

    return model


def load_pipeline():
    tokenizer = read_pickle(TOKENIZER_PATH)


    model = CustomModel(
        "Encoder v1",
        build_model_1(tokenizer.word_index)
    )

    return tokenizer, model.model
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
plt.style.use('seaborn')
from scipy.stats import skew
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
df = pd.read_csv('data.csv',delimiter=',',keep_default_na=False)
df=df.pivot(index="time", columns="fileID", values="no_req")
seq_len=100
number_of_content=100
total_content=100
df2=df[1]
correlation= df.corrwith(df2, axis = 0)
sorted_corr=correlation.sort_values(ascending=False)
index_sorted_corr=sorted_corr.index
df4 = pd.DataFrame([index_sorted_corr])
sorted_content=df4.iloc[0]
Sorted_df = pd.DataFrame(df, columns =sorted_content)
data = Sorted_df.values
min_return = Sorted_df.min(axis=0)
max_return = Sorted_df.max(axis=0)
normalized_df = (Sorted_df - min_return) / (max_return - min_return)
normalized_data = normalized_df.values
index_array= Sorted_df.columns
Y_label=np.zeros((number_of_content),dtype='float')
skewness=np.zeros((1,number_of_content),dtype='float')
X_data, y_data_classification = [], []
for i in range(seq_len, len(df)):
    X=data[i-seq_len:i,:]
    Normalized_X=normalized_data[i-seq_len:i,:]
    cumulative_requests=np.sum(X,axis=0)
    all_requests=np.sum(cumulative_requests)
    probability=cumulative_requests/all_requests
    for l in range(number_of_content):
        skewness[0,l]=skew(X[:,l])
    popularity=[index_array, probability,skewness.squeeze(axis=0)]
    df_popularity = pd.DataFrame(popularity)
    sorted_df_popularity = df_popularity.sort_values(by=1,axis=1,ascending=False)
    sorted_df_popularity = sorted_df_popularity.values
    storage=0
    for k in range(number_of_content):
        index=int(sorted_df_popularity[0][k])-1
        if storage<=10:
            Y_label[index]=1
            storage=storage+1
        else:
            Y_label[index]=0
    X_ = np.expand_dims(Normalized_X, axis=2)
    X_data.append(X_)
    y_data_classification.append(Y_label)
    Y_label=np.zeros((number_of_content),dtype='float')

X_data, y_data_classification = np.array(X_data), np.array(y_data_classification)
X_train_C, X_test_C, y_train_C, y_test_C = train_test_split(X_data, y_data_classification, test_size=0.2, random_state=1)
input_shape = (100, 100, 1)
num_classes=100
learning_rate = 0.001
weight_decay = 0.01
batch_size = 512
num_epochs = 100

def my_func(arg):
  arg = tf.convert_to_tensor(arg)
  return arg

##
img_size = [100,100]
patch_size =  [100, 1]
num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
projection_dim = 50
num_heads = 5
transformer_units = [
    projection_dim * 3,
    projection_dim,
]
transformer_layers = 2
mlp_head_units = [128]
data_augmentation = keras.Sequential(
    [
        layers.Normalization(),
        layers.Resizing(img_size[0], img_size[1]),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(factor=0.001),
        layers.RandomZoom(
            height_factor=0.01, width_factor=0.01
        ),
    ],
    name="data_augmentation",
)
data_augmentation.layers[0].adapt(X_train_C)
##
def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size_L = patch_size[0]
        self.patch_size_W = patch_size[1]
    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1,self.patch_size_W, self.patch_size_L, 1],
            strides=[1, self.patch_size_W, self.patch_size_L, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches
image = X_train_C[np.random.choice(range(X_train_C.shape[0]))]
resized_image = tf.image.resize(
    tf.convert_to_tensor([image]), size=(img_size[0], img_size[1])
)
patches = Patches(patch_size)(resized_image)
n = int(np.sqrt(patches.shape[1]))

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

def create_vit_classifier_T():
    inputs = layers.Input(shape=input_shape)
    patches = Patches(patch_size)(inputs)
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)
    for _ in range(transformer_layers):
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim , dropout=0.2
        )(x1, x1)
        x2 = layers.Add()([attention_output, encoded_patches])
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = mlp(x3, hidden_units=transformer_units,dropout_rate=0.2) #
        encoded_patches = layers.Add()([x3, x2])
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.1)(representation)
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.2)
    logits = layers.Dense(num_classes, activation='sigmoid')(features)
    model_t = keras.Model(inputs=inputs, outputs=[logits,features])
    return model_t

patch_size =  [10,100]
num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
def create_vit_classifier_F():
    inputs = layers.Input(shape=input_shape)
    patches = Patches(patch_size)(inputs)
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)
    for _ in range(transformer_layers):
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim , dropout=0.2
        )(x1, x1)
        x2 = layers.Add()([attention_output, encoded_patches])
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = mlp(x3, hidden_units=transformer_units,dropout_rate=0.2) #
        encoded_patches = layers.Add()([x3, x2])
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.1)(representation)
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.2)
    logits = layers.Dense(num_classes, activation='sigmoid' )(features)
    model_F = keras.Model(inputs=inputs, outputs=[logits,features])
    return model_F

def Hybrid (create_vit_classifier_T, create_vit_classifier_F):
    in_seq = Input(shape = input_shape )
    Out_T=create_vit_classifier_T(in_seq)[0]
    Out_F=create_vit_classifier_F(in_seq)[0]
    midlayer_model_1 = create_vit_classifier_T(in_seq)[1]
    midlayer_model_2 = create_vit_classifier_F(in_seq)[1]
    num_patches = 1
    projection_dim = 50
    midlayer_model_1 = PatchEncoder(num_patches, projection_dim)(midlayer_model_1)
    midlayer_model_2 = PatchEncoder(num_patches, projection_dim)(midlayer_model_2)
    midlayer_model_1 = layers.LayerNormalization(epsilon=1e-6)(midlayer_model_1)
    midlayer_model_2 = layers.LayerNormalization(epsilon=1e-6)(midlayer_model_2)
    query = midlayer_model_1
    key = midlayer_model_2
    value = midlayer_model_2
    attention_output = layers.Attention(use_scale=True, dropout=0.0)(
        [query, key, value], return_attention_scores=False)
    x2 = layers.Add()([attention_output, midlayer_model_1])
    x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
    x3 = mlp(x3, hidden_units=transformer_units,dropout_rate=0.2)
    encoded_patches = layers.Add()([x3, x2])
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.1)(representation)
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.2)
    logits = layers.Dense(100, activation='sigmoid')(features)
    model_Hybrid = Model(inputs=[in_seq], outputs=[logits])
    return model_Hybrid

def run_experiment(model_Hybrid):
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
    optimizer = tf.keras.optimizers.experimental.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    )
    model_Hybrid.compile(
        optimizer=optimizer,
        loss=keras.losses.BinaryCrossentropy(from_logits=True, label_smoothing=0, axis=-1, name='binary_crossentropy'),
        metrics=keras.metrics.BinaryAccuracy(name='binary_accuracy', dtype=None, threshold=0.5)
    )
    history = model_Hybrid.fit(
        x=X_train_C,
        y=y_train_C,
        batch_size=batch_size,
        epochs=num_epochs,
        validation_split=0.2,
        callbacks=[callback],
    )
    _, accuracy = model_Hybrid.evaluate(X_test_C, y_test_C)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    return history

model_T_=create_vit_classifier_T()
model_F_=create_vit_classifier_F()
model_Hybrid_ = Hybrid(model_T_,model_F_)
model_Hybrid_.summary()
history= run_experiment(model_Hybrid_)
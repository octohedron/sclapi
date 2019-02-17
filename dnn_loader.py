import tensorflow as tf
import tensorflow_hub as hub
import os
from tensorflow.contrib.learn.python.learn.estimators import run_config

MODEL_DIR = os.path.dirname(
    os.getcwd() + "/../data/models/DNNClassifier_model")

embedded_text_feature_column = hub.text_embedding_column(
    key="sentence",
    module_spec="https://tfhub.dev/google/nnlm-en-dim128/1")

config = run_config.RunConfig(
    model_dir=MODEL_DIR,
    save_checkpoints_steps=1000,
    keep_checkpoint_max=10
)

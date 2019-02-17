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

estimator = tf.estimator.DNNClassifier(
    hidden_units=[500, 100],
    feature_columns=[embedded_text_feature_column],
    n_classes=2,
    optimizer=tf.train.AdagradOptimizer(learning_rate=0.003),
    config=config)

def input_eval_set(text):
    td = {}
    td["sentence"] = []
    td["sentence"].append(text)
    dataset = tf.data.Dataset.from_tensor_slices(td)
    dataset = dataset.batch(1)
    return dataset.make_one_shot_iterator().get_next()


result = estimator.predict(input_fn=input_eval_set, predict_keys="classes")



HAVE_COLORLOG = True
MODEL_DIR = os.path.dirname(
    os.getcwd() + "/../data/models/DNNClassifier_model")



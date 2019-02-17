# -*- coding: utf-8 -*-
from flask import Flask
import json
from flask_cors import CORS
from flask import request
import tensorflow as tf

from dnn_loader import embedded_text_feature_column, config

app = Flask(__name__)
CORS(app)

estimator = tf.estimator.DNNClassifier(
    hidden_units=[500, 100],
    feature_columns=[embedded_text_feature_column],
    n_classes=2,
    optimizer=tf.train.AdagradOptimizer(learning_rate=0.003),
    config=config)


@app.route('/api', methods=['POST'])
def api():
    try:
        if request.method == 'POST':
            text = request.form["text"]

            def input_eval_set():
                td = {}
                td["sentence"] = []
                td["sentence"].append(text)
                dataset = tf.data.Dataset.from_tensor_slices(td)
                dataset = dataset.batch(1)
                return dataset.make_one_shot_iterator().get_next()
            pred = estimator.predict(
                input_fn=input_eval_set, predict_keys="classes")
            r = next(pred)["classes"][0]
            if r == b'1':
                return"Positive"
            return "Negative"
    except Exception:
        return "Invalid request"

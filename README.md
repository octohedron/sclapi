# sclapi

Simple text classifier API.

+ Trains a Deep Neural Network classifier with 50K IMDB reviews
+ Saves the trained model in a /data folder in the parent directory
+ Uses flask for serving an API accepting `POST` requests with a text field
+ Responds with 'Positive' or 'Negative'

## Usage

1. Install `tensorflow`, `tensorflow-hub`, `pandas`, `numpy`, `flask`, `flask-cors`, etc.
2. Train the model, might require some time depending on computing power, simple as running the script `python3 DNNClassifier.py`
3. Export flask app, `export FLASK_APP=$(pwd)/api.py`
4. Run it with `flask run`
5. Make `POST` request to `http://127.0.0.1:5000/api` with `form-data` field `text` and some text, i.e. `Today was a wonderful day`, should return `Positive`.
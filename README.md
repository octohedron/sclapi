# sclapi

Simple text classifier API.

## See it in action

Make a `POST` request to `http://datahangar.net/api` with `form-data` field `text` -> `string` i.e. `Today was a wonderful day.` should return `Positive`

## Description

+ Trains a Deep Neural Network classifier with 50K IMDB reviews
+ Saves the trained model in a /data folder in the parent directory
+ Uses flask for serving an API accepting `POST` requests with a text field
+ Responds with 'Positive' or 'Negative'

## Usage

1. Install requirements with `pip3 install -r requeriments.txt`, etc.
2. Train the model, might require some time depending on computing power, simple as running the script `python3 DNNClassifier.py`, you can train it locally and then upload the data folder to your server.
3. Export flask app, `export FLASK_APP=$(pwd)/api.py`
4. Run it with `flask run`
5. Make a `POST` request to `http://127.0.0.1:5000/api` with `form-data` field `text` and some text, i.e. `Today was a wonderful day`, should return `Positive`.

# Grabcite NET

## How to use (train and evaluate)

1. Use [GrabCite][https://github.com/agrafix/grabcite] to prepare your data set
2. Adjust `config.py`
3. Run `python3 prepare_data.py`
4. Run `python3 net.py` to train the binary classifier
5. Run `python3 recommender_lsi.py` to train the LSI
6. Evaluate both using `python3 evaluate_perf.py` and `python3 recommender_lsi_evaluate.py`

## How to use (web frontend)

1. Train everything
2. Run `FLASK_APP=web_frontend.py python3 -m flask run`
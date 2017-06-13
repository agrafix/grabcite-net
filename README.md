# Grabcite NET

## Files

### prepare_data.py

Generate a training/validation/test set from sentences with citation markers (generated by the `grabcite` tool)

### net.py

Train the network with data prepared by `prepare_data.py`

### arch.py

Describes all used network architectures

### load_data.py

Load data from a `prepare_data.py` training set, used by `net.py` and `predict.py`

### evaluate_perf.py

Run predictions on test set and print various scores
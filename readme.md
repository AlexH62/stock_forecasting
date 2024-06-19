# Stock Price Forecaster

This project contains all the code related to my final year project "Evaluating Neural ODEs for Stock Price Forecasting". We implement ARIMA and GARCH models in the baseline files, and all ML models under the models folder.

## Running

To run, first pip install requirements.txt. We use Python 3.11.7 for compatibility. Baselines can then  be ran via:

```
python3 baseline_arima.py {forecasting horizon}
```

Outputs will be saved in results.csv, and graphs in figs/{Model name}/lookahead_{horizon}

To run other models:

```
python3 main.py {model name} {depth} {epochs} {forecasting horizon}
```

The models to pick from are `lstm`, `gru`, `transformer`, `node`, `anode`, or `odetransformer`.
Outputs once again will be saved in results.csv, and graphs in figs/{Model name}/lookahead_{horizon}

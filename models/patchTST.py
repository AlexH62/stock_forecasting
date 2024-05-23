from models.model import Model
from gluonts.torch import PatchTSTEstimator
import numpy as np
import pandas as pd
from gluonts.dataset.common import ListDataset
from gluonts.evaluation import make_evaluation_predictions
from gluonts.dataset.split import split

class PatchTST(Model):
    def __init__(self):
        self.name = "PatchTST"

    def fit(self, train, val=None, neurons=10, epochs=200, lookback=30):
        estimator = PatchTSTEstimator(
                prediction_length=1,
                num_encoder_layers=neurons,
                stride=2,
                patch_len=10,
                context_length=lookback,
                trainer_kwargs={
                    "enable_progress_bar": False,
                    "enable_model_summary": False,
                    "max_epochs": epochs,
                }
            )
        
        self.train = train.reshape(1, -1)
        if val is not None:
            self.val = np.concatenate((train, val), axis=0).reshape(1, -1)
        
        self.freq = "1D"
        self.start = pd.Period("01-01-2019", freq=self.freq)

        train_dataset = ListDataset(
                [{"target": x, "start": self.start} for x in train],
                freq=self.freq,
            )

        if val is not None:
            val_dataset = ListDataset(
                    [{"target": x, "start": self.start} for x in val],
                    freq=self.freq,
                )
            self.predictor = estimator.train(train_dataset, val_dataset)
        else:
            self.predictor = estimator.train(train_dataset)
    
    def predict(self, data):
        test_sample = len(data)

        if hasattr(self, 'val'):
            val = self.val.reshape(-1, 1)
            test = np.concatenate((val, data), axis=0).reshape(1, -1)
        else:
            train = self.train.reshape(-1, 1)
            test = np.concatenate((train, data), axis=0).reshape(1, -1)

        test_ds = ListDataset(
            [{"target": x, "start": self.start} for x in test], freq=self.freq
        )

        prediction_length = 1
        _, test_template = split(
            test_ds, offset=-test_sample
        )
        test_pairs = test_template.generate_instances(
            prediction_length=prediction_length,
            windows=test_sample,
        )

        preds = []

        for i, _ in test_pairs:
            forecast_it, _ = make_evaluation_predictions(
                dataset=[i],  # test dataset
                predictor=self.predictor,  # predictor
                num_samples=100,  # number of sample paths we want for evaluation
            )

            forecasts = list(forecast_it)
            preds.append(forecasts[0].mean[0])
        return np.array(preds)
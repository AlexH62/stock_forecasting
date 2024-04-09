from repository import Repository
from preprocessor import Preprocessor
from models import Models
from metrics import Metrics

TICKER = ["MS"]#"NVDA", "IBM", "AAPL", "NFLX", "GOOG", "GS", "JPM", "BCS", "SAN", "MS"]
PERIOD = "1y"

rmses = []
for ticker in TICKER:
	repository = Repository()
	data = repository.get_data(ticker, PERIOD)

	N_FEATURES = 1
	STEPS = 30
	LOOKAHEAD = 1

	preprocessor = Preprocessor()
	scaled = preprocessor.scale(data)
	X_train, y_train, X_test, y_test = preprocessor.sequence(scaled, STEPS, lookahead=LOOKAHEAD)

	models = Models()
	name, model = models.LSTM(5, STEPS, N_FEATURES)
	model.fit(X_train, y_train, epochs=100)

	y_hat = model.predict(X_test)

	unscaled_train = preprocessor.reverse_transform(y_train)
	unscaled_actual = preprocessor.reverse_transform(y_test)
	unscaled_prediction = preprocessor.reverse_transform(y_hat)

	metrics = Metrics()
	rmse = metrics.print_RMSE(unscaled_actual, unscaled_prediction)
	metrics.plot(unscaled_train, unscaled_actual, unscaled_prediction, name, ticker, LOOKAHEAD)
	rmses.append(rmse)

print(rmses)
file = open("dump.txt", "a")
file.write(name + "\n")
for val in rmses:
	file.write(str(round(val, 4)) + "\n")
file.close()
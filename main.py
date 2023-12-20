import yfinance as yahooFinance

GetFacebookInformation = yahooFinance.Ticker("AMZN")

# Let us get historical stock prices for Facebook 
# covering the past few years.
# max->maximum number of daily prices available 
# for Facebook.
# Valid options are 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 
# 5y, 10y and ytd.
print(GetFacebookInformation.history(period="max"))

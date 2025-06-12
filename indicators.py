def moving_average(data, window):
    return sum(data[-window:]) / window if len(data) >= window else 0

def rsi(data, period=14):
    return 50

class Indicator:
    pass

def calculateSMA(data, period):
    return [0] * len(data)

def calculateEMA(data, period):
    return [0] * len(data)

def calculateRSI(data, period):
    return [0] * len(data)

def calculateMACD(data, fast_period, slow_period, signal_period):
    return [0] * len(data), [0] * len(data), [0] * len(data)

def calculateBollingerBands(data, period, std_dev):
    return [0] * len(data), [0] * len(data), [0] * len(data)

def calculateVWAP(data, volume):
    return [0] * len(data)

def calculateATR(data, period):
    return [0] * len(data)

def calculateStochastic(data, period):
    return [0] * len(data) 
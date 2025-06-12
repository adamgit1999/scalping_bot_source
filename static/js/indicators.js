// Technical Analysis Indicators

// Simple Moving Average (SMA)
function calculateSMA(data, period) {
    const sma = [];
    for (let i = period - 1; i < data.length; i++) {
        const sum = data.slice(i - period + 1, i + 1).reduce((acc, val) => acc + val.close, 0);
        sma.push({
            time: data[i].time,
            value: sum / period
        });
    }
    return sma;
}

// Exponential Moving Average (EMA)
function calculateEMA(data, period) {
    const k = 2 / (period + 1);
    const ema = [];
    let prevEma = data[0].close;

    for (let i = 1; i < data.length; i++) {
        const currentEma = (data[i].close - prevEma) * k + prevEma;
        ema.push({
            time: data[i].time,
            value: currentEma
        });
        prevEma = currentEma;
    }
    return ema;
}

// Relative Strength Index (RSI)
function calculateRSI(data, period = 14) {
    const rsi = [];
    let gains = 0;
    let losses = 0;

    // Calculate initial average gain and loss
    for (let i = 1; i <= period; i++) {
        const change = data[i].close - data[i - 1].close;
        if (change >= 0) {
            gains += change;
        } else {
            losses -= change;
        }
    }

    let avgGain = gains / period;
    let avgLoss = losses / period;

    // Calculate RSI for the rest of the data
    for (let i = period + 1; i < data.length; i++) {
        const change = data[i].close - data[i - 1].close;
        if (change >= 0) {
            avgGain = (avgGain * (period - 1) + change) / period;
            avgLoss = (avgLoss * (period - 1)) / period;
        } else {
            avgGain = (avgGain * (period - 1)) / period;
            avgLoss = (avgLoss * (period - 1) - change) / period;
        }

        const rs = avgGain / avgLoss;
        const rsiValue = 100 - (100 / (1 + rs));

        rsi.push({
            time: data[i].time,
            value: rsiValue
        });
    }

    return rsi;
}

// Moving Average Convergence Divergence (MACD)
function calculateMACD(data, fastPeriod = 12, slowPeriod = 26, signalPeriod = 9) {
    const fastEMA = calculateEMA(data, fastPeriod);
    const slowEMA = calculateEMA(data, slowPeriod);
    const macdLine = [];
    const signalLine = [];
    const histogram = [];

    // Calculate MACD line
    for (let i = 0; i < fastEMA.length; i++) {
        const macdValue = fastEMA[i].value - slowEMA[i].value;
        macdLine.push({
            time: fastEMA[i].time,
            value: macdValue
        });
    }

    // Calculate Signal line (EMA of MACD)
    const signalEMA = calculateEMA(macdLine, signalPeriod);

    // Calculate Histogram
    for (let i = 0; i < signalEMA.length; i++) {
        const histValue = macdLine[i + signalPeriod - 1].value - signalEMA[i].value;
        histogram.push({
            time: signalEMA[i].time,
            value: histValue
        });
    }

    return {
        macd: macdLine.slice(signalPeriod - 1),
        signal: signalEMA,
        histogram: histogram
    };
}

// Bollinger Bands
function calculateBollingerBands(data, period = 20, multiplier = 2) {
    const bands = [];
    const sma = calculateSMA(data, period);

    for (let i = period - 1; i < data.length; i++) {
        const slice = data.slice(i - period + 1, i + 1);
        const standardDeviation = Math.sqrt(
            slice.reduce((acc, val) => acc + Math.pow(val.close - sma[i - period + 1].value, 2), 0) / period
        );

        bands.push({
            time: data[i].time,
            middle: sma[i - period + 1].value,
            upper: sma[i - period + 1].value + (multiplier * standardDeviation),
            lower: sma[i - period + 1].value - (multiplier * standardDeviation)
        });
    }

    return bands;
}

// Volume Weighted Average Price (VWAP)
function calculateVWAP(data) {
    const vwap = [];
    let cumulativeTPV = 0; // Total Price * Volume
    let cumulativeVolume = 0;

    for (let i = 0; i < data.length; i++) {
        const typicalPrice = (data[i].high + data[i].low + data[i].close) / 3;
        const volume = data[i].volume || 0;

        cumulativeTPV += typicalPrice * volume;
        cumulativeVolume += volume;

        vwap.push({
            time: data[i].time,
            value: cumulativeTPV / cumulativeVolume
        });
    }

    return vwap;
}

// Average True Range (ATR)
function calculateATR(data, period = 14) {
    const tr = [];
    const atr = [];

    // Calculate True Range
    for (let i = 1; i < data.length; i++) {
        const high = data[i].high;
        const low = data[i].low;
        const prevClose = data[i - 1].close;

        const tr1 = high - low;
        const tr2 = Math.abs(high - prevClose);
        const tr3 = Math.abs(low - prevClose);

        tr.push(Math.max(tr1, tr2, tr3));
    }

    // Calculate ATR
    let sum = tr.slice(0, period).reduce((acc, val) => acc + val, 0);
    atr.push({
        time: data[period].time,
        value: sum / period
    });

    for (let i = period; i < tr.length; i++) {
        sum = (atr[atr.length - 1].value * (period - 1) + tr[i]) / period;
        atr.push({
            time: data[i + 1].time,
            value: sum
        });
    }

    return atr;
}

// Stochastic Oscillator
function calculateStochastic(data, kPeriod = 14, dPeriod = 3) {
    const stoch = [];
    const kLine = [];
    const dLine = [];

    for (let i = kPeriod - 1; i < data.length; i++) {
        const slice = data.slice(i - kPeriod + 1, i + 1);
        const highestHigh = Math.max(...slice.map(candle => candle.high));
        const lowestLow = Math.min(...slice.map(candle => candle.low));
        const currentClose = data[i].close;

        const k = ((currentClose - lowestLow) / (highestHigh - lowestLow)) * 100;
        kLine.push({
            time: data[i].time,
            value: k
        });
    }

    // Calculate D line (SMA of K line)
    const d = calculateSMA(kLine, dPeriod);
    dLine.push(...d);

    return {
        k: kLine,
        d: dLine
    };
}

// Export all indicator functions
export {
    calculateSMA,
    calculateEMA,
    calculateRSI,
    calculateMACD,
    calculateBollingerBands,
    calculateVWAP,
    calculateATR,
    calculateStochastic
};
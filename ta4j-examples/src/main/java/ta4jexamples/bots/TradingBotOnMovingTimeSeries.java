/**
 * The MIT License (MIT)
 * <p>
 * Copyright (c) 2014-2017 Marc de Verdelhan & respective authors (see AUTHORS)
 * <p>
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 * <p>
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * <p>
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */
package ta4jexamples.bots;

import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.conf.preprocessor.FeedForwardToRnnPreProcessor;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.DateAxis;
import org.jfree.chart.plot.XYPlot;
import org.jfree.data.time.TimeSeriesCollection;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.ta4j.core.analysis.criteria.BuyAndHoldCriterion;
import org.ta4j.core.analysis.criteria.TotalProfitCriterion;
import org.ta4j.core.trading.rules.CrossedDownIndicatorRule;
import org.ta4j.core.trading.rules.CrossedUpIndicatorRule;
import ta4jexamples.reader.*;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.ta4j.core.*;
import org.ta4j.core.indicators.*;
import org.ta4j.core.indicators.helpers.*;
import org.ta4j.core.trading.rules.OverIndicatorRule;
import org.ta4j.core.trading.rules.UnderIndicatorRule;

import ta4jexamples.enums.ActionType;
import ta4jexamples.loaders.CsvTradesLoader;
import ta4jexamples.strategy.AISTrategy;

import java.text.SimpleDateFormat;
import java.time.ZonedDateTime;
import java.util.*;

import static ta4jexamples.analysis.BuyAndSellSignalsToChart.addBuySellSignals;
import static ta4jexamples.analysis.BuyAndSellSignalsToChart.buildChartTimeSeries;
import static ta4jexamples.analysis.BuyAndSellSignalsToChart.displayChart;

/**
 * This class is an example of a dummy trading bot using ta4j.
 * <p>
 */
public class TradingBotOnMovingTimeSeries {

    /**
     * Close price of the last tick
     */
    private static Decimal LAST_TICK_CLOSE_PRICE;

    static List<Indicator<Decimal>> indicators = new ArrayList<>();

    private static int nEpochs = 10;

    static Indicator cuttedSma;

    static TimeSeries cutted;

    static ClosePriceIndicator cuttedClosePrice;

    /**
     * Builds a moving time series (i.e. keeping only the maxTickCount last ticks)
     *
     * @param maxTickCount the number of ticks to keep in the time series (at maximum)
     * @return a moving time series
     */
    private static TimeSeries[] initMovingTimeSeries() {
        List<Tick> ticks = CsvTradesLoader.loadBitstampSeries().getTickData();

        TimeSeries[] timeSeries = new TimeSeries[3];


        List<Tick> fakeTicks = new ArrayList<>();

        int frame = 5;
        for (int i = 0; i < ticks.size() - frame; i++) {
            Tick tick = ticks.get(i);
            Tick tickFrame = ticks.get(i + frame);
            tick = new BaseTick(tick.getTimePeriod(), tick.getEndTime(), tick.getOpenPrice(), tick.getMaxPrice(), tick.getMinPrice(), tickFrame.getClosePrice(), tick.getVolume(), tick.getAmount());
            fakeTicks.add(tick);
        }


        timeSeries[0] = new BaseTimeSeries(new ArrayList<>(ticks.subList(0, 1600)));
        timeSeries[2] = new BaseTimeSeries(new ArrayList<>(fakeTicks.subList(0, 1600)));
        timeSeries[1] = new BaseTimeSeries(new ArrayList<>(ticks.subList(1600, ticks.size() - 1)));

        System.out.print("Initial tick count: " + timeSeries[0].getTickCount());
        LAST_TICK_CLOSE_PRICE = timeSeries[0].getTick(timeSeries[0].getEndIndex()).getClosePrice();

        System.out.print("Test tick count: " + timeSeries[0].getTickCount());

        return timeSeries;
    }

    /**
     * @param series a time series
     * @return a dummy strategy
     */
    private static Rule[] buildStrategy(TimeSeries[] series) {
        if (series == null) {
            throw new IllegalArgumentException("Series cannot be null");
        }

//        cutted = new BaseTimeSeries(new ArrayList<>(series.getTickData().subList(3,series.getTickData().size())));
        cutted = series[2];
        cuttedClosePrice = new ClosePriceIndicator(cutted);
        cuttedSma = new SMAIndicator(cuttedClosePrice, 10);
//        cuttedSma = new EMAIndicator(cuttedClosePrice, 5);

        ClosePriceIndicator closePrice = new ClosePriceIndicator(series[0]);
        VolumeIndicator volumeIndicator = new VolumeIndicator(series[0]);
        OpenPriceIndicator openPriceIndicator = new OpenPriceIndicator(series[0]);
        MinPriceIndicator minPriceIndicator = new MinPriceIndicator(series[0]);
        MaxPriceIndicator maxPriceIndicator = new MaxPriceIndicator(series[0]);

        SMAIndicator sma5 = new SMAIndicator(closePrice, 5);

        SMAIndicator sma12 = new SMAIndicator(closePrice, 12);
        SMAIndicator sma50 = new SMAIndicator(closePrice, 50);
        SMAIndicator sma200 = new SMAIndicator(closePrice, 200);

        RSIIndicator rsi1 = new RSIIndicator(closePrice, 1);
        RSIIndicator rsi2 = new RSIIndicator(closePrice, 2);

        EMAIndicator ema5 = new EMAIndicator(closePrice, 5);
        EMAIndicator ema12 = new EMAIndicator(closePrice, 12);
        EMAIndicator ema50 = new EMAIndicator(closePrice, 50);
        EMAIndicator ema200 = new EMAIndicator(closePrice, 200);

        MACDIndicator macd = new MACDIndicator(closePrice, 9, 26);

        CCIIndicator CCI5 = new CCIIndicator(series[0], 5);
        CCIIndicator CCI50 = new CCIIndicator(series[0], 50);
        CCIIndicator CCI200 = new CCIIndicator(series[0], 200);

        ROCIndicator rocIndicator = new ROCIndicator(closePrice, 12);
        ROCIndicator rocIndicatorVolume = new ROCIndicator(closePrice, 12);

        SumIndicator sumIndicator = new SumIndicator(minPriceIndicator, maxPriceIndicator);
        MultiplierIndicator multiplierIndicator = new MultiplierIndicator(sumIndicator, Decimal.valueOf(0.5));
        AwesomeOscillatorIndicator awesomeOscillatorIndicator = new AwesomeOscillatorIndicator(multiplierIndicator,5,50);

        indicators.add(closePrice);
        indicators.add(volumeIndicator);
        indicators.add(openPriceIndicator);
        indicators.add(minPriceIndicator);
        indicators.add(maxPriceIndicator);

        indicators.add(sma5);
        indicators.add(sma12);
        indicators.add(sma50);
//        indicators.add(sma200);

        indicators.add(ema5);
        indicators.add(ema12);
        indicators.add(ema50);
//        indicators.add(ema200);

        indicators.add(rsi1);
//        indicators.add(rsi2);

        indicators.add(macd);

        indicators.add(CCI5);
        indicators.add(CCI50);
//        indicators.add(CCI200);

        indicators.add(rocIndicator);
        indicators.add(rocIndicatorVolume);

        indicators.add(awesomeOscillatorIndicator);

        // Signals
        // Buy when SMA goes over close price
        // Sell when close price goes over SMA

        Rule[] rules = new Rule[2];
        rules[0] = new OverIndicatorRule(sma5, sma200) // Trend
                .and(new CrossedDownIndicatorRule(rsi2, Decimal.valueOf(5))) // Signal 1
                .and(new OverIndicatorRule(sma5, closePrice));
        ;
        rules[1] = new UnderIndicatorRule(sma5, sma200) // Trend
                .and(new CrossedUpIndicatorRule(rsi2, Decimal.valueOf(95))) // Signal 1
                .and(new UnderIndicatorRule(sma5, closePrice));

        rules[0] = new OverIndicatorRule(cuttedSma, closePrice);
        rules[1] = new UnderIndicatorRule(cuttedSma, closePrice);

        rules[0] = new OverIndicatorRule(cuttedSma, closePrice).and(new OverIndicatorRule(sma5, sma50));
        rules[1] = new UnderIndicatorRule(cuttedSma, closePrice).and(new UnderIndicatorRule(sma5, sma50));
/*
        rules[0] = new OverIndicatorRule(cuttedSma, closePrice)
                .and(new OverIndicatorRule(sma5, sma50))
                .and(new CrossedDownIndicatorRule(rsi2, Decimal.valueOf(35)));
        rules[1] = new UnderIndicatorRule(cuttedSma, closePrice)
                .and(new UnderIndicatorRule(sma5, sma50))
                .and(new CrossedUpIndicatorRule(rsi2, Decimal.valueOf(65)));
                */
        return rules;
    }

    /**
     * Generates a random decimal number between min and max.
     *
     * @param min the minimum bound
     * @param max the maximum bound
     * @return a random decimal number between min and max
     */
    private static Decimal randDecimal(Decimal min, Decimal max) {
        Decimal randomDecimal = null;
        if (min != null && max != null && min.isLessThan(max)) {
            randomDecimal = max.minus(min).multipliedBy(Decimal.valueOf(Math.random())).plus(min);
        }
        return randomDecimal;
    }

    /**
     * Generates a random tick.
     *
     * @return a random tick
     */
    private static Tick generateRandomTick() {
        final Decimal maxRange = Decimal.valueOf("0.03"); // 3.0%
        Decimal openPrice = LAST_TICK_CLOSE_PRICE;
        Decimal minPrice = openPrice.minus(openPrice.multipliedBy(maxRange.multipliedBy(Decimal.valueOf(Math.random()))));
        Decimal maxPrice = openPrice.plus(openPrice.multipliedBy(maxRange.multipliedBy(Decimal.valueOf(Math.random()))));
        Decimal closePrice = randDecimal(minPrice, maxPrice);
        LAST_TICK_CLOSE_PRICE = closePrice;
        return new BaseTick(ZonedDateTime.now(), openPrice, maxPrice, minPrice, closePrice, Decimal.ONE);
    }

    private static MultiLayerConfiguration getNetwork(int layerSize, int classes) {

        int secondLayerSize = (int) Math.ceil(2.5 * layerSize);
/*
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .iterations(1)
                .weightInit(WeightInit.XAVIER)
                .activation(Activation.RELU)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .learningRate(0.05)
                .list()
                .backprop(true)
                .layer(0, new DenseLayer.Builder()
                        .nIn(layerSize) // Number of input datapoints.
                        .nOut(secondLayerSize) // Number of output datapoints.
                        .activation(Activation.RELU) // Activation function.
                        .weightInit(WeightInit.XAVIER) // Weight initialization.
                        .build())
                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nIn(secondLayerSize)
                        .nOut(classes)
                        .activation(Activation.SOFTMAX)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .build();
*/

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(140)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .iterations(1)
                .weightInit(WeightInit.XAVIER)
                .updater(Updater.NESTEROVS)
                .learningRate(0.03)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(layerSize).nOut(secondLayerSize)
                        .activation(Activation.TANH).build())
                .layer(1, new DenseLayer.Builder().nIn(secondLayerSize).nOut(secondLayerSize)
                        .activation(Activation.TANH).build())
                .layer(2, new DenseLayer.Builder().nIn(secondLayerSize).nOut(secondLayerSize)
                        .activation(Activation.TANH).build())
                .layer(3, new DenseLayer.Builder().nIn(secondLayerSize).nOut(secondLayerSize)
                        .activation(Activation.TANH).build())
                .layer(4, new GravesLSTM.Builder()
                        .activation(Activation.SOFTSIGN)
                        .nIn(secondLayerSize)
                        .nOut(secondLayerSize)
                        .weightInit(WeightInit.XAVIER)
                        .updater(Updater.ADAGRAD)
                        .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                        .gradientNormalizationThreshold(10)
                        .learningRate(0.008)
                        .build())
                .layer(5, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX)
                        .nIn(secondLayerSize)
                        .nOut(classes)
                        .updater(Updater.ADAGRAD)
                        .weightInit(WeightInit.XAVIER)
                        .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                        .gradientNormalizationThreshold(10)
                        .build())
                .inputPreProcessor(4, new FeedForwardToRnnPreProcessor())
                .pretrain(false)
                .backprop(true)
                .tBPTTForwardLength(100)
                .tBPTTBackwardLength(100)
                .build();

        return conf;
    }

    public static void main(String[] args) throws InterruptedException {

        System.out.println("********************** Initialization **********************");
        // Getting the time series
        TimeSeries[] series = initMovingTimeSeries();

        // Building the trading strategy
        Rule[] rules = buildStrategy(series);

        int frameSize = 10;
        int batchSize = 100;
        int possibleLabels = 2;

        DataSetIterator trainData = new RecordReaderDataSetIterator(new TickRecordReader(series[0], indicators, frameSize, rules), batchSize, indicators.size(), possibleLabels) {
            @Override
            public List<String> getLabels() {
                return Arrays.asList(ActionType.BUY.getName(), ActionType.SELL.getName());
            }
        };
//        DataSetIterator iter = new SequenceRecordReaderDataSetIterator(new TickRecordReader(series[0], indicators, frameSize), batchSize, indicators.size(), possibleLabels);

        MultiLayerConfiguration multiLayerConfiguration = getNetwork(indicators.size(), possibleLabels);
        MultiLayerNetwork model = new MultiLayerNetwork(multiLayerConfiguration);
        model.init();
        model.setListeners(new ScoreIterationListener(10));    //Print score every 100 parameter updates

        /*
        for (int n = 0; n < nEpochs; n++) {
            model.fit(iter);
        }
        */

        NormalizerStandardize normalizer = new NormalizerStandardize();
        normalizer.fit(trainData);
        trainData.reset();

        trainData.setPreProcessor(normalizer);

        System.out.println("Means original: " + normalizer.getMean());
        System.out.println("Stds original:  " + normalizer.getStd());

        for (int i = 0; i < nEpochs; i++) {
            while (trainData.hasNext()) {
                DataSet dataSet = trainData.next();
                model.fit(dataSet);
            }
            trainData.reset();
            System.out.println("Epoch " + i + " complete");
        }


        trainData.reset();
        while (trainData.hasNext()) {
            DataSet dataSet = trainData.next();
            model.rnnTimeStep(dataSet.getFeatureMatrix());
        }

//        Strategy aisTrategy = new AISTrategy(model, indicators, normalizer);
        Strategy aisTrategy = new AISTrategy(model, indicators, normalizer);


        // Initializing the trading history
        TradingRecord tradingRecord = new BaseTradingRecord();
        System.out.println("************************************************************");

        /**
         * We run the strategy for the 50 next ticks.
         */
//        List<Tick> testTicks = series[1].getTickData().stream().map(SerializationUtils::clone).collect(Collectors.toList());
        List<Tick> testTicks = series[1].getTickData();

        for (Tick tick : testTicks) {

            Thread.sleep(30); // I know...

            series[0].addTick(tick);

            int endIndex = series[0].getEndIndex();
            if (aisTrategy.shouldEnter(endIndex)) {
                // Our strategy should enter
                System.out.println("Strategy should ENTER on " + endIndex);
                boolean entered = tradingRecord.enter(endIndex, tick.getClosePrice(), Decimal.TEN);
                if (entered) {
                    Order entry = tradingRecord.getLastEntry();
                    System.out.println("Entered on " + entry.getIndex()
                            + " (price=" + entry.getPrice().toDouble()
                            + ", amount=" + entry.getAmount().toDouble() + ")");
                }
            } else if (aisTrategy.shouldExit(endIndex)) {
                // Our strategy should exit
                System.out.println("Strategy should EXIT on " + endIndex);
                boolean exited = tradingRecord.exit(endIndex, tick.getClosePrice(), Decimal.TEN);
                if (exited) {
                    Order exit = tradingRecord.getLastExit();
                    System.out.println("Exited on " + exit.getIndex()
                            + " (price=" + exit.getPrice().toDouble()
                            + ", amount=" + exit.getAmount().toDouble() + ")");
                }
            }
        }


        Evaluation eval = new Evaluation(possibleLabels);

        DataSetIterator testIter = new RecordReaderDataSetIterator(new TickRecordReader(series[1], indicators, frameSize, rules), batchSize, indicators.size(), possibleLabels);

        testIter.setPreProcessor(normalizer);
//        Evaluation evaluation = model.evaluate(testIter);
//        System.out.print(evaluation.accuracy() + " " + evaluation.f1());

        while (testIter.hasNext()) {
            DataSet t = testIter.next();
            INDArray features = t.getFeatureMatrix();
            INDArray labels = t.getLabels();
            INDArray predicted = model.output(features, false);

            eval.eval(labels, predicted);
//            eval.evalTimeSeries(labels, predicted);
        }

        System.out.print(eval.stats());

        System.out.println("Total profit for the strategy: " + new TotalProfitCriterion().calculate(series[0], tradingRecord));
        System.out.println("Buy-and-hold: " + new BuyAndHoldCriterion().calculate(series[1], tradingRecord));

        TimeSeriesCollection dataset = new TimeSeriesCollection();
        dataset.addSeries(buildChartTimeSeries(series[0], new ClosePriceIndicator(series[0]), "Bitstamp Bitcoin (BTC) - learn"));
        dataset.addSeries(buildChartTimeSeries(cutted, cuttedSma, "cuttedSma"));
        dataset.addSeries(buildChartTimeSeries(cutted, cuttedClosePrice, "ccc"));

        /**
         * Creating the chart
         */
        JFreeChart chart = ChartFactory.createTimeSeriesChart(
                "Bitstamp BTC", // title
                "Date", // x-axis label
                "Price", // y-axis label
                dataset, // data
                true, // create legend?
                true, // generate tooltips?
                false // generate URLs?
        );
        XYPlot plot = (XYPlot) chart.getPlot();
        DateAxis axis = (DateAxis) plot.getDomainAxis();
        axis.setDateFormatOverride(new SimpleDateFormat("MM-dd HH:mm"));
        displayChart(chart);

        addBuySellSignals(series[0], tradingRecord.getTrades(), plot);
    }
}

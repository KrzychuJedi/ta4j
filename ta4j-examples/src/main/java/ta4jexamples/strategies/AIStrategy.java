package ta4jexamples.strategies;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.ta4j.core.Rule;
import org.ta4j.core.Strategy;
import org.ta4j.core.TradingRecord;

public class AIStrategy implements Strategy {

    public AIStrategy() {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .iterations(1)
                .weightInit(WeightInit.XAVIER)
                .activation(Activation.RELU)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .learningRate(0.05)
                .list()
                .backprop(true)
                .layer(0, new DenseLayer.Builder()
                        .nIn(5) // Number of input datapoints.
                        .nOut(10) // Number of output datapoints.
                        .activation(Activation.RELU) // Activation function.
                        .weightInit(WeightInit.XAVIER) // Weight initialization.
                        .build())
                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nIn(10)
                        .nOut(3)
                        .activation(Activation.SOFTMAX)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .build();

    }

    @Override
    public Rule getEntryRule() {
        return null;
    }

    @Override
    public Rule getExitRule() {
        return null;
    }

    @Override
    public void setUnstablePeriod(int unstablePeriod) {

    }

    @Override
    public boolean isUnstableAt(int index) {
        return false;
    }

    @Override
    public boolean shouldEnter(int index, TradingRecord tradingRecord) {
        return false;
    }

    @Override
    public boolean shouldExit(int index, TradingRecord tradingRecord) {
        return false;
    }
}

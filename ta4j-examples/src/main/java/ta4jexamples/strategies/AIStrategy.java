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

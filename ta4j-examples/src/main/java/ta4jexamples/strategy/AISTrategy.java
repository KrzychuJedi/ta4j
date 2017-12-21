package ta4jexamples.strategy;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;
import org.ta4j.core.*;

import java.util.List;
import java.util.Objects;

public class AISTrategy implements Strategy {

    private final MultiLayerNetwork model;

    private final List<Indicator<Decimal>> indicators;
    private NormalizerStandardize normalizerStandardize = null;

    private int last = -1;

    public AISTrategy(MultiLayerNetwork model, List<Indicator<Decimal>> indicators) {
        this.model = model;
        this.indicators = indicators;
    }

    public AISTrategy(MultiLayerNetwork model, List<Indicator<Decimal>> indicators, NormalizerStandardize normalizerStandardize) {
        this(model, indicators);
        this.normalizerStandardize = normalizerStandardize;
    }

    public boolean shouldEnter(int idx) {

        if (last != 1) {
            double[] doubles = indicators.stream().mapToDouble(indicator -> indicator.getValue(idx).toDouble()).toArray();
            INDArray features = Nd4j.create(doubles, new int[]{indicators.size()});
            if (normalizerStandardize != null) {
                normalizerStandardize.transform(features);
            }
//        INDArray output = model.output(features, false);
            INDArray output = model.output(features);
            boolean enter = Objects.equals(output.getColumn(0).maxNumber(), output.getRow(0).maxNumber())
                    && !Objects.equals(output.getColumn(0).maxNumber(), output.getColumn(1).maxNumber());
            if (enter) {
                last = 1;
            }
            return enter;
        }
        return false;
    }

    public boolean shouldExit(int idx) {
        if (last == 1) {
            double[] doubles = indicators.stream().mapToDouble(indicator -> indicator.getValue(idx).toDouble()).toArray();
            INDArray features = Nd4j.create(doubles, new int[]{indicators.size()});
            if (normalizerStandardize != null) {
                normalizerStandardize.transform(features);
            }
//        INDArray output = model.output(features, false);
            INDArray output = model.output(features);
            boolean exit = Objects.equals(output.getColumn(1).maxNumber(), output.getRow(0).maxNumber())
                    && !Objects.equals(output.getColumn(0).maxNumber(), output.getColumn(1).maxNumber());
            if (exit) {
                last = 0;
            }
            return exit;
        }
        return false;
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
}

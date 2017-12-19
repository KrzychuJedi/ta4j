package ta4jexamples.strategy;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.ta4j.core.*;

import java.util.List;
import java.util.Objects;

public class AISTrategy implements Strategy {

    private final MultiLayerNetwork model;

    private final List<Indicator<Decimal>> indicators;

    private int last = -1;

    public AISTrategy(MultiLayerNetwork model, List<Indicator<Decimal>> indicators) {
        this.model = model;
        this.indicators = indicators;
    }

    public boolean shouldEnter(int idx) {

        double[] doubles = indicators.stream().mapToDouble(indicator -> indicator.getValue(idx).toDouble()).toArray();
        INDArray features = Nd4j.create(doubles, new int[]{indicators.size()});
//        INDArray output = model.output(features, false);
        INDArray output = model.rnnTimeStep(features);
        boolean enter = Objects.equals(output.getColumn(0).maxNumber(), output.getRow(0).maxNumber())&& last != 1;
        if (enter) {
            last = 1;
        }
        return enter;
    }

    public boolean shouldExit(int idx) {
        double[] doubles = indicators.stream().mapToDouble(indicator -> indicator.getValue(idx).toDouble()).toArray();
        INDArray features = Nd4j.create(doubles, new int[]{indicators.size()});
//        INDArray output = model.output(features, false);
        INDArray output = model.rnnTimeStep(features);
        boolean exit = Objects.equals(output.getColumn(1).maxNumber(), output.getRow(0).maxNumber()) && last == 1;
        if(exit){
            last =0;
        }
        return exit;
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

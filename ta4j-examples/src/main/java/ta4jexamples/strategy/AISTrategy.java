package ta4jexamples.strategy;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.ta4j.core.*;

import java.util.List;
import java.util.stream.IntStream;

public class AISTrategy {

    private final MultiLayerNetwork model;

    private int last = -1;

    public AISTrategy(MultiLayerNetwork model) {
        this.model = model;
    }

    public boolean shouldEnter(int idx, List<Indicator<Decimal>> indicators) {

        double[] doubles = indicators.stream().mapToDouble(indicator -> indicator.getValue(idx).toDouble()).toArray();
        INDArray features = Nd4j.create(doubles, new int[]{21});
        INDArray output = model.output(features, false);
        boolean enter = output.getColumn(0).data().asInt()[0] == 1 && last != 1;
        if (enter) {
            last = 1;
        }
        return enter;
    }

    public boolean shouldExit(int idx, List<Indicator<Decimal>> indicators) {
        double[] doubles = indicators.stream().mapToDouble(indicator -> indicator.getValue(idx).toDouble()).toArray();
        INDArray features = Nd4j.create(doubles, new int[]{21});
        INDArray output = model.output(features, false);
        boolean exit = output.getColumn(1).data().asInt()[0] == 1 && last == 1;
        if(exit){
            last =0;
        }
        return exit;
    }
}

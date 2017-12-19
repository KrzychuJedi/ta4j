package ta4jexamples.reader;

import org.datavec.api.conf.Configuration;
import org.datavec.api.records.Record;
import org.datavec.api.records.listener.RecordListener;
import org.datavec.api.records.metadata.RecordMetaData;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.split.InputSplit;
import org.datavec.api.writable.DoubleWritable;
import org.datavec.api.writable.IntWritable;
import org.datavec.api.writable.Text;
import org.datavec.api.writable.Writable;
import org.ta4j.core.*;

import ta4jexamples.enums.*;

import java.io.DataInputStream;
import java.io.IOException;
import java.net.URI;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import java.util.stream.Collectors;

public class TickRecordReader implements RecordReader {

    private final Rule[] rules;
    private List<RecordListener> recordListeners = new ArrayList<>();

    private final TimeSeries timeSeries;
    private final List<Indicator<Decimal>> indicators;

    private int counter = 0;
    private int frame;

    public TickRecordReader(TimeSeries timeSeries, List<Indicator<Decimal>> indicators, int frame, Rule[] rules) {
        this.timeSeries = timeSeries;
        this.indicators = indicators;
        this.frame = frame;
        this.rules = rules;
    }

    private ActionType getType(TimeSeries series, int idx, int frame) {
        Tick first = series.getTick(idx);
        Tick firstAndOne = series.getTick(idx + 1);
        Tick second = series.getTick(idx + frame);

        Decimal pertencage = second.getClosePrice().multipliedBy(Decimal.valueOf(100)).dividedBy(first.getClosePrice());

        if (this.rules[1].isSatisfied(idx)) {
            return ActionType.SELL;
        } else if (this.rules[0].isSatisfied(idx)) {
            return ActionType.BUY;
        }
        return ActionType.BUY;
    }

    /*
        private static ActionType getType(TimeSeries series, int idx, int frame){
            Tick first = series.getTick(idx);
            Tick second = series.getTick(frame);

            if(second.getClosePrice().isGreaterThanOrEqual(first.getClosePrice())){
                return ActionType.BUY;
            } else if (second.getClosePrice().isLessThanOrEqual(first.getClosePrice())){
                return ActionType.SELL;
            }
            return ActionType.NOTHING;
        }
    */
    @Override
    public void initialize(InputSplit split) throws IOException, InterruptedException {

    }

    @Override
    public void initialize(Configuration conf, InputSplit split) throws IOException, InterruptedException {

    }

    @Override
    public boolean batchesSupported() {
        return false;
    }

    @Override
    public List<Writable> next(int num) {
        List<Writable> writables = new ArrayList<>();
        for (Indicator<Decimal> indicator : indicators) {
            writables.add(new DoubleWritable(indicator.getValue(num).toDouble()));
        }
        writables.add(new Text(getType(timeSeries, num, frame).getName()));
        return writables;
    }

    @Override
    public List<Writable> next() {
        List<Writable> writables = new ArrayList<>();
        for (Indicator<Decimal> indicator : indicators) {
            writables.add(new DoubleWritable(indicator.getValue(counter).toDouble()));
        }
        writables.add(new IntWritable(getType(timeSeries, counter, frame).getClassNr()));
        counter++;
        return writables;
    }

    @Override
    public boolean hasNext() {
        return counter <= timeSeries.getEndIndex() - frame;
    }

    @Override
    public List<String> getLabels() {
        return Arrays.stream(ActionType.values()).map(ActionType::getName).collect(Collectors.toList());
    }

    @Override
    public void reset() {

    }

    @Override
    public List<Writable> record(URI uri, DataInputStream dataInputStream) throws IOException {
        throw new UnsupportedOperationException("Reading CSV data from DataInputStream not yet implemented");
    }

    @Override
    public Record nextRecord() {
        return new org.datavec.api.records.impl.Record(next(), null);
    }

    @Override
    public Record loadFromMetaData(RecordMetaData recordMetaData) throws IOException {
        return null;
    }

    @Override
    public List<Record> loadFromMetaData(List<RecordMetaData> recordMetaDatas) throws IOException {
        return null;
    }

    @Override
    public List<RecordListener> getListeners() {
        return recordListeners;
    }

    @Override
    public void setListeners(RecordListener... listeners) {
        recordListeners.addAll(Arrays.asList(listeners));
    }

    @Override
    public void setListeners(Collection<RecordListener> listeners) {
        recordListeners.addAll(listeners);
    }

    @Override
    public void close() throws IOException {

    }

    @Override
    public void setConf(Configuration conf) {

    }

    @Override
    public Configuration getConf() {
        return null;
    }
}

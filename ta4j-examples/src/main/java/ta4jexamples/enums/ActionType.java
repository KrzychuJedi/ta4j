package ta4jexamples.enums;

public enum ActionType {

    BUY("BUY", 0),
    SELL("SELL", 1),
    NOTHING("NOTHING", 2);

    ActionType(String name, int classNr) {
        this.name = name;
        this.classNr = classNr;
    }

    private String name;
    private int classNr;

    public String getName() {
        return name;
    }

    public int getClassNr() {
        return classNr;
    }
}

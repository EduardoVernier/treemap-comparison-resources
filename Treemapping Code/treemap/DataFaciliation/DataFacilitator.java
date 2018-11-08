package treemap.DataFaciliation;

import java.awt.Color;
import java.util.HashMap;
import treemap.dataStructure.DataMap;

/**
 *
 * @author Max Sondag
 */
public interface DataFacilitator {


    public DataMap getData(int time);

    public String getDataIdentifier();

    public String getExperimentName();

    public String getParamaterDescription();
    
    /**
     * Rereads the data from the input file or regenerated it using the specified seed
     * @param seed
     * @return 
     */
    public DataFacilitator reinitializeWithSeed(int seed);

    public boolean hasMaxTime();

    public int getMaxTime();

    public int getLastTime();
    
    public void recolor(HashMap<String, Color> colors);
}

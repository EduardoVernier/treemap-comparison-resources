/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package dataSetClassifier;

import UserControl.Simulator;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import treemap.DataFaciliation.DataFacilitator;
import treemap.dataStructure.DataMap;

/**
 * This class classifies each dataset according to its attributes
 *
 * @author msondag
 */
public class DataSetClassifier {

    List<DataFacilitator> datasets;

    public static void main(String args[]) {

        System.out.println("start reading datasets");
        //eduardo new data
        List<DataFacilitator> datasets = Simulator.getDataFacilitatorFromFolder(new File("D:\\Development\\TreemapStability\\EduardoDatasets\\AllAndFlat"));
        //temp
//        List<DataFacilitator> datasets = Simulator.getDataFacilitatorFromFolder(new File("D:\\Development\\TreemapStability\\dataset-temp"));
//        List<DataFacilitator> datasets = Simulator.getDataFacilitatorFromFolder(new File("D:\\Development\\TreemapStability\\EduardoDatasets\\converted-sports-deep"));

        System.out.println("datasets.size() = " + datasets.size());
        System.out.println("got the datasets");
        DataSetClassifier dsc = new DataSetClassifier(datasets);
        System.out.println("start classifying");

//        dsc.printStatisticsDatasets();
        dsc.classifyDataSets();
    }

    public DataSetClassifier(List<DataFacilitator> datasets) {
        this.datasets = datasets;
    }

    public void printStatisticsDatasets() {
        for (DataFacilitator dataset : datasets) {
            printStatistics(dataset);
        }
    }

    public void printStatistics(DataFacilitator dataset) {
        String name = dataset.getDataIdentifier();
        int maxHeight = getHeight(dataset); //gets the maximum height of the tree        
        int numberOfTimeSteps = getTimeSteps(dataset);//get the number of time steps in the dataset

        //calculate the variation
        List<Double> nodeSizes = getLeafSizes(dataset); //get the distribution of leaf sizes(aggregated over all time)
        double sizeCoefficient = getCoefficentOfVariation(nodeSizes);

        //calculate the changes in sizes
        List<Double> sizeChangeValues = getSumChangeValues(dataset); //returns how much the data changed per timestep.
        double meanSizeChange = getMean(sizeChangeValues);
        double varianceSizeChange = getVariance(sizeChangeValues);

        //calculate the insertion/deletion amount
        List<Double> deletionAmount = getDeletionAmount(dataset); //returns how many leafs were deleted at t+1 normalized on the amount of leafs at time t for all t
        List<Double> additionAmount = getAdditionAmount(dataset); //returns how many leafs were added at t+1 normalized on the amount of leafs at time t for all t
        List<Double> dataChangeValues = new ArrayList();
        for (int i = 0; i < deletionAmount.size(); i++) {
            //add the additions to the deletion number
            double delAmount = deletionAmount.get(i);
            double addAmount = additionAmount.get(i);
            dataChangeValues.add(addAmount + delAmount);
        }

        double meanDataChange = getMean(dataChangeValues);
        double varianceDataChange = getVariance(dataChangeValues);

        String outputString = "";
        outputString += "datasetName;" + name + ";";
        outputString += "height;" + maxHeight + ";";
        outputString += "timesteps;" + numberOfTimeSteps + ";";
        outputString += "sizeCoefficient;" + sizeCoefficient + ";";
        outputString += "meanSizeChange;" + meanSizeChange + ";";
        outputString += "varianceSizeChange;" + varianceSizeChange + ";";
        outputString += "meanDataChange;" + meanDataChange + ";";
        outputString += "varianceDataChange;" + varianceDataChange + ";";

        System.out.println(outputString);
    }

    public void classifyDataSets() {
        StringBuilder sb = new StringBuilder();
        sb.append("Verify that all properties are correct \r\n");
        sb.append("title,"
                  + "maxNodes,"
                  + "timesteps,"
                  + "depthCategory,"
                  + "weightVarianceCategory,"
                  + "weightChangeCategory,"
                  + "InsertionsAndDeletionsChangeCategory"
                  + "\r\n"
        );

        for (DataFacilitator dataset : datasets) {
            classifyDataSet(sb, dataset);
        }

        //write the output
        try (BufferedWriter bw = new BufferedWriter(new FileWriter("D:\\Development\\TreemapStability\\datasetsClassified.csv"))) {
            //header row
            bw.append(sb);
            bw.flush();
        } catch (IOException e) {
            e.printStackTrace();
        }
        System.out.println(sb);

    }

    private void classifyDataSet(StringBuilder sb, DataFacilitator dataset) {
        //stringBuilder will store the classification
        int uniqueNodes = getUniqueNodeAmount(dataset); //get the total amount of nodes in the tree
        int maxNodes = getMaxNodes(dataset); //get the maximum amount of nodes in the tree at a point in time
        int maxHeight = getHeight(dataset); //gets the maximum height of the tree        
        int numberOfTimeSteps = getTimeSteps(dataset);//get the number of time steps in the dataset

        //calculate the changes in sizes
        List<Double> sizeChangeValues = getSumChangeValues(dataset); //returns how much the data changed per timestep.

        //calculate the insertion/deletion amount
        List<Double> deletionAmount = getDeletionAmount(dataset); //returns how many leafs were deleted at t+1 normalized on the amount of leafs at time t for all t
        List<Double> additionAmount = getAdditionAmount(dataset); //returns how many leafs were added at t+1 normalized on the amount of leafs at time t for all t
        List<Double> dataChangeValues = new ArrayList();
        for (int i = 0; i < deletionAmount.size(); i++) {
            //add the additions to the deletion number
            double delAmount = deletionAmount.get(i);
            double addAmount = additionAmount.get(i);
            dataChangeValues.add(addAmount + delAmount);
        }

        //calculate the variation
        List<Double> nodeSizes = getLeafSizes(dataset); //get the distribution of leaf sizes(aggregated over all time)
        double sizeCoefficient = getCoefficentOfVariation(nodeSizes);
        if (Double.isNaN(sizeCoefficient)) {
            System.out.println("nan");
        }

        String depthClassification = classifyDepth(maxHeight);
        String sizeVarianceClassification = classifySizeVariance(sizeCoefficient);
        String sizeChangeClassification = classifySizeChange(sizeChangeValues);
        String dataChangeClassification = classifyDataChange(dataChangeValues);

        writeClassification(sb, dataset.getDataIdentifier(), maxNodes, numberOfTimeSteps, depthClassification, sizeVarianceClassification, sizeChangeClassification, dataChangeClassification);
    }

    private int getUniqueNodeAmount(DataFacilitator dataSet) {
        Set<String> nodeNames = new HashSet();
        for (int time = 0; time < dataSet.getMaxTime(); time++) {
            DataMap data = dataSet.getData(time);
            List<DataMap> allLeafs = data.getAllLeafs();
            for (DataMap dm : allLeafs) {
                nodeNames.add(dm.getLabel());
            }
        }
        return nodeNames.size();
    }

    private int getMaxNodes(DataFacilitator dataSet) {
        int maxNodes = 0;
        for (int time = 0; time < dataSet.getMaxTime(); time++) {
            DataMap data = dataSet.getData(time);
            List<DataMap> allLeafs = data.getAllLeafs();
            maxNodes = Math.max(allLeafs.size(), maxNodes);
        }
        return maxNodes;
    }

    private int getHeight(DataFacilitator dataSet) {
        int maxHeight = 0;
        for (int time = 0; time < dataSet.getMaxTime(); time++) {
            DataMap data = dataSet.getData(time);
            int height = getHeight(data);

            maxHeight = Math.max(height, maxHeight);
        }
        return maxHeight;
    }

    private int getHeight(DataMap data) {
        int maxHeight = 0;
        for (DataMap dm : data.getChildren()) {
            maxHeight = Math.max(getHeight(dm) + 1, maxHeight);
        }
        return maxHeight;
    }

    /**
     * Returns a list of relative sizes. For each leaf node we store the
     * relative size compared to the root
     *
     * @param dataSet
     * @return
     */
    private List<Double> getLeafSizes(DataFacilitator dataSet) {
        List<Double> sizes = new ArrayList();
        for (int time = 0; time < dataSet.getMaxTime(); time++) {
            DataMap data = dataSet.getData(time);
            double sizeT = data.getTargetSize();

            List<DataMap> allLeafs = data.getAllLeafs();
            for (DataMap dm : allLeafs) {
                if (dm.getTargetSize() == 0) {
                    continue;
                }
                double relSize = dm.getTargetSize() / sizeT;
                if (!Double.isNaN(relSize)) {
                    sizes.add(relSize);
                }
            }
        }
        return sizes;
    }

    private int getTimeSteps(DataFacilitator dataset) {
        return dataset.getMaxTime() + 2;
    }

    private List<Double> getSumChangeValues(DataFacilitator dataset) {
        List<Double> sumDataChange = new ArrayList();
        for (int t1 = 0; t1 < (dataset.getMaxTime()); t1++) {
            //get the data change for a single timestep
            double sum = 0;
            DataMap dataT1 = dataset.getData(t1);
            List<DataMap> allLeafsT1 = dataT1.getAllLeafs();

            DataMap dataT2 = dataset.getData(t1 + 1);
            List<DataMap> allLeafsT2 = dataT2.getAllLeafs();

            //get the total weights of all leafs present in both datasets
            double tw1 = 0;
            double tw2 = 0;

            for (DataMap dm1 : allLeafsT1) {
                for (DataMap dm2 : allLeafsT2) {
                    if (dm1.getLabel().equals(dm2.getLabel())) {
                        //dm1 is present in dm2
                        tw1 += dm1.getTargetSize();
                        tw2 += dm2.getTargetSize();
                        //only 1 with identical label
                        break;
                    }
                }
            }

            for (DataMap dm1 : allLeafsT1) {
                for (DataMap dm2 : allLeafsT2) {
                    if (dm1.getLabel().equals(dm2.getLabel())) {
                        //dm1 is present in dm2
                        double w1 = dm1.getTargetSize() / tw1;
                        double w2 = dm2.getTargetSize() / tw2;

                        double absoluteDiff = Math.abs(w2 - w1);
                        sum += absoluteDiff;
                        //only 1 with identical label
                        break;
                    }
                }
            }
            if (!Double.isNaN(sum)) {
                sumDataChange.add(sum);
            }
        }
        return sumDataChange;
    }

    private List<Double> getDeletionAmount(DataFacilitator dataset) {
        List<Double> deletionAmounts = new ArrayList();
        for (int t1 = 0; t1 < (dataset.getMaxTime()); t1++) {
            int amountDeleted = 0;

            DataMap dataT1 = dataset.getData(t1);
            List<DataMap> allLeafsT1 = dataT1.getAllLeafs();

            DataMap dataT2 = dataset.getData(t1 + 1);
            List<DataMap> allLeafsT2 = dataT2.getAllLeafs();

            for (DataMap dm1 : allLeafsT1) {
                boolean found = false;
                for (DataMap dm2 : allLeafsT2) {
                    if (dm1.getLabel().equals(dm2.getLabel())) {
                        found = true;
                        break;//no need to look further for dm2
                    }
                }
                if (!found) {
                    amountDeleted++;
                }
            }

            deletionAmounts.add((double) amountDeleted / (double) allLeafsT1.size());
        }
        return deletionAmounts;
    }

    private List<Double> getAdditionAmount(DataFacilitator dataset) {
        List<Double> addedAmounts = new ArrayList();
        for (int t1 = 0; t1 < (dataset.getMaxTime()); t1++) {
            int amountAdded = 0;

            DataMap dataT1 = dataset.getData(t1);
            List<DataMap> allLeafsT1 = dataT1.getAllLeafs();

            DataMap dataT2 = dataset.getData(t1 + 1);
            List<DataMap> allLeafssT2 = dataT2.getAllLeafs();

            for (DataMap dm2 : allLeafssT2) {
                boolean found = false;
                for (DataMap dm1 : allLeafsT1) {
                    if (dm1.getLabel().equals(dm2.getLabel())) {
                        found = true;
                        break;//no need to look further for dm2
                    }
                }
                if (!found) {
                    amountAdded++;
                }
            }
            addedAmounts.add((double) amountAdded / (double) allLeafsT1.size());
        }
        return addedAmounts;
    }

    private void writeClassification(
            StringBuilder sb, String dataIdentifier, int maxNodes, int numberOfTimeSteps,
            String depthCategory, String sizeVarianceCategory, String dataChangeCategory, String sizeChangeCategory
    ) {
        String dataSetName = "";
        if (dataIdentifier.contains(".")) {//remove the extenstion
            dataSetName = dataIdentifier.substring(dataIdentifier.lastIndexOf("\\") + 1, dataIdentifier.lastIndexOf("."));
        } else {
            dataSetName = dataIdentifier.substring(dataIdentifier.lastIndexOf("\\") + 1);
        }

        sb.append(dataSetName);
        sb.append("," + maxNodes);
        sb.append("," + numberOfTimeSteps);
        sb.append("," + depthCategory);
        sb.append("," + sizeVarianceCategory);
        sb.append("," + dataChangeCategory);
        sb.append("," + sizeChangeCategory);
        sb.append("," + "\r\n");
    }

    private String classifyDepth(int maxHeight) {
        if (maxHeight == 1) {
            return "Single level";
        }
        if (maxHeight == 2 || maxHeight == 3) {
            return "2 or 3 levels";
        }
        if (maxHeight >= 4) {
            return "4+ levels";
        }
        throw new IllegalStateException("height " + maxHeight + " is not valid for classification");
    }

    private String classifySizeVariance(double sizeCoefficient) {
        if (sizeCoefficient > 1) {
            return "High weight variance";
        }
        if (sizeCoefficient < 1) {
            return "Low weight variance";
        }
        if (sizeCoefficient == 1) {
            System.err.println("sizeCoefficient is exactly 1");
            return "Exactly 1";
        }
        throw new IllegalStateException();
    }

    private String classifySizeChange(List<Double> sizeChangeValues) {
        double mean = getMean(sizeChangeValues);
        double sd = getSd(sizeChangeValues);
        double cv = getCoefficentOfVariation(sizeChangeValues);

        if (mean < 0.05 && sd < 0.05) {
            return "Low weight change";
        }
        if (mean < 0.2 && cv < 1) {
            return "Regular weight change";
        }

        return "Spiky weight change";
    }

    private String classifyDataChange(List<Double> dataChangeValues) {
        double mean = getMean(dataChangeValues);
        double sd = getSd(dataChangeValues);
        double cv = getCoefficentOfVariation(dataChangeValues);

        if (mean < 0.05 && sd < 0.05) {
            return "Low insertions and deletions";
        }
        if (mean < 0.2 && cv < 1) {
            return "Regular insertions and deletions";
        }

        return "Spiky insertions and deletions";
    }

    private double getMean(List<Double> doubleList) {

        double sum = 0;
        for (double d : doubleList) {
            if (Double.isNaN(d)) {
                continue;
            }
            sum += d;
        }
        return sum / (double) doubleList.size();
    }

    private double getMedian(List<Double> doubleList) {
        Collections.sort(doubleList);
        return doubleList.get(doubleList.size() / 2);
    }

    private double getMax(List<Double> doubleList) {
        double max = 0;
        for (double d : doubleList) {
            max = Math.max(max, d);
        }
        return max;
    }

    private double getVariance(List<Double> doubleList) {
        double mean = getMean(doubleList);
        double variance = 0;
        for (Double d : doubleList) {
            if (Double.isNaN(d)) {
                continue;
            }
            variance += Math.pow((d - mean), 2);
        };
        variance = variance / (doubleList.size() - 1);
        return variance;
    }

    private double getSd(List<Double> doubleList) {
        return Math.sqrt(getVariance(doubleList));
    }

    private double getCoefficentOfVariation(List<Double> doubleList) {
        double mean = getMean(doubleList);
        double sd = getSd(doubleList);
        double cv = sd / mean;
        return cv;
    }
}

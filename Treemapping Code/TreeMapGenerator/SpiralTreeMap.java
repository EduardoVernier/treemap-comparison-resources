package TreeMapGenerator;

import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import treemap.dataStructure.DataMap;
import treemap.dataStructure.Rectangle;
import treemap.dataStructure.TreeMap;

/**
 *
 * @author max
 */
public class SpiralTreeMap implements TreeMapGenerator {

    @Override
    public String getParamaterDescription() {
        return "";
    }

    public enum Orientation {

        TOP, RIGHT, BOTTOM, LEFT;

        public Orientation nextOrientation() {
            if (this == Orientation.TOP) {
                return Orientation.RIGHT;
            }
            if (this == Orientation.RIGHT) {
                return Orientation.BOTTOM;
            }
            if (this == Orientation.BOTTOM) {
                return Orientation.LEFT;
            }
            if (this == Orientation.LEFT) {
                return Orientation.TOP;
            }
            //never happens
            return null;
        }
    };

    @Override
    public TreeMap generateTreeMap(DataMap dataMap, Rectangle treeMapRectangle) {
        List<TreeMap> children = new LinkedList();
        //If the dataMap has children we will recurse into them to generate
        //those treeMaps first
        if (dataMap.hasChildren()) {
            //get the rectangles for all the children according to the strip layout
            Map<DataMap, Rectangle> rectangleMapping = generateLevel(dataMap.getChildren(), treeMapRectangle, Orientation.TOP);

            //recurse into the children and generate the rectangles for them
            //Add the children treemaps to the list
            for (DataMap dm : dataMap.getChildren()) {
                Rectangle r = rectangleMapping.get(dm);
                TreeMap child = generateTreeMap(dm, r);
                children.add(child);
            }
        }
        TreeMap tm = new TreeMap(treeMapRectangle, dataMap.getLabel(), dataMap.getColor(), dataMap.getTargetSize(), children);
        return tm;
    }

    public Map<DataMap, Rectangle> generateLevel(List<DataMap> children, Rectangle treeMapRectangle, Orientation orientation) {
        Map<DataMap, Rectangle> rectangleMapping = new HashMap<>();

        double totalSize = DataMap.getTotalSize(children);
        double totalArea = treeMapRectangle.getWidth() * treeMapRectangle.getHeight();

        //factor which we need to scale the areas for each rectangle
        double scaleFactor = totalArea / totalSize;

        Rectangle remainingRectangle = treeMapRectangle;

        List<DataMap> remainingDataMaps = new LinkedList();
        remainingDataMaps.addAll(children);

        while (!remainingDataMaps.isEmpty()) {
            //generate a new strip
            List<DataMap> bestStrip = getBestStrip(remainingDataMaps, treeMapRectangle, scaleFactor, orientation);

            //figure out the postitioning of the stirps
            Map<DataMap, Rectangle> rectanglesFromStrip = getRectanglesFromStrip(bestStrip, scaleFactor, remainingRectangle, orientation);
            rectangleMapping.putAll(rectanglesFromStrip);

            //update the remaining rectangle
            Rectangle[] stripRectangles = rectanglesFromStrip.values().toArray(new Rectangle[0]);
            remainingRectangle = getRemainingRectangle(remainingRectangle, stripRectangles, orientation);

            //prepare for the next strip
            remainingDataMaps.removeAll(bestStrip);
            orientation = orientation.nextOrientation();
        }
        //generate the strips and return them
        return rectangleMapping;
    }

    protected List<DataMap> getBestStrip(List<DataMap> remainingDataMaps, Rectangle remainingRectangle, double scaleFactor, Orientation orientation) {
        //holds the new strip
        List<DataMap> strip = new LinkedList();

        for (DataMap dm : remainingDataMaps) {
            //check if adding the next rectangle would degenerate the aspect ratio
            List<DataMap> newStrip = new LinkedList();
            newStrip.addAll(strip);
            newStrip.add(dm);
            double currentAspectRatio = getStripAspectRatio(strip, remainingRectangle, scaleFactor, orientation);
            double newAspectRatio = getStripAspectRatio(newStrip, remainingRectangle, scaleFactor, orientation);

            if (newAspectRatio <= currentAspectRatio) {
                //aspect ratio improved so we add it
                strip.add(dm);
            } else {
                //adding does not improve it anymore so we return
                break;
            }
        }
        return strip;
    }

    protected Rectangle getRemainingRectangle(Rectangle initialRectangle, Rectangle[] stripRectangles, Orientation orientation) {

        double x = initialRectangle.getX();
        double y = initialRectangle.getY();
        double width = initialRectangle.getWidth();
        double height = initialRectangle.getHeight();

        //if it was filled horizontally we only need to change in the y direction
        //if not we only need to change in the x direction
        if (orientation == Orientation.TOP) {
            //all items in the list have the same height
            double fillHeight = stripRectangles[0].getHeight();
            y = stripRectangles[0].getY2();
            height -= fillHeight;
        }
        if (orientation == Orientation.BOTTOM) {
            //all items in the list have the same height
            double fillHeight = stripRectangles[0].getHeight();
            height -= fillHeight;
        }
        if (orientation == Orientation.LEFT) {
            //all items in the list have the same width
            double fillWidth = stripRectangles[0].getWidth();
            x = stripRectangles[0].getX2();
            width -= fillWidth;
        }
        if (orientation == Orientation.RIGHT) {
            //all items in the list have the same width
            double fillWidth = stripRectangles[0].getWidth();
            width -= fillWidth;
        }
        return new Rectangle(x, y, width, height);
    }

    protected Map<DataMap, Rectangle> getRectanglesFromStrip(List<DataMap> dmList, double scaleFactor, Rectangle treeMapRectangle, Orientation orientation) {
        Map<DataMap, Rectangle> rectangleMapping = new HashMap();
        //holds the side that is of a fixed side for the entire strip
        double sideLength;
        double totalArea = DataMap.getTotalSize(dmList);
        if (orientation == Orientation.BOTTOM || orientation == Orientation.TOP) {
            //calculate the height
            sideLength = totalArea * scaleFactor / treeMapRectangle.getWidth();
        } else {
            //calculate the width
            sideLength = totalArea * scaleFactor / treeMapRectangle.getHeight();
        }

        double curX;
        double curY;
        if (orientation == Orientation.TOP) {//fill left to right
            curX = treeMapRectangle.getX();
            curY = treeMapRectangle.getY();
        } else if (orientation == Orientation.BOTTOM) {//fill from right to left
            curX = treeMapRectangle.getX() + treeMapRectangle.getWidth();
            curY = treeMapRectangle.getY() + treeMapRectangle.getHeight() - sideLength;
        } else if (orientation == Orientation.RIGHT) {//fill top to bottom
            curX = treeMapRectangle.getX() + treeMapRectangle.getWidth() - sideLength;
            curY = treeMapRectangle.getY();
        } else {//orientation == Orientation.LEFT //fill bottom to top
            curX = treeMapRectangle.getX();
            curY = treeMapRectangle.getY() + treeMapRectangle.getHeight();
        }

        for (DataMap dm : dmList) {
            //calculates how much of the strip this dataMap takes up
            double otherSide = dm.getTargetSize() * scaleFactor / sideLength;
            Rectangle r;
            //creates a rectangle and moves the startposition further to
            //accomodate the next rectangle
            if (orientation == Orientation.TOP) {
                r = new Rectangle(curX, curY, otherSide, sideLength);
                curX += otherSide;
            } else if (orientation == Orientation.BOTTOM) {
                r = new Rectangle(curX - otherSide, curY, otherSide, sideLength);
                curX -= otherSide;
            } else if (orientation == Orientation.RIGHT) {
                r = new Rectangle(curX, curY, sideLength, otherSide);
                curY += otherSide;
            } else {//(orientation == Orientation.LEFT) 
                r = new Rectangle(curX, curY - otherSide, sideLength, otherSide);
                curY -= otherSide;
            }
            rectangleMapping.put(dm, r);
        }
        return rectangleMapping;
    }

    protected double getStripAspectRatio(List<DataMap> strip, Rectangle treeMapRectangle, double scaleFactor, Orientation orientation) {

        if (strip.isEmpty()) {
            return Double.MAX_VALUE;
        }

        double totalSize = DataMap.getTotalSize(strip);

        //holds how long the fixed side is
        double sideLength;

        if (orientation == Orientation.BOTTOM || orientation == Orientation.TOP) {
            sideLength = totalSize * scaleFactor / treeMapRectangle.getWidth();
        } else {
            sideLength = totalSize * scaleFactor / treeMapRectangle.getHeight();
        }

        double maxAspectRatio = 0;
        for (DataMap dm : strip) {
            double otherSide = dm.getTargetSize() * scaleFactor / sideLength;
            double aspectRatio = Math.max(sideLength / otherSide, otherSide / sideLength);
            maxAspectRatio = Math.max(aspectRatio, maxAspectRatio);
        }
        return maxAspectRatio;
    }

    @Override
    public TreeMapGenerator reinitialize() {
        return this;
    }
}

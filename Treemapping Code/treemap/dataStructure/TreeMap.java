package treemap.dataStructure;

import java.awt.Color;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedList;
import java.util.List;

/**
 *
 * @author Max Sondag
 */
public class TreeMap {

    /**
     * Find the smallest enclosing rectangle of all treemaps in the list
     *
     * @param tmList
     * @return
     */
    public static Rectangle findEnclosingRectangle(List<TreeMap> tmList) {
        //Find the enclosing rectangle of all the treeMap
        double x1 = Double.MAX_VALUE;
        double y1 = Double.MAX_VALUE;
        double x2 = 0;
        double y2 = 0;

        for (TreeMap tm : tmList) {
            Rectangle r = tm.getRectangle();
            x1 = Math.min(x1, r.getX());
            y1 = Math.min(y1, r.getY());
            x2 = Math.max(x2, r.getX2());
            y2 = Math.max(y2, r.getY2());
        }

        return new Rectangle(x1, y1, x2 - x1, y2 - y1);
    }

    /**
     * The position of this treemap
     */
    private Rectangle rectangle;
    /**
     * The color that the rectangle should have
     */
    private Color color;
    /**
     * The unique label of this treemapF
     */
    private String label;
    /**
     * The size we want this rectangle to have
     */
    private double targetSize;
    /**
     * The size this rectangle currently has.
     */
    private double actualSize;
    /**
     * The children of this treemap
     */
    private List<TreeMap> children;

    /**
     * The parent of this treemap. null if it does not exist
     */
    private TreeMap parent;

    public TreeMap(Rectangle rectangle, String label, Color color, double targetSize, List<TreeMap> children) {
        this.rectangle = rectangle;
        this.label = label;
        this.color = color;
        this.children = children;
        this.targetSize = targetSize;
        this.actualSize = targetSize;
        if (children == null) {
            this.children = new LinkedList();
        } else {
            for (TreeMap tm : children) {
                tm.setParent(this);
            }
        }
        this.parent = null;
    }

    public TreeMap(Rectangle rectangle, String label, Color color, double targetSize, double actualSize, List<TreeMap> children) {
        this.rectangle = rectangle;
        this.label = label;
        this.color = color;
        this.children = children;
        this.targetSize = targetSize;
        this.actualSize = actualSize;
        if (children == null) {
            this.children = new LinkedList();
        } else {
            for (TreeMap tm : children) {
                tm.setParent(this);
            }
        }
        this.parent = null;
    }

    /**
     * Returns whether this treemap has children
     *
     * @return
     */
    public boolean hasChildren() {
        if (children == null || children.isEmpty()) {
            return false;
        }
        return true;
    }

    public List<TreeMap> getAllChildren() {
        if (!hasChildren()) {
            return new LinkedList();
        }
        List<TreeMap> childList = new LinkedList<>();
        for (TreeMap tm : children) {
            childList.addAll(tm.getAllChildren());
            childList.add(tm);
        }

        return childList;
    }

    /**
     * Returns the child with the given label. Does not search recursively
     *
     * @param label
     * @return
     */
    public TreeMap getChildWithLabel(String label) {
        if (!hasChildren()) {
            return null;
        }
        for (TreeMap tm : children) {
            if (tm.getLabel().equals(label)) {
                return tm;
            }
        }
        return null;
    }

    /**
     * Returns the treemap with the given label
     *
     * @param label
     * @return
     */
    public TreeMap getTreeMapWithLabel(String label) {
        if (this.label.equals(label)) {
            return this;
        }
        for (TreeMap tm : children) {
            //recurse
            TreeMap recurseTm = tm.getTreeMapWithLabel(label);
            if (recurseTm != null) {
                return recurseTm;
            }
        }
        return null;
    }

    public List<TreeMap> getAllLeafs() {
        List<TreeMap> childList = new LinkedList<>();
        if (!hasChildren()) {
            childList.add(this);
            return childList;
        }
        for (TreeMap tm : children) {
            childList.addAll(tm.getAllLeafs());
        }
        return childList;
    }

    /**
     * Returns the maximum aspect ratio of a leaf node in the treeMap
     *
     * @return
     */
    public double getMaxAspectRatio() {
        List<TreeMap> allLeafs = getAllLeafs();
        double maxAspectRatio = Double.MIN_VALUE;
        for (TreeMap tm : allLeafs) {
            maxAspectRatio = Math.max(maxAspectRatio, tm.getRectangle().getAspectRatio());
        }
        return maxAspectRatio;
    }

    /**
     * Returns whether this treemap has the same structure as another treemap.
     *
     * @param tm
     * @return
     */
    public boolean equalStructure(TreeMap tm) {
        if (tm == this) {
            return true;
        }
        if (!tm.getColor().equals(getColor())) {
            return false;
        }
        if (!tm.getLabel().equals(getLabel())) {
            return false;
        }
        //treemaps themselves are fine, comparing their child structure now
        if (tm.hasChildren() != hasChildren()) {
            return true;
        }
        //they both either have or don't have children
        if (!hasChildren()) {
            return true;
        }
        //they have children

        //verify that they have the same amount of children
        if (tm.getAllChildren().size() != getAllChildren().size()) {
            return false;
        }

        //if we can find a matching child in tm for every child this treemap has
        //then the two treemap share the same structure as they must then have the 
        //same children
        for (TreeMap child : children) {
            boolean found = false;

            for (TreeMap childrenTm : tm.getChildren()) {
                if (childrenTm.equalStructure(child)) {
                    found = true;
                    break;
                }
            }
            if (found == false) {
                return false;
            }
        }

        //we could find a matching treemap in tm.getChildren for every child in children
        return true;
    }

    //<editor-fold defaultstate="collapsed" desc="getters">
    /**
     * @return the rectangle
     */
    public Rectangle getRectangle() {
        return rectangle;
    }

    /**
     * @return the color
     */
    public Color getColor() {
        return color;
    }

    /**
     * @return the label
     */
    public String getLabel() {
        return label;
    }

    public double getTargetSize() {
        return targetSize;
    }

    public double getActualSize() {
        return actualSize;
    }

    /**
     * @return the children
     */
    public List<TreeMap> getChildren() {
        return children;
    }

    /**
     * Returns the average aspect ratio of all(including non-leaf) nodes
     *
     * @return
     */
    public double getAverageAspectRatio() {
        double sumAspectRatio = 0;
        List<TreeMap> allChildren = getAllChildren();
        for (TreeMap tm : allChildren) {
            sumAspectRatio += tm.rectangle.getAspectRatio();
        }

        return sumAspectRatio / (allChildren.size() + 1);
    }

    /**
     * Returns the total aspect ratio of all leaf nodes
     *
     * @return
     */
    public double getTotalAspectRatio() {
        double sumAspectRatio = 0;
        List<TreeMap> leafs = getAllLeafs();
        for (TreeMap tm : leafs) {
            sumAspectRatio += tm.rectangle.getAspectRatio();
        }

        return sumAspectRatio;
    }

    /**
     * Returns the median aspect ratio of all (including non-leaf) nodes,
     *
     * @return
     */
    public double getMedianAspectRatio() {

        ArrayList<Double> aspectRatios = new ArrayList();

        List<TreeMap> allChildren = getAllChildren();
        for (TreeMap tm : allChildren) {
            aspectRatios.add(tm.rectangle.getAspectRatio());
        }

        aspectRatios.sort((Double o1, Double o2) -> Double.compare(o1, o2));

        int middleElement = (int) aspectRatios.size() / 2;
        return aspectRatios.get(middleElement);
    }

    /**
     * @return
     */
    public TreeMap getParent() {
        return parent;
    }

    /**
     * Returns the height of this node
     *
     * @return
     */
    public int getHeight() {
        int height = 0;
        for (TreeMap tm : getChildren()) {
            height = Math.max(tm.getHeight() + 1, height);
        }
        return height;
    }

    //</editor-fold>
    /**
     * Updates the targetsize of this treemap
     *
     * @param targetSize
     */
    public void updateTargetSize(double targetSize) {
        this.targetSize = targetSize;
    }

    /**
     * Updates the rectangle of this treemap and makes sure that all its
     * children are still in the rectangle of this treemap in the same
     * proportions.
     * Does NOT update the actualsize
     *
     * @param newR
     */
    public void updateRectangle(Rectangle newR) {

        double newX = newR.getX();
        double newY = newR.getY();
        double newW = newR.getWidth();
        double newH = newR.getHeight();

        if (rectangle != null) {
            //if it already had a rectangle
            double oldX = rectangle.getX();
            double oldW = rectangle.getWidth();

            double oldY = rectangle.getY();
            double oldH = rectangle.getHeight();

            double scaleWidth = newW / oldW;
            double scaleHeight = newH / oldH;

            for (TreeMap child : children) {
                //X percentage indicates at what percentages of the width/height 
                //the child used to be.
                double oldChildX = child.getRectangle().getX();
                double xPercent = (oldChildX - oldX) / (oldW);

                double oldChildY = child.getRectangle().getY();
                double yPercent = (oldChildY - oldY) / (oldH);

                double oldChildW = child.getRectangle().getWidth();
                double oldChildH = child.getRectangle().getHeight();

                //update the rectangle of the child
                double newChildX = newX + newW * xPercent;
                double newChildY = newY + newH * yPercent;
                double newChildW = oldChildW * scaleWidth;
                double newChildH = oldChildH * scaleHeight;
                Rectangle newRectangle = new Rectangle(newChildX, newChildY, newChildW, newChildH);
                //recurse in the child
                child.updateRectangle(newRectangle);
            }
        }
        //update this rectangle
        this.rectangle = new Rectangle(newR);
    }

    /**
     * Removes the treemaps from the structure and updates the weights
     *
     * @param treemaps
     */
    public void removeTreeMaps(List<TreeMap> treemaps) {
        while (!treemaps.isEmpty()) {
            TreeMap tm = treemaps.get(0);
            removeTreeMap(tm.getLabel());
            treemaps.remove(tm);
        }
    }

    /**
     * Removes the treemap with the given label and updates the weights
     *
     * @param label
     */
    public void removeTreeMap(String label) {
        TreeMap childToRemove = getTreeMapWithLabel(label);
//        this.label = this.label.replace(label, "");

        if (getAllChildren().contains(childToRemove)) {
            targetSize -= childToRemove.targetSize;

            //it is a child of this treemap remove it
            if (children.contains(childToRemove)) {
                children.remove(childToRemove);
            } else {
                //recurse to find it
                for (TreeMap child : children) {
                    child.removeTreeMap(label);
                }
            }
        }

    }

    /**
     * Adds child to parent and updates all the targetsizes of the treemaps
     *
     * @param addChild
     * @param parent
     */
    public void addTreeMap(TreeMap addChild, TreeMap parent) {

        //adjust the size
        if (getAllChildren().contains(parent)) {
            targetSize += addChild.targetSize;
        } else {
            if (this == parent) {
                targetSize += addChild.targetSize;
                children.add(addChild);
                addChild.setParent(this);
            }
            //either found or there is no parent
            return;
        }

        for (TreeMap child : children) {
            child.addTreeMap(addChild, parent);
        }

    }

    /**
     * Adds a list of treemaps to a parent and updates the targetsizes
     *
     * @param updatedChildList
     * @param parent
     */
    public void addTreeMaps(List<TreeMap> updatedChildList, TreeMap parent) {
        for (TreeMap child : updatedChildList) {
            addTreeMap(child, parent);
        }
    }

    /**
     * Sets the parent of this treemap
     *
     * @param parent
     */
    public void setParent(TreeMap parent) {
        this.parent = parent;
    }

    /**
     * Makes a deepcopy of this treemap
     *
     * @return
     */
    public TreeMap deepCopy() {
        List<TreeMap> copyChildren = new ArrayList();
        for (TreeMap tm : children) {
            copyChildren.add(tm.deepCopy());
        }

        TreeMap copyTm = new TreeMap(rectangle.deepCopy(), label, color, targetSize, copyChildren);
        return copyTm;
    }

    /**
     * Returns the first parent that has more than 1 child
     *
     * @return
     */
    public TreeMap getFirstLevelParent() {

        TreeMap parent = getParent();
        if (parent == null) {
            //tm is the root
            return null;
        }

        while (parent.getChildren().size() == 1) {
            parent = parent.getParent();
            if (parent == null) {
                //first level parent is the root
                return null;
            }
        }
        //parent has more than 1 child
        return parent;
    }

    /**
     * Replaces the treemap with the same label as {@code replaceTm} with
     * {@code replaceTm}.
     *
     * @param replaceTm
     */
    public void replaceTreemap(TreeMap replaceTm) {
        TreeMap originalTm = getTreeMapWithLabel(replaceTm.getLabel());
        removeTreeMap(replaceTm.getLabel());
        addTreeMap(replaceTm, originalTm.parent);
    }

    /**
     * Removes all children of the treemap with label newDm and updates the
     * sizes
     *
     * @param newDm
     * @param oldSize
     * @return
     */
    public boolean removeChildrenAndUpdateSize(DataMap newDm, double oldSize) {
        if (label.equals(newDm.getLabel())) {
            children = new ArrayList();
            targetSize = newDm.getTargetSize();
            return true;
        } else {
            for (TreeMap child : children) {
                boolean contained = child.removeChildrenAndUpdateSize(newDm, oldSize);
                if (contained) {
                    this.targetSize += (newDm.getTargetSize() - oldSize);
                    return true;
                }
            }
        }
        return false;
    }

    /**
     * sets the actualSize of the treemap
     *
     * @param actualSize
     */
    public void setActualSize(double actualSize) {
        this.actualSize = actualSize;
    }

    /**
     * Updates all the target sizes for the treemap. newDm and this treemap
     * should have the exact same structure
     *
     * @param newDm
     */
    public void setTargetSizes(DataMap newDm) {
        targetSize = newDm.getTargetSize();

        for (TreeMap childTm : children) {
            DataMap childDm = newDm.getDataMapWithLabel(childTm.getLabel());
            childTm.setTargetSizes(childDm);
        }
    }

}

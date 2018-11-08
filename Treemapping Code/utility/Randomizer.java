/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package utility;

import java.util.Random;

/**
 *
 * @author msondag
 */
public class Randomizer {

    private static Random random;
    private static boolean init = false;

    public static double getRandomDouble() {
        if (init == false) {
            init = true;
            random = new Random(0);
        }
        return random.nextDouble();
    }

    /**
     * Returns a random int between lowerbound (inclusive) and upperbound(exclusive)
     * @param lowerBound
     * @param upperBound
     * @return 
     */
    public static int getRandomInt(int lowerBound, int upperBound) {
        double range = upperBound - lowerBound;
        if (range <= 1) {
            throw new IllegalArgumentException("Range between upperBound and lowerBound must be larger than 1");
        }

        if (init == false) {
            init = true;
            random = new Random(0);
        }
        double val = random.nextDouble();

        //every 1/range from val adds 1 to the return value
        int returnInt = (int) Math.floor(val / (1 / range));
        return returnInt;
    }
}

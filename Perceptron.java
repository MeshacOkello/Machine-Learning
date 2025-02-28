public class Perceptron {
    private double[] weight;

    // Creates a perceptron with n inputs. It should create an array
    // of n weights and initialize them all to 0.
    public Perceptron(int n) {
        weight = new double[n];

    }

    // Returns the number of inputs n.
    public int numberOfInputs() {
        return weight.length;
    }

    // Returns the weighted sum of the weight vector and x.
    public double weightedSum(double[] x) {
        int n = x.length;
        if (x.length != weight.length) {
            throw new IllegalArgumentException(
                    "Input array length does not match number of weights.");
        }
        double sum = 0;
        for (int i = 0; i < n; i++) {
            sum = sum + x[i] * weight[i];
        }
        return sum;
    }

    // Predicts the binary label (+1 or -1) of input x. It returns +1
    // if the weighted sum is positive and -1 if it is negative (or zero).
    public int predict(double[] x) {
        if (weightedSum(x) > 0) {
            return +1;
        }
        return -1;
    }

    // Trains this perceptron on the binary labeled (+1 or -1) input x.
    // The weights vector is updated accordingly.
    public void train(double[] x, int binaryLabel) {
        int prediction = predict(x);
        if (prediction == binaryLabel && binaryLabel > 0)
            return;
        for (int j = 0; j < weight.length; j++) {
            if (prediction == 1 && binaryLabel == -1) {
                // False positive: predicted +1 but label is -1
                weight[j] = weight[j] - x[j];
            }
            else if (prediction == -1 && binaryLabel == 1) {
                // False negative: predicted -1 but label is +1
                weight[j] = weight[j] + x[j];
            }
        }
    }

    // Returns a String representation of the weight vector, with the
    // individual weights separated by commas and enclosed in parentheses.
    // Example: (2.0, 1.0, -1.0, 5.0, 3.0)
    public String toString() {
        String result = "(";
        for (int i = 0; i < weight.length; i++) {
            result = result + weight[i];
            if (i < weight.length - 1) {
                result = result + ", ";
            }
        }
        result += ")";
        return result;
    }

    // Tests this class by directly calling all instance methods.
    public static void main(String[] args) {
        // Create a perceptron with 3 inputs
        Perceptron p = new Perceptron(3);

        // Print initial weights
        System.out.println("Initial weights: " + p);

        // Sample input data
        double[] x1 = { 1.0, 2.0, 3.0 };
        double[] x2 = { 2.0, 1.0, 0.0 };

        // Train the perceptron with labeled data
        p.train(x1, 1);  // Positive example
        System.out.println("Weights after training with x1 (label +1): " + p);

        p.train(x2, -1); // Negative example
        System.out.println("Weights after training with x2 (label -1): " + p);

        // Make predictions
        System.out.println("Prediction for x1: " + p.predict(x1)); // Should output +1
        System.out.println("Prediction for x2: " + p.predict(x2)); // Should output -1
    }
}

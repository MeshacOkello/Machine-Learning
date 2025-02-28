public class MultiPerceptron {
    private Perceptron[] perceptrons;

    // Creates a multi-perceptron object with m classes and n inputs.
    // It creates an array of m perceptrons, each with n inputs.
    public MultiPerceptron(int m, int n) {
        perceptrons = new Perceptron[m];
        for (int i = 0; i < m; i++) {
            perceptrons[i] = new Perceptron(n);
        }
    }

    // Returns the number of classes m.
    public int numberOfClasses() {
        return perceptrons.length;
    }

    // Returns the number of inputs n (length of the feature vector).
    public int numberOfInputs() {
        return perceptrons[0].numberOfInputs();
    }

    // Returns the predicted class label (between 0 and m-1) for the given input.
    public int predictMulti(double[] x) {
        int predictedClass = 0;
        double highestScore = perceptrons[0].predict(x);

        // Iterate through each perceptron to get its prediction score for the input
        for (int i = 1; i < perceptrons.length; i++) {
            double score = perceptrons[i].predict(x);
            if (score > highestScore) {
                highestScore = score;
                predictedClass = i;
            }
        }
        return predictedClass;
    }

    // Trains this multi-perceptron on the labeled (between 0 and m-1) input.
    public void trainMulti(double[] x, int classLabel) {
        for (int i = 0; i < perceptrons.length; i++) {
            // If the current perceptron corresponds to the correct class label, train it with target output 1
            if (i == classLabel) {
                perceptrons[i].train(x, 1);
            }
            // Otherwise, train it with target output -1 (negative label)
            else {
                perceptrons[i].train(x, -1);
            }
        }
    }

    // Returns a String representation of this MultiPerceptron, with
    // the string representations of the perceptrons separated by commas
    // and enclosed in parentheses.
    // Example with m = 2 and n = 3: ((2.0, 0.0, -2.0), (3.0, 4.0, 5.0))
    public String toString() {
        String result = "(";

        for (int i = 0; i < perceptrons.length; i++) {
            result += perceptrons[i].toString();
            if (i < perceptrons.length - 1) {
                result += ", ";
            }
        }

        result += ")";
        return result;
    }

    // Tests this class by directly calling all instance methods.
    public static void main(String[] args) {
        // Create a MultiPerceptron with 2 classes and 3 inputs per perceptron
        MultiPerceptron multiPerceptron = new MultiPerceptron(2, 3);

        // Test numberOfClasses method
        System.out.println(
                "Number of classes: " + multiPerceptron.numberOfClasses()); // Expected: 2

        // Test numberOfInputs method
        System.out.println("Number of inputs: " + multiPerceptron.numberOfInputs()); // Expected: 3

        // Test predictMulti method with a sample input
        double[] input = { 1.0, -1.5, 0.5 };
        int predictedClass = multiPerceptron.predictMulti(input);
        System.out.println("Predicted class: " + predictedClass);

        // Test trainMulti method with a sample input and class label
        int correctClass = 1;
        multiPerceptron.trainMulti(input, correctClass);
        System.out.println("Training completed for class label " + correctClass);

        // Test toString method
        System.out.println("MultiPerceptron representation: " + multiPerceptron);
    }
}

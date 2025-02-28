import java.awt.Color;

public class ImageClassifier {
    private String configFile;
    private int width;  // Width of the images
    private int height; // Height of the images
    private String[] classes; // Array to store class names
    private int numClasses; // Number of classes
    private MultiPerceptron multiperceptron;

    // Uses the provided configuration file to create an
    // ImageClassifier object.
    public ImageClassifier(String configFile) {
        this.configFile = configFile;
        In configFileData = new In(configFile);

        this.width = configFileData.readInt();
        this.height = configFileData.readInt();
        this.numClasses = configFileData.readInt();
        classes = new String[numClasses];
        for (int i = 0; i < this.numClasses; i++) {
            classes[i] = configFileData.readString();
        }

        multiperceptron = new MultiPerceptron(numClasses, width * height);

    }

    // Creates a feature vector (1D array) from the given picture.
    public double[] extractFeatures(Picture picture) {
        if (picture.width() != width || picture.height() != height) {
            throw new IllegalArgumentException(
                    "Image dimensions do not match configuration dimensions.");
        }
        double[] inputFeatures = new double[width * height];
        int featureIndex = 0;
        for (int row = 0; row < picture.height(); row++)
            for (int col = 0; col < picture.width(); col++) {
                Color color = picture.get(col, row);
                inputFeatures[featureIndex] = color.getRed();
                featureIndex += 1;
            }
        return inputFeatures;

    }

    // Trains the perceptron on the given training data file.
    public void trainClassifier(String trainFile) {
        In train = new In(trainFile);
        while (!train.isEmpty()) {
            String fileName = train.readString();
            double[] features = this.extractFeatures(new Picture(fileName));
            int label = train.readInt();
            this.multiperceptron.trainMulti(features, label);
        }
    }

    // Returns the name of the class for the given class label.
    public String classNameOf(int classLabel) {
        if (classLabel < 0 || classLabel > classes.length - 1) {
            throw new IllegalArgumentException("Invalid class label");
        }
        return classes[classLabel];
    }

    // Returns the predicted class for the given picture.
    public int classifyImage(Picture picture) {
        double[] inputFeatures = extractFeatures(picture);
        return multiperceptron.predictMulti(inputFeatures);
    }

    // Returns the error rate on the given testing data file.
    // Also prints the misclassified examples - see specification.
    public double testClassifier(String testFile) {
        double numTest = 0;
        double numMissed = 0;
        In testFileContent = new In(testFile);
        while (!testFileContent.isEmpty()) {
            String imageFile = testFileContent.readString();
            numTest++;
            int classLabel = testFileContent.readInt();
            int predictedClass = classifyImage(new Picture(imageFile));
            if (predictedClass != classLabel) {
                numMissed++;
                StdOut.printf("%s, label=%s, predict=%s\n", imageFile,
                              classNameOf(classLabel), classNameOf(predictedClass));
            }
        }
        return numMissed / numTest;
    }

    // Tests this class using a configuration file, training file and test file.
    // See below.
    public static void main(String[] args) {
        ImageClassifier classifier = new ImageClassifier(args[0]);
        classifier.trainClassifier(args[1]);
        double testErrorRate = classifier.testClassifier(args[2]);
        System.out.println("test error rate = " + testErrorRate);
    }
}

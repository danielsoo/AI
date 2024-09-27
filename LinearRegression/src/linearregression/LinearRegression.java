package linearregression;

import java.util.Arrays;

/**
 * Linear Regression model implemented as a white box model.
 * Implements the Normal Equation method.
 * 
 * Author: younsoopark
 */

//***could you help me out how should I fix my errors?***
//***I watched lots of Youtube sources and asked for help to ChatGPT too but still I have no idea what should I do.***




public class LinearRegression {
    private double[] theta; // Parameters (including intercept)

    /**
     * Fits the Linear Regression model to the data using the Normal Equation.
     *
     * @param X Feature matrix (each row represents a sample, each column a feature)
     * @param y Target vector
     */
    public void fit(double[][] X, double[] y) {
        // Add a column of ones to X for the intercept term
        double[][] X_b = addIntercept(X);

        // Convert to Matrix format
        Matrix X_matrix = new Matrix(X_b);
        Matrix y_matrix = new Matrix(y.length, 1);
        for (int i = 0; i < y.length; i++) {
            y_matrix.setValue(i, 0, y[i]);
        }

        // Compute theta using Normal Equation: theta = (X^T X)^-1 X^T y
        Matrix X_transpose = X_matrix.transpose();
        Matrix X_transpose_X = X_transpose.multiply(X_matrix);
        Matrix X_transpose_X_inv = X_transpose_X.inverse();
        Matrix X_transpose_y = X_transpose.multiply(y_matrix);
        Matrix theta_matrix = X_transpose_X_inv.multiply(X_transpose_y);

        // Store theta parameters
        theta = new double[theta_matrix.getRows()];
        for (int i = 0; i < theta.length; i++) {
            theta[i] = theta_matrix.getValue(i, 0);
        }
    }

    /**
     * Predicts the target values for given input features.
     *
     * @param X Feature matrix
     * @return Predicted target values
     */
    public double[] predict(double[][] X) {
        double[][] X_b = addIntercept(X);
        double[] predictions = new double[X_b.length];

        for (int i = 0; i < X_b.length; i++) {
            double prediction = 0.0;
            for (int j = 0; j < theta.length; j++) {
                prediction += theta[j] * X_b[i][j];
            }
            predictions[i] = prediction;
        }

        return predictions;
    }

    /**
     * Calculates the Mean Squared Error between actual and predicted values.
     *
     * @param y_true Actual target values
     * @param y_pred Predicted target values
     * @return Mean Squared Error
     */
    public double meanSquaredError(double[] y_true, double[] y_pred) {
        if (y_true.length != y_pred.length) {
            throw new IllegalArgumentException("Arrays must have the same length.");
        }

        double sum = 0.0;
        for (int i = 0; i < y_true.length; i++) {
            double error = y_true[i] - y_pred[i];
            sum += error * error;
        }

        return sum / y_true.length;
    }

    /**
     * Adds a column of ones to the feature matrix for the intercept term.
     *
     * @param X Original feature matrix
     * @return Feature matrix with intercept
     */
    private double[][] addIntercept(double[][] X) {
        double[][] X_b = new double[X.length][X[0].length + 1];
        for (int i = 0; i < X.length; i++) {
            X_b[i][0] = 1.0; // Intercept term
            System.arraycopy(X[i], 0, X_b[i], 1, X[i].length);
        }
        return X_b;
    }

    /**
     * Returns the model parameters (theta).
     *
     * @return Model parameters
     */
    public double[] getTheta() {
        return theta;
    }

    /**
     * Matrix class for basic matrix operations.
     * Declared as static to allow instantiation without an instance of LinearRegression.
     */
    public static class Matrix {
        private final int rows;
        private final int cols;
        private final double[][] data;

        /**
         * Constructs a Matrix with specified data.
         *
         * @param data 2D array representing matrix data
         */
        public Matrix(double[][] data) {
            this.rows = data.length;
            this.cols = data[0].length;
            this.data = new double[rows][cols];
            for(int i=0;i<rows;i++) {
                System.arraycopy(data[i], 0, this.data[i], 0, cols);
            }
        }

        /**
         * Constructs a zero matrix with specified rows and columns.
         *
         * @param rows Number of rows
         * @param cols Number of columns
         */
        public Matrix(int rows, int cols) {
            this.rows = rows;
            this.cols = cols;
            this.data = new double[rows][cols];
        }

        /**
         * Returns the number of rows.
         *
         * @return Number of rows
         */
        public int getRows() {
            return rows;
        }

        /**
         * Returns the number of columns.
         *
         * @return Number of columns
         */
        public int getCols() {
            return cols;
        }

        /**
         * Sets the value at specified position.
         *
         * @param row Row index
         * @param col Column index
         * @param value Value to set
         */
        public void setValue(int row, int col, double value) {
            data[row][col] = value;
        }

        /**
         * Gets the value at specified position.
         *
         * @param row Row index
         * @param col Column index
         * @return Value at position
         */
        public double getValue(int row, int col) {
            return data[row][col];
        }

        /**
         * Transposes the matrix.
         *
         * @return Transposed matrix
         */
        public Matrix transpose() {
            double[][] transposed = new double[cols][rows];
            for(int i=0;i<rows;i++) {
                for(int j=0;j<cols;j++) {
                    transposed[j][i] = data[i][j];
                }
            }
            return new Matrix(transposed);
        }

        /**
         * Multiplies the current matrix with another matrix.
         *
         * @param other Matrix to multiply with
         * @return Resultant matrix
         */
        public Matrix multiply(Matrix other) {
            if(this.cols != other.rows) {
                throw new IllegalArgumentException("Incompatible matrix sizes for multiplication.");
            }

            double[][] result = new double[this.rows][other.cols];
            for(int i=0;i<this.rows;i++) {
                for(int j=0;j<other.cols;j++) {
                    for(int k=0;k<this.cols;k++) {
                        result[i][j] += this.data[i][k] * other.data[k][j];
                    }
                }
            }

            return new Matrix(result);
        }

        /**
         * Inverts the matrix using Gaussian elimination.
         *
         * @return Inverted matrix
         */
        public Matrix inverse() {
            if(this.rows != this.cols) {
                throw new IllegalArgumentException("Only square matrices can be inverted.");
            }

            int n = this.rows;
            double[][] augmented = new double[n][2*n];

            // Create augmented matrix [A | I]
            for(int i=0;i<n;i++) {
                for(int j=0;j<n;j++) {
                    augmented[i][j] = this.data[i][j];
                }
                for(int j=n;j<2*n;j++) {
                    augmented[i][j] = (i == (j - n)) ? 1.0 : 0.0;
                }
            }

            // Perform Gaussian elimination
            for(int i=0;i<n;i++) {
                // Find the pivot
                double pivot = augmented[i][i];
                if(pivot == 0) {
                    // Swap with a non-zero row
                    boolean swapped = false;
                    for(int j=i+1;j<n;j++) {
                        if(augmented[j][i] != 0) {
                            double[] temp = augmented[i];
                            augmented[i] = augmented[j];
                            augmented[j] = temp;
                            pivot = augmented[i][i];
                            swapped = true;
                            break;
                        }
                    }
                    if(!swapped) {
                        throw new ArithmeticException("Matrix is singular and cannot be inverted.");
                    }
                }

                // Normalize the pivot row
                for(int j=0;j<2*n;j++) {
                    augmented[i][j] /= pivot;
                }

                // Eliminate the current column in other rows
                for(int k=0;k<n;k++) {
                    if(k != i) {
                        double factor = augmented[k][i];
                        for(int j=0;j<2*n;j++) {
                            augmented[k][j] -= factor * augmented[i][j];
                        }
                    }
                }
            }

            // Extract the inverse matrix
            double[][] inverse = new double[n][n];
            for(int i=0;i<n;i++) {
                for(int j=0;j<n;j++) {
                    inverse[i][j] = augmented[i][j+n];
                }
            }

            return new Matrix(inverse);
        }

        /**
         * Prints the matrix to the console.
         */
        public void printMatrix() {
            for(int i=0;i<rows;i++) {
                System.out.println(Arrays.toString(data[i]));
            }
        }
    }

    public static void main(String[] args) {
        // Generate synthetic data
        int numSamples = 100;
        double[][] X = new double[numSamples][1];
        double[] y = new double[numSamples];
        java.util.Random rand = new java.util.Random(42);

        for(int i=0;i<numSamples;i++) {
            X[i][0] = 2 * rand.nextDouble(); // Feature X between 0 and 2
            y[i] = 4 + 3 * X[i][0] + rand.nextGaussian(); // y = 4 + 3X + noise
        }

        // Initialize and train the Linear Regression model
        LinearRegression lr = new LinearRegression();
        lr.fit(X, y);

        // Display the learned parameters
        double[] theta = lr.getTheta();
        System.out.printf("Parameters learned via Normal Equation:%n");
        System.out.printf("Intercept (theta_0): %.2f%n", theta[0]);
        System.out.printf("Coefficient (theta_1): %.2f%n", theta[1]);

        // Make predictions on the training data
        double[] y_pred = lr.predict(X);

        // Calculate Mean Squared Error
        double mse = lr.meanSquaredError(y, y_pred);
        System.out.printf("Mean Squared Error: %.2f%n", mse);

        // Optional: Predict for new values
        double[][] X_new = { {0}, {2} };
        double[] y_new_pred = lr.predict(X_new);
        System.out.printf("Predictions for new inputs:%n");
        for(int i=0;i<X_new.length;i++) {
            System.out.printf("Input: %.2f, Predicted y: %.2f%n", X_new[i][0], y_new_pred[i]);
        }

        // Note: Visualization is not straightforward in Java without external libraries.
        // However, you can print the predicted values and compare them with actual data.
    }
}
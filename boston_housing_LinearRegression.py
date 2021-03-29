import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def scale_data(training_set=None, test_set=None):
    # see: It resembles the style of keras where the preprocessing layer is now imported to the model.
    # https://stackoverflow.com/questions/40337744/scalenormalise-a-column-in-spark-dataframe-pyspark
    # https://spark.apache.org/docs/2.2.0/ml-pipeline.html
    from pyspark.ml.feature import VectorAssembler

    # columns names
    cols = training_set.columns
    label = cols[-1]
    #Input all the features in one vector column, scaling seems to work with row vector created by the assembler
    assembler = VectorAssembler(inputCols=cols[:-1], outputCol = 'features')
    # we select the columns 'features', label since the assembler concatenates the new merged col
    training_set = assembler.transform(training_set).select('features', label)
    test_set = assembler.transform(test_set).select('features', label)

    # also see: https://medium.com/@dhiraj.p.rai/essentials-of-feature-engineering-in-pyspark-part-i-76a57680a85
    scaler = StandardScaler(inputCol='features', outputCol='scaledFeatures',
                            withStd=True, withMean=False)

    scalerModel = scaler.fit(training_set)

    # Normalize each feature to have unit standard deviation.
    training_set = scalerModel.transform(training_set).select('scaledFeatures', label)
    test_set = scalerModel.transform(test_set).select('scaledFeatures', label) # apply the scaler of the training set
    return training_set, test_set

def main(spark_instance=None, training_set=None, test_set=None):
    # scale the data
    training_set, test_set = scale_data(training_set, test_set)
    training_set.show()
    # Linear Regression
    from pyspark.ml.regression import LinearRegression
    linear_regressor = LinearRegression(featuresCol='scaledFeatures', labelCol = 'median_val', maxIter=100, regParam=0.2, elasticNetParam=0.0, standardization=False)

    # Fit the model
    linear_model = linear_regressor.fit(training_set)

    # Print the Weights and Bias for linear regression
    print(f"Weights: {str(linear_model.coefficients)}")
    print(f"Bias: {str(linear_model.intercept)}")

    # Summarize the model over the training set and print out some metrics
    trainingSummary = linear_model.summary

    # predict on the test set (evaluate creates a new field)
    predictions = linear_model.evaluate(test_set)

    # show the predictions on the test set
    predictions.predictions.show()

    print(f"RMSE: {trainingSummary.rootMeanSquaredError}")
    print(f"r2: {trainingSummary.r2}")

    
if __name__ == "__main__":

    # import findSpark
    import findspark
    findspark.init()
    try:
        from pyspark.sql import SparkSession
        from pyspark.ml.feature import StandardScaler
    except ImportError as error:
        raise ImportError('Spark was not found!')
        exit()

    # initialize pyspark
    spark = SparkSession.builder.master("local[*]").appName("LinearRegression_BH").getOrCreate()
    spark.sparkContext.setLogLevel('OFF')                                                      # silent spark

    # load the data (already splitted by an other script)
    train_data = spark.read.csv('data/boston_train_data.csv',inferSchema=True, header =True)
    test_data  = spark.read.csv('data/boston_test_data.csv',inferSchema=True, header =True)
    
    train_data.printSchema()
    
    # main function
    main(spark, train_data, test_data)
    
    spark.stop()
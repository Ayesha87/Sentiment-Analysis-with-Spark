from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.ml.classification import LogisticRegression
from pyspark.sql import SparkSession
from pyspark.ml.feature import HashingTF, Tokenizer, StopWordsRemover
import pickle

#create Spark session
appName = "Sentiment Analysis in Spark"
spark = SparkSession \
    .builder \
    .appName(appName) \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()
tweets_csv = spark.read.csv('dataset/tweets.csv', inferSchema=True, header=True)
#tweets_csv.show(truncate=False, n=3)

#select only "SentimentText" and "Sentiment" column,
#and cast "Sentiment" column data into integer
data = tweets_csv.select("SentimentText", col("Sentiment").cast("Int").alias("label"))
data.printSchema()
#data.show(truncate=False, n=5)

#divide data, 70% for training, 30% for testing
dividedData = data.randomSplit([0.7, 0.3])
trainingData = dividedData[0] #index 0 = data training
testingData = dividedData[1] #index 1 = data testing
train_rows = trainingData.count()
test_rows = testingData.count()
#print("Training data rows:", train_rows, "; Testing data rows:", test_rows)


tokenizer = Tokenizer(inputCol="SentimentText", outputCol="SentimentWords")
tokenizedTrain = tokenizer.transform(trainingData)
#tokenizedTrain.show(truncate=False, n=5)

swr = StopWordsRemover(inputCol=tokenizer.getOutputCol(),
                       outputCol="MeaningfulWords")
SwRemovedTrain = swr.transform(tokenizedTrain)
SwRemovedTrain.show(truncate=False, n=5)

hashTF = HashingTF(inputCol=swr.getOutputCol(), outputCol="features")
numericTrainData = hashTF.transform(SwRemovedTrain).select(
    'label', 'MeaningfulWords', 'features')
#numericTrainData.show(truncate=False, n=3)

lr = LogisticRegression(labelCol="label", featuresCol="features",
                        maxIter=10, regParam=0.01)
model = lr.fit(numericTrainData)
print("Training is done!")

LogisticRegression.save(model, "lrm_model.model")




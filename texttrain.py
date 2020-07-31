from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegressionModel

from pyspark.ml.feature import HashingTF, Tokenizer, StopWordsRemover
import pickle
import matplotlib.pyplot as plt

class Traintext:
    def __init__(self):
        appName = "Sentiment Analysis in Spark"
        self.spark = SparkSession \
            .builder \
            .appName(appName) \
            .config("spark.some.config.option", "some-value") \
            .getOrCreate()

    def train(self):
        model = LogisticRegressionModel.load("lrm_model.model")
        tw_csv = self.spark.read.csv('tweet.csv', inferSchema=True, header=True)
        #tw_csv.show(truncate=False, n=3)
        self.tokenizer = Tokenizer(inputCol="SentimentText", outputCol="SentimentWords")
        self.swr = StopWordsRemover(inputCol=self.tokenizer.getOutputCol(),
                    outputCol="MeaningfulWords")
        self.hashTF = HashingTF(inputCol=self.swr.getOutputCol(), outputCol="features")
        self.tokenized_tw = self.tokenizer.transform(tw_csv)
        self.SwRemoved_tw = self.swr.transform(self.tokenized_tw)
        self.numeric_tw = self.hashTF.transform(self.SwRemoved_tw).select(
             'MeaningfulWords', 'features')
        #numeric_tw.show(n=3, truncate=False)
        #umeric_tw.show(truncate=False, n=2)
        self.prediction_tw = model.transform(self.numeric_tw)
        self.predictionFinal_tw = self.prediction_tw.select("MeaningfulWords", "prediction")
        self.predictionFinal_tw.show(truncate = False, n=20)

    def textplot(self,topic):
        neg = self.predictionFinal_tw.filter(self.predictionFinal_tw.prediction == '0.0').count()
        pos = self.predictionFinal_tw.filter(self.predictionFinal_tw.prediction == '1.0').count()
        total = self.predictionFinal_tw.count()
        negative = (neg / total) * 100
        positive = (pos / total) * 100
        negative = format(negative, ".2f")
        positive = format(positive, '.2f')
        labels = ['Positive[' + str(positive) + '%]', 'Negative[' + str(negative) + '%]']
        sizes = [positive, negative]
        colors = ['#009ACD', '#ADD8E6']
        patches, texts = plt.pie(sizes, colors=colors, startangle=90)
        plt.legend(patches, labels, loc='best')
        plt.title('Sentiments on  topic: '+topic)
        plt.axis('equal')
        plt.tight_layout()

        plt.savefig('Static/assets/img/text.png')
        plt.show()

'''if __name__=="__main__":
    t=Traintext()
    t.train()'''
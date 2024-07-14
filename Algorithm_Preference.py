from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS

# Initialize Spark session for real-time processing
spark = SparkSession.builder.appName('ContentRecommendation').getOrCreate()

# Load data
data = spark.read.csv('user_data.csv', header=True, inferSchema=True)

# Train ALS model
als = ALS(userCol='user_id', itemCol='item_id', ratingCol='rating', coldStartStrategy='drop')
model = als.fit(data)

# Make predictions
predictions = model.transform(data)
predictions.show()

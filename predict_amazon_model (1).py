from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Initialize Spark session
spark = SparkSession.builder \
    .appName("AmazonSentimentPrediction") \
    .getOrCreate()

# Paths
test_csv_path = "hdfs://user/mahad/processed/processed_test/preprocessed_complete_test"  # folder or CSV
saved_model_path = "saved_model_amazon"      # folder where model is saved
output_csv_path = "amazon_predictions.csv"   # output predictions

# Load test data
test_df = spark.read.csv(test_csv_path, header=True, inferSchema=True)

# Load saved model
model = PipelineModel.load(saved_model_path)

# Run predictions
predictions = model.transform(test_df)

# Show some predictions
predictions.select("text", "prediction").show(10, truncate=100)

# Evaluate model
evaluator = MulticlassClassificationEvaluator(
    labelCol="polarity",
    predictionCol="prediction",
    metricName="accuracy"
)
accuracy = evaluator.evaluate(predictions)
print(f"Test Accuracy: {accuracy:.4f}")

# Save predictions to CSV
predictions.select("text", "polarity", "prediction") \
    .coalesce(1) \
    .write.csv(output_csv_path, header=True, mode="overwrite")

spark.stop()

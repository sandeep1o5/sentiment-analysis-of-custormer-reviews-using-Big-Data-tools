from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, HashingTF, IDF, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.sql.functions import concat_ws, col
import datetime
import os

# =========================================================
# 1. Initialize Spark Session
# =========================================================

spark = SparkSession.builder \
    .appName("Amazon Polarity Classification") \
    .config("spark.eventLog.enabled", "true") \
    .config("spark.eventLog.dir", "file:///C:/Users/mahad/spark-events") \
    .getOrCreate()


# =========================================================
# 2. Load Data from HDFS
# =========================================================
train_path = "hdfs://localhost:9000/user/mahad/processed/processed_train/processed_complete_train.csv"
test_path = "hdfs://localhost:9000/user/mahad/processed/processed_test/processed_complete_test.csv"

train_df = spark.read.option("header", "true").option("inferSchema", "true").csv(train_path)
test_df = spark.read.option("header", "true").option("inferSchema", "true").csv(test_path)

# =========================================================
# 3. Data Cleaning
# =========================================================
train_df = train_df.na.drop(subset=["polarity", "title", "text"])
test_df = test_df.na.drop(subset=["polarity", "title", "text"])

print("Data loaded successfully.")
train_df.printSchema()
train_df.show(3, truncate=False)

if train_df.count() == 0 or test_df.count() == 0:
    raise ValueError("One of the datasets is empty after cleaning.")

# =========================================================
# 4. Combine Title + Text → combined_text
# =========================================================
train_df = train_df.withColumn("combined_text", concat_ws(" ", col("title"), col("text")))
test_df = test_df.withColumn("combined_text", concat_ws(" ", col("title"), col("text")))

# =========================================================
# 5. Text Feature Engineering (TF-IDF)
# =========================================================
tokenizer = Tokenizer(inputCol="combined_text", outputCol="words")
hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=20000)
idf = IDF(inputCol="rawFeatures", outputCol="features")

# =========================================================
# 6. Label Indexing
# =========================================================
label_indexer = StringIndexer(inputCol="polarity", outputCol="label")

# =========================================================
# 7. Classifier
# =========================================================
lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=10)

# =========================================================
# 8. Full Pipeline
# =========================================================
pipeline = Pipeline(stages=[tokenizer, hashingTF, idf, label_indexer, lr])

# =========================================================
# 9. Train Model
# =========================================================
print("Training model...")
model = pipeline.fit(train_df)
print("Model training completed successfully!")

# =========================================================
# 10. Evaluate Model
# =========================================================
predictions = model.transform(test_df)
predictions.select("label", "prediction", "probability").show(10, truncate=False)

# Multiple metrics
acc_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
prec_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedPrecision")
rec_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedRecall")
f1_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
auc_eval = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="probability")

metrics = {
    "accuracy": acc_eval.evaluate(predictions),
    "precision": prec_eval.evaluate(predictions),
    "recall": rec_eval.evaluate(predictions),
    "f1_score": f1_eval.evaluate(predictions),
    "auc": auc_eval.evaluate(predictions)
}

print("\nModel Evaluation Metrics:")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")

# =========================================================
# 11. Save Metrics Locally
# =========================================================
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
local_metrics_path = f"./output/metrics_{timestamp}.json"

# Ensure output directory exists
os.makedirs("./output", exist_ok=True)

metrics_df = spark.createDataFrame(
    [(metrics["accuracy"], metrics["precision"], metrics["recall"], metrics["f1_score"], metrics["auc"], timestamp)],
    ["accuracy", "precision", "recall", "f1_score", "auc", "timestamp"]
)

metrics_df.coalesce(1).write.mode("overwrite").json(local_metrics_path)

print(f"Metrics saved locally to {local_metrics_path}")

# =========================================================
# 12. Save Model Locally
# =========================================================
local_model_path = "./output/logreg_amazon_tfidf_model"
model.write().overwrite().save(local_model_path)
print(f"Model saved locally to {local_model_path}")

# =========================================================
# 13. Stop Spark
# =========================================================
spark.stop()
print("Spark session stopped cleanly.")
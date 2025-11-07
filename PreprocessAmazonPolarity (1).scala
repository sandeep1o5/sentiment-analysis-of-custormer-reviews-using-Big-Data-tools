import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

object PreprocessCSV {
  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder()
      .appName("Preprocess CSV")
      .master("local[*]")
      .getOrCreate()

    // HDFS paths
    val trainPath = "hdfs://localhost:9000/user/mahad/input/train.csv"
    val testPath = "hdfs://localhost:9000/user/mahad/input/test.csv"
    val processedTrainPath = "hdfs://localhost:9000/user/mahad/processed/processed_train"
    val processedTestPath = "hdfs://localhost:9000/user/mahad/processed/processed_test"

    // Read CSVs without header
    val trainDF = spark.read.option("header", "false").option("inferSchema", "true").csv(trainPath)
      .toDF("polarity", "title", "text")
    val testDF = spark.read.option("header", "false").option("inferSchema", "true").csv(testPath)
      .toDF("polarity", "title", "text")

    // Fill missing values
    val trainFilled = trainDF.na.fill(Map(
      "polarity" -> 1,
      "title" -> "unknown",
      "text" -> "unknown"
    ))
    val testFilled = testDF.na.fill(Map(
      "polarity" -> 1,
      "title" -> "unknown",
      "text" -> "unknown"
    ))

    // No need to StringIndex text columns, only the label if required
    // Save processed CSVs
    trainFilled.write.mode("overwrite").option("header", "true").csv(processedTrainPath)
    testFilled.write.mode("overwrite").option("header", "true").csv(processedTestPath)

    println("✅ Preprocessing completed. Files saved to HDFS.")
    spark.stop()
  }
}

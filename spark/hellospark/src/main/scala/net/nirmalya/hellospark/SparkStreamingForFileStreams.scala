package net.nirmalya.hellospark

import org.apache.spark._
import org.apache.spark.streaming._
import org.apache.spark.streaming.StreamingContext._

/**
 * An example using Spark Streaming to read stream of raw data from files on any
 * file system compatible with the HDFS API (HDFS, S3, NFS, etc.).
 * <p>
 * On its own, this example does not do much, but it forms the basis of
 * implementing near real-time machine learning use cases.
 *
 * @author Nirmalya Ghosh
 */
object ScalaSparkStreamingForFileStreams {
  def main(args: Array[String]) {
    val dataDirectory = args(0)
    val batchDurationSeconds = args(1).toInt
    val conf = new SparkConf().setAppName("SparkStreamingForFileStreams")
    val ssc = new StreamingContext(conf, Seconds(batchDurationSeconds * 1000))
    val lines = ssc.textFileStream(dataDirectory);
    val words = lines.flatMap(_.split(" "))
    val pairs = words.map(word => (word, 1))
    val wordCounts = pairs.reduceByKey(_ + _)
    wordCounts.print()
    ssc.start()
    ssc.awaitTermination()
  }
}
package net.nirmalya.hellospark.streaming

import org.apache.spark._
import org.apache.spark.storage._
import org.apache.spark.streaming._
import org.apache.spark.streaming.StreamingContext._
import org.apache.spark.streaming.kafka._

/**
 * An example using Spark Streaming to read Apache access logs from a Kafka
 * topic and compute relevant metrics.
 *
 * @author Nirmalya Ghosh
 */
object SparkStreamingPageViewMetrics {

  def main(args: Array[String]): Unit = {
    run(args)
  }

  def run(args: Array[String]): Unit = {
    //TODO add check for the number of arguments passed
    val Array(batchDurationSeconds, zkQuorum, groupId, topic, numThreads) = args
    val conf = new SparkConf().setAppName("SparkStreamingPageViewMetrics")
    conf.setMaster("local[2]")
    val batchDuration = Seconds(batchDurationSeconds.toInt)
    val ssc = new StreamingContext(conf, batchDuration)
    val topics = Map(topic -> 1)
    val accessLogUtil = new ApacheAccessLogUtils()

    val stream = KafkaUtils.createStream(ssc, zkQuorum, groupId, topics).map(_._2)

    // Parse the Apache access logs
    val rdd1 = stream.map { line => accessLogUtil.parseAccessLogRecord(line) }

    // Filter out images, JSON, JavaScript, CSS
    val rdd2 = rdd1
      .filter(x => !x.requestedResource.endsWith(".jpg"))
      .filter(x => !x.requestedResource.endsWith(".json"))
      .filter(x => !x.requestedResource.endsWith(".js"))
      .filter(x => !x.requestedResource.endsWith(".css"))

    val rdd3 = rdd2.map(x => (x.requestedResource, 1))
      .reduceByKey((x, y) => x + y)

    rdd3.print()

    //TODO compute other metrics

    ssc.start()
    ssc.awaitTermination()
  }

}
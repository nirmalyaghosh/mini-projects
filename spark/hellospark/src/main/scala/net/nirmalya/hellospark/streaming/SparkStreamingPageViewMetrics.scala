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
    conf.setMaster("local[1]")
    val batchDuration = Seconds(batchDurationSeconds.toInt * 1000)
    val ssc = new StreamingContext(conf, batchDuration)
    val kafkaParams = Map(
      "zookeeper.connect" -> zkQuorum,
      "zookeeper.connection.timeout.ms" -> "10000",
      "group.id" -> groupId)
    val topics = Map(topic -> 1)

    val accessLogUtil = new ApacheAccessLogUtils()

    val rdd1 = KafkaUtils
      .createStream(ssc, zkQuorum, groupId, topics, StorageLevel.MEMORY_AND_DISK )
      .map(_._2).cache()

    // Parse the Apache access logs
    val rdd2 = rdd1.map { line => accessLogUtil.parseAccessLogRecord(line) }

    // Filter out images, JSON, JavaScript, CSS
    val rdd3 = rdd2
      .filter(x => !x.requestedResource.endsWith(".jpg"))
      .filter(x => !x.requestedResource.endsWith(".json"))
      .filter(x => !x.requestedResource.endsWith(".js"))
      .filter(x => !x.requestedResource.endsWith(".css"))

    val rdd4 = rdd3.map(x => (x.requestedResource, 1))
      .reduceByKey((x, y) => x + y)

    rdd4.print()

    //TODO compute other metrics

    ssc.start()
    ssc.awaitTermination()
  }

}
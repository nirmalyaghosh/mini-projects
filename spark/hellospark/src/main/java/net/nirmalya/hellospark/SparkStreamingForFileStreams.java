package net.nirmalya.hellospark;

import java.util.Arrays;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.streaming.Duration;
import org.apache.spark.streaming.api.java.JavaDStream;
import org.apache.spark.streaming.api.java.JavaPairDStream;
import org.apache.spark.streaming.api.java.JavaStreamingContext;

import scala.Tuple2;

/**
 * An example using Spark Streaming to read stream of raw data from files on any
 * file system compatible with the HDFS API (HDFS, S3, NFS, etc.).
 * <p>
 * On its own, this example does not do much, but it forms the basis of
 * implementing near real-time machine learning use cases.
 * 
 * @author Nirmalya Ghosh
 */
public class SparkStreamingForFileStreams {

  @SuppressWarnings("serial")
  public static void main(String[] args) {
    String dataDirectory = args[0];
    int batchDurationSeconds;
    try {
      batchDurationSeconds = Integer.parseInt(args[1]);
    } catch (Exception e) {
      batchDurationSeconds = 10;
    }

    SparkConf sc = new SparkConf().setAppName("SparkStreamingForFileStreams");
    JavaStreamingContext ssc =
        new JavaStreamingContext(sc, new Duration(batchDurationSeconds * 1000));
    JavaDStream<String> lines = ssc.textFileStream(dataDirectory);
    JavaDStream<String> words = lines.flatMap(new FlatMapFunction<String, String>() {
      @Override
      public Iterable<String> call(String x) {
        return Arrays.asList(x.split(" "));
      }
    });

    JavaPairDStream<String, Integer> pairs =
        words.mapToPair(new PairFunction<String, String, Integer>() {
          @Override
          public Tuple2<String, Integer> call(String s) throws Exception {
            return new Tuple2<String, Integer>(s, 1);
          }
        });

    JavaPairDStream<String, Integer> wordCounts =
        pairs.reduceByKey(new Function2<Integer, Integer, Integer>() {
          @Override
          public Integer call(Integer i1, Integer i2) throws Exception {
            return i1 + i2;
          }
        });

    wordCounts.print();
    ssc.start();
    ssc.awaitTermination();
  }
}

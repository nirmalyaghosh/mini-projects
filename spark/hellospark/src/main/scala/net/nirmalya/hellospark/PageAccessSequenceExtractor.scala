package net.nirmalya.hellospark

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf

/**
 * Reads web server log files to extract individual browsing sessions. Current
 * implementation relies on the presence of cookies in every resource access
 * event logged in the web server log files.
 *
 * Sample line (containing a cookie) :
 * 202.156.9.227 - - [07/Feb/2014:17:59:07 +0000] "GET /category/page-1.html HTTP/1.1" 200 4661 "http://50.56.188.172/" "Mozilla/5.0 (Linux; Android 4.3; GT-N7105 Build/JSS15J) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/31.0.1650.59 Mobile Safari/537.36" "testservercookie=2622857bceebc571b945fe0a49ec6a4e"
 * 
 * All fields in each line are separated by a single white space
 * @author Nirmalya Ghosh
 */
object PageAccessSequenceExtractor {

  def main(args: Array[String]) {
    val inputFile = args(0)
    val outputFile = args(1)
    val conf = new SparkConf().setAppName("PageAccessSequenceExtractor")
    val sc = new SparkContext(conf)
    val webLog = sc.textFile(inputFile)
    val pairs = webLog.map { s =>
      val fields = s.split(" ")
      val key = fields(0)
      if (fields.length > 1) {
        val pageId = fields(1)
        (key, pageId)
      } else
        (key, None)
    }
    val result = pairs.groupByKey()

    // If we want to save user with page sequence mapping,
    result.saveAsTextFile(outputFile)

    // If we only want the sequence of pages
    // result.map(x => x._2 ).saveAsTextFile(outputFile)
  }

}
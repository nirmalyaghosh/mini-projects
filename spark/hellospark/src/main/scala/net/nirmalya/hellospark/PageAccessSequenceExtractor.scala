package net.nirmalya.hellospark

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import java.util.regex.Pattern

/**
 * Reads web server log files to extract individual browsing sessions.
 *
 * Current implementation used the uniqueness of the combination of IP address
 * and user agent string in resource access events logged in the web server log
 * files.
 *
 * It will eventually rely on the presence of cookies in a resource access event.
 *
 * Sample line (containing a cookie) :
 * 202.156.9.227 - - [07/Feb/2014:17:59:07 +0000] "GET /category/page-1.html HTTP/1.1" 200 4661 "http://50.56.188.172/" "Mozilla/5.0 (Linux; Android 4.3; GT-N7105 Build/JSS15J) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/31.0.1650.59 Mobile Safari/537.36" "testservercookie=2622857bceebc571b945fe0a49ec6a4e"
 *
 * All fields in each line are separated by a single white space
 *
 * Credit for parsing server log record goes to Alvin Alexander. It was based on,
 * https://github.com/alvinj/ScalaApacheAccessLogParser/blob/master/src/main/scala/AccessLogParser.scala
 * 
 * To run it, 
 * $ sbt clean compile package
 * $ ~/spark/bin/spark-submit --class net.nirmalya.hellospark.PageAccessSequenceExtractor ./target/scala-2.10/hellospark_2.10-0.0.1.jar src/test/resources/sample-access-log.txt ./joboutput
 * 
 * @author Nirmalya Ghosh
 */
object PageAccessSequenceExtractor {

  // Regex for a record in the Apache access log format
  private val ddd = "\\d{1,3}" // at least 1 but not more than 3 times (possessive)
  private val ip = s"($ddd\\.$ddd\\.$ddd\\.$ddd)?" // like `202.156.9.227`
  private val client = "(\\S+)" // '\S' is 'non-whitespace character'
  private val user = "(\\S+)"
  private val dateTime = "(\\[.+?\\])" // like `[07/Feb/2014:17:59:07 +0000]`
  private val request = "\"(.*?)\"" // any number of any character, reluctant
  private val status = "(\\d{3})"
  private val bytes = "(\\S+)" // this can be a "-"
  private val referer = "\"(.*?)\""
  private val agent = "\"(.*?)\""
  private val regex = s"$ip $client $user $dateTime $request $status $bytes $referer $agent"
  private val p = Pattern.compile(regex)

  // Regex for extracting the 'Request-URI' from a 'Request-Line' which looks like
  // (Method SP Request-URI SP HTTP-Version)
  private val regex2 = s".*?((?:\\/[\\w\\.\\-]+)+)"
  private val p2 = Pattern.compile(regex2)

  def extractRequestURI(s: String): String = {
    val matcher2 = p2.matcher(s)
    if (matcher2.find) {
      return matcher2.group(1)
    } else {
      return ""
    }
  }

  def main(args: Array[String]) {
    val inputFile = args(0)
    val outputFile = args(1)
    val conf = new SparkConf().setAppName("PageAccessSequenceExtractor")
    val sc = new SparkContext(conf)
    val webLog = sc.textFile(inputFile)
    val pairs = webLog.map { s =>
      val matcher = p.matcher(s)
      if (matcher.find) {
        val ip = matcher.group(1)
        matcher.group(2) // $client
        matcher.group(3) // $user
        matcher.group(4) // $dateTime
        val request = extractRequestURI(matcher.group(5)) // $request
        matcher.group(6) // $status
        matcher.group(7) // $bytes
        matcher.group(8) // $referer
        val agent = matcher.group(9) // $agent
        val key = ip + "_" + agent
        (key, request)
      } else {
        (None, None)
      }

    }
    val result = pairs.groupByKey()

    // If we want to save user with page sequence mapping,
    result.saveAsTextFile(outputFile)

    // If we only want the sequence of pages
    // result.map(x => x._2 ).saveAsTextFile(outputFile)
  }

}
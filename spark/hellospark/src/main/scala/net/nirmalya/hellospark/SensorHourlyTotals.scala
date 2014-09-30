package net.nirmalya.hellospark

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext.rddToPairRDDFunctions
import java.util.Calendar
import java.text.SimpleDateFormat

/**
 * Reads sensor readings of the format <tt>2014-09-28T00:02:25,431000776,341</tt>
 * and computes the hourly totals.
 */
object SensorHourlyTotals {

  val format = new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss")
  val format2 = new SimpleDateFormat("yyyy-MM-dd")

  case class SimpleCSVHeader(header: Array[String]) extends Serializable {
    val index = header.zipWithIndex.toMap
    def apply(array: Array[String], key: String): String = array(index(key))
  }

  def hourOfDay(cal: Calendar, ts: String): Int = {
    try {
      cal.setTime(format.parse(ts))
      return cal.get(Calendar.HOUR_OF_DAY)
    } catch {
      case e: Exception => { return -1 }
    }
  }

  def makeKey(cal: Calendar, ts: String, id: String): (String, Int, String) = {
    try {
      cal.setTime(format.parse(ts))
      return (format2.format(cal.getTime()), cal.get(Calendar.HOUR_OF_DAY), id)
    } catch {
      case e: Exception => { return ("", -1, id) }
    }
  }

  def main(args: Array[String]) {
    val inputFile = args(0)
    val outputFile = args(1)

    val cal = Calendar.getInstance();

    val conf = new SparkConf().setAppName("SensorHourlyTotals")
    val sc = new SparkContext(conf)
    val csv = sc.textFile(inputFile, 2).cache()
    val data = csv.map(line => line.split(",").map(elem => elem.trim))
    val header = new SimpleCSVHeader(data.take(1)(0)) // first line is a header
    val sensorReadings = data.filter(line => header(line, "SensorID") != "SensorID")
    val sensorIDs = sensorReadings.map(row => header(row, "SensorID"))
    val sensorIDsByReadings = sensorReadings
      .map(row => makeKey(cal, header(row, "Timestamp"), header(row, "SensorID"))
        -> header(row, "Reading").toInt)
      .reduceByKey((a, b) => a + b)

    sensorIDsByReadings.saveAsTextFile(outputFile)
  }
}


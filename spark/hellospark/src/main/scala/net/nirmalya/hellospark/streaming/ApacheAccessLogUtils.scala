package net.nirmalya.hellospark.streaming

import java.util.regex.Pattern

/**
 * Simple utility to parse Apache access logs.
 *
 * It is based on Alvin Alexander's Scala Apache access log parser library.
 *
 * @author Nirmalya Ghosh
 */
class ApacheAccessLogUtils extends java.io.Serializable {

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

  private def extractRequestURI(requestLine: String): String = {
    val matcher2 = p2.matcher(requestLine)
    if (matcher2.find) {
      return matcher2.group(1)
    } else {
      return ""
    }
  }

  def parseAccessLogRecord(accessLogLine: String): (AccessLogRecord) = {
    var zz = new Array[String](5)
    val matcher = p.matcher(accessLogLine)
    if (matcher.find) {
      zz(0) = matcher.group(1) // $ip
      zz(1) = matcher.group(2) // $client
      matcher.group(3) // $user
      zz(2) = matcher.group(4) // $dateTime
      zz(3) = extractRequestURI(matcher.group(5)) // $request
      matcher.group(6) // $status
      matcher.group(7) // $bytes
      zz(4) = matcher.group(8) // $referer
    }

    AccessLogRecord(zz(0), zz(1), zz(2), zz(3), zz(4))
  }

}
package net.nirmalya.hellospark.streaming

case class AccessLogRecord(
    clientIpAddress: String,
    dateTime: String,
    requestedResource: String,
    referer: String, // page that linked to this URL
    userAgent: String // browser identification string
    )
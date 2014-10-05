//import AssemblyKeys._

//assemblySettings

name := "hellospark"

version := "0.0.1"

scalaVersion := "2.10.4"

val sparkVersion = "1.0.0"

libraryDependencies <<= scalaVersion {
  scala_version => Seq(
    // Spark and Spark Streaming
    "org.apache.spark" %% "spark-core" % sparkVersion,
    "org.apache.spark" %% "spark-streaming" % sparkVersion,
    "org.apache.spark" %% "spark-streaming-kafka" % sparkVersion,
    // Kafka
    "org.apache.kafka" %% "kafka" % "0.8.1.1",
    // Others
    "net.sf.opencsv" % "opencsv" % "2.3"
  )
}

packSettings

packMain := Map(
  //"generator" -> "com.chimpler.sparkstreaminglogaggregation.RandomLogGenerator",
  "streamingPageViewMetrics" -> "net.nirmalya.hellospark.streaming.SparkStreamingPageViewMetrics"
)

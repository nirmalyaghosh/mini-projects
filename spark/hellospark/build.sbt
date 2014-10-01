name := "hellospark"

version := "0.0.1"

scalaVersion := "2.10.4"

// additional libraries
libraryDependencies ++= Seq(
"org.apache.spark" %% "spark-core" % "1.1.0" % "provided",
"org.apache.spark" % "spark-streaming_2.10" % "1.1.0",
"net.sf.opencsv" % "opencsv" % "2.3",
"org.scalaz" % "scalaz-core_2.10" % "7.1.0"
)


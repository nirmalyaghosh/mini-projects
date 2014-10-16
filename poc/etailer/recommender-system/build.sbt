name := "recommender-system"

version := "0.1"

scalaVersion := "2.10.4"

libraryDependencies ++= Seq(
  "mysql" % "mysql-connector-java" % "5.1.33",
  "org.apache.mahout" % "mahout-core" % "0.9"
)

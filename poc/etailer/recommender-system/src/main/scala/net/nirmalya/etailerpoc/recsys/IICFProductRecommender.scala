package net.nirmalya.etailerpoc.recsys

import java.io.{ File, FileInputStream }
import java.sql.{ Connection, DriverManager, SQLException }
import java.util.Properties

import scala.collection.JavaConverters.asScalaBufferConverter
import scala.collection.mutable.Buffer

import org.apache.mahout.cf.taste.impl.model.file.FileDataModel
import org.apache.mahout.cf.taste.impl.recommender.GenericBooleanPrefItemBasedRecommender
import org.apache.mahout.cf.taste.impl.similarity.TanimotoCoefficientSimilarity
import org.apache.mahout.cf.taste.recommender.RecommendedItem

/**
 * Item-based recommender.
 *
 * @author Nirmalya Ghosh
 */
object IICFProductRecommender {

  def main(args: Array[String]) {
    val cfg: Properties = readConfigurations(args(0))
    val modelFile = cfg.getProperty("recommendations.model.file");
    val howMany = cfg.getProperty("recommendations.per.unit", "4").toInt;
    val model = new FileDataModel(new File(modelFile))
    val itemSimilarity = new TanimotoCoefficientSimilarity(model)
    val recommender = new GenericBooleanPrefItemBasedRecommender(model, itemSimilarity)
    val allItemIds = model.getUserIDs() // here we have item - item
    val recommendations: Buffer[(Long, RecommendedItem)] = Buffer()
    val batchSize = cfg.getProperty("recommendations.storage.batch","50").toInt
    while (allItemIds.hasNext()) {
      val itemId = allItemIds.nextLong()
      try {
        val recommendedItems = recommender.recommend(itemId, howMany).asScala
        recommendedItems.foreach(item => recommendations += ((itemId, item)))
        if (recommendations.length >= batchSize) {
          saveRecommendations(recommendations, cfg)
          recommendations.clear
        }
      } catch {
        case e: Exception => println("Caught an exception", e.getMessage())
      }
    }

    if (!recommendations.isEmpty) saveRecommendations(recommendations, cfg)
  }

  def readConfigurations(configFilePath: String): Properties = {
    val p = new Properties()
    try {
      val in = new FileInputStream(configFilePath)
      p.load(in)
      in.close();
    } catch {
      case e: Exception => System.exit(1)
    }
    return p
  }

  def saveRecommendations(
    recommendations: Buffer[(Long, RecommendedItem)],
    cfg: Properties) {
    val storageType = cfg.getProperty("recommendations.storage", "mysql");
    if (storageType.trim().equalsIgnoreCase("mysql")) {
      saveRecommendationsIntoMySQL(recommendations, cfg)
    }
  }

  def saveRecommendationsIntoMySQL(
    recommendations: Buffer[(Long, RecommendedItem)],
    cfg: Properties) {
    Class.forName("com.mysql.jdbc.Driver")
    val conn: Connection = null
    try {
      val h = cfg.getProperty("mysql.dbhost")
      val d = cfg.getProperty("mysql.dbname")
      val u = cfg.getProperty("mysql.username")
      val p = cfg.getProperty("mysql.password")
      val connStr = "jdbc:mysql://%s/%s?user=%s&password=%s" format (h, d, u, p)

      val conn = DriverManager.getConnection(connStr)
      val pstmt = conn.prepareStatement(
        "INSERT INTO sku_recommendations " +
          "(base_sku_id, recommended_sku_id, relevance) VALUES (?,?,?)" +
          "ON DUPLICATE KEY UPDATE relevance = VALUES(relevance)")

      recommendations.foreach(r => {
        pstmt.setLong(1, r._1)
        pstmt.setLong(2, r._2.getItemID())
        pstmt.setDouble(3, r._2.getValue())
        val x = pstmt.executeUpdate
      })

      // Fix : Due to some bug an item is getting itself recommended as well
      val stmt = conn.createStatement()
      stmt.execute("DELETE FROM sku_recommendations WHERE base_sku_id=recommended_sku_id")
    } catch {
      case se: SQLException => println("Caught SQLException", se.printStackTrace())
      case _: Throwable => println("Got some other kind of exception")
    } finally {
      try { if (conn != null) conn.close() }
      catch {
        case _: Throwable => {}
      }
    }
  }

}
package net.nirmalya.etailerpoc.recsys

import java.io.File
import java.util.List
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel
import org.apache.mahout.cf.taste.impl.recommender.GenericItemBasedRecommender
import org.apache.mahout.cf.taste.impl.similarity.PearsonCorrelationSimilarity
import org.apache.mahout.cf.taste.recommender.RecommendedItem
import scala.collection.JavaConverters._

class IICFProductRecommender {

  def main(args: Array[String]) {
    val modelFile = args(0)
    val itemId = args(1).toLong
    val howMany = args(2).toInt
    val model = new FileDataModel(new File(modelFile))
    val itemSimilarity = new PearsonCorrelationSimilarity(model)
    val recommender = new GenericItemBasedRecommender(model, itemSimilarity)
    val similarItems = recommender.mostSimilarItems(itemId, howMany).asScala
    similarItems.foreach(r => println(r.getItemID() + ": " + r.getValue()))
  }

}
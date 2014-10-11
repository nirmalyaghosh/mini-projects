package net.nirmalya.etailerpoc.recsys

import java.io.File

import scala.collection.JavaConverters._

import org.apache.mahout.cf.taste.impl.model.file.FileDataModel
import org.apache.mahout.cf.taste.impl.recommender.GenericBooleanPrefItemBasedRecommender
import org.apache.mahout.cf.taste.impl.similarity.TanimotoCoefficientSimilarity

/**
 * Item-based recommender.
 * 
 * @author Nirmalya Ghosh
 */
object IICFProductRecommender {

  def main(args: Array[String]) {
    val modelFile = args(0)
    val itemId = args(1).toLong
    val howMany = args(2).toInt
    val model = new FileDataModel(new File(modelFile))
    val itemSimilarity = new TanimotoCoefficientSimilarity(model)
    val recommender = new GenericBooleanPrefItemBasedRecommender(model, itemSimilarity)
    val recommendedItems = recommender.recommend(itemId, howMany).asScala
    recommendedItems.foreach(r => println(r.getItemID() + ":" + r.getValue()))
  }

}
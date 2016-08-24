/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.flink.ml.evaluation

import org.apache.flink.api.common.typeinfo.TypeInformation
import org.apache.flink.api.scala._
import org.apache.flink.ml._
import java.io.Serializable

import scala.reflect.ClassTag
import scala.math.log

/**
 * Evaluation score
 *
 * Can be used to calculate a performance score for an algorithm, when provided with a DataSet
 * of (truth, prediction) tuples
 *
 * @tparam PredictionType output type
 */
trait Score[PredictionType] {
  def evaluate(trueAndPredicted: DataSet[(PredictionType, PredictionType)]): DataSet[Double]
}

/** Traits to allow us to determine at runtime if a Score is a loss (lower is better) or a
  * performance score (higher is better)
  */
trait Loss

trait PerformanceScore

/**
 * Metrics expressible as a mean of a function taking output pairs as input
 *
 * @param scoringFct function to apply to all elements
 * @tparam PredictionType output type
 */
abstract class MeanScore[PredictionType: TypeInformation: ClassTag](
    scoringFct: (PredictionType, PredictionType) => Option[Double])
    (implicit yyt: TypeInformation[(PredictionType, PredictionType)])
  extends Score[PredictionType] with Serializable {

  def evaluate(trueAndPredicted: DataSet[(PredictionType, PredictionType)]): DataSet[Double] = {
    trueAndPredicted.map(yy => scoringFct(yy._1, yy._2)).filter(_.nonEmpty).map(_.get).mean()
  }
}

/** Scores aimed at evaluating the performance of regression algorithms
  *
  */
object RegressionScores {
  /**
   * Mean Squared loss function
   *
   * Calculates (y1 - y2)^2^ and returns the mean.
   *
   * @return a Loss object
   */
  def squaredLoss = new MeanScore[Double]((y1,y2) => Some((y1 - y2) * (y1 - y2))) with Loss

  /**
   * Mean Zero One Loss Function also usable for score information
   *
   * Assigns 1 if sign of outputs differ and 0 if the signs are equal, and returns the mean
   *
   * @return a Loss object
   */
  def zeroOneSignumLoss = new MeanScore[Double]({ (y1, y2) =>
    val sy1 = y1.signum
    val sy2 = y2.signum
    if (sy1 == sy2) Some(0) else Some(1)
  }) with Loss

  /** Calculates the coefficient of determination, $R^2^$
    *
    * $R^2^$ indicates how well the data fit the a calculated model
    * Reference: [[http://en.wikipedia.org/wiki/Coefficient_of_determination]]
    */
  def r2Score = new Score[Double] with PerformanceScore {
    override def evaluate(trueAndPredicted: DataSet[(Double, Double)]): DataSet[Double] = {
      val onlyTrue = trueAndPredicted.map(truthPrediction => truthPrediction._1)
      val meanTruth = onlyTrue.mean()

      val ssRes = trueAndPredicted
        .map(tp => (tp._1 - tp._2) * (tp._1 - tp._2)).reduce(_ + _)
      val ssTot = onlyTrue
        .mapWithBcVariable(meanTruth) {
          case (truth: Double, meanTruth: Double) => (truth - meanTruth) * (truth - meanTruth)
        }.reduce(_ + _)

      val r2 = ssRes
        .mapWithBcVariable(ssTot) {
          case (ssRes: Double, ssTot: Double) =>
          // We avoid dividing by 0 and just assign 0.0
          if (ssTot == 0.0) {
            0.0
          }
          else {
            1 - (ssRes / ssTot)
          }
      }
      r2
    }
  }
}

/** Scores aimed at evaluating the performance of classification algorithms
  *
  */
object ClassificationScores {
  /** Calculates the fraction of correct predictions
    *
    */
  def accuracyScore = {
    new MeanScore[Int]((y1, y2) => if (y1 == y2) Some(1) else Some(0))
      with PerformanceScore
  }

  def errorRateScore = {
    new MeanScore[Int]((y1, y2) => if (y1 == y2) Some(0) else Some(1))
      with PerformanceScore //TODO
  }

  def precisionScore = {
    new MeanScore[Boolean]((y1, y2) =>
      if (y1 && y2) Some(1) else if (y2 && !y1) Some(0) else None)
      with PerformanceScore //TODO
  }

  def recallScore = {
    new MeanScore[Boolean]((y1, y2) =>
      if (y1 && y2) Some(1) else if (y1 && !y2) Some(0) else None)
      with PerformanceScore //TODO
  }

  def trueNegativeRateScore = {
    new MeanScore[Boolean]((y1, y2) =>
      if (!y1 && !y2) Some(1) else if (!y1 && y2) Some(0) else None)
      with PerformanceScore //TODO
  }

  def falsePositiveRateScore = {
    new MeanScore[Boolean]((y1, y2) =>
      if (!y1 && y2) Some(1) else if (!y1 && !y2) Some(0) else None)
      with PerformanceScore //TODO
  }

  def falseDiscoveryRateScore = {
    new MeanScore[Boolean]((y1, y2) =>
      if (!y1 && y2) Some(1) else if (y1 && y2) Some(0) else None)
      with PerformanceScore //TODO
  }

  def negativePredictiveValueScore = {
    new MeanScore[Boolean]((y1, y2) =>
      if (!y1 && !y2) Some(1) else if (y1 && !y2) Some(0) else None)
      with PerformanceScore //TODO
  }

  def fMeasureScore = {
    new Score[Boolean] with PerformanceScore { //TODO
      override def evaluate(trueAndPredicted: DataSet[(Boolean, Boolean)]): DataSet[Double]  ={
        val precision = precisionScore.evaluate(trueAndPredicted)
        val recall = recallScore.evaluate(trueAndPredicted)
        val p = precision.collect().head
        val r = recall.collect().head
        trueAndPredicted.getExecutionEnvironment.fromElements(2 / (1/p + 1/r))
      }
    }
  }

  def AUCScore = {
    new Score[Double] with PerformanceScore { //TODO
      override def evaluate(trueAndPredicted: DataSet[(Double, Double)]): DataSet[Double]  = {
        val pos = 1.0 / trueAndPredicted.filter(_._1 == 1.0).count()
        val neg = 1.0 / trueAndPredicted.filter(_._1 == 0.0).count()
        val grouped = trueAndPredicted.collect().groupBy(_._2)
          .map(g => (g._1, (g._2.count(_._1 == 0.0), g._2.count(_._1 == 1.0))))
        val sorted = grouped.toSeq.sortBy(_._1)
        val (height, area) = sorted.foldLeft((0.0, 0.0)){ (acc, elem) =>
          (acc._1 + elem._2._1 * neg, acc._2 + acc._1 * elem._2._2 * pos)}
        trueAndPredicted.getExecutionEnvironment.fromElements(area)
      }
    }
  }

  def logLossScore = {
    new Score[Double] with PerformanceScore { //TODO
      override def evaluate(trueAndPredicted: DataSet[(Double, Double)]): DataSet[Double]  = {
        val n = trueAndPredicted.count()
        val l = trueAndPredicted.map(actAndPred =>
          actAndPred._1 * log(actAndPred._2) + (1.0 - actAndPred._1) * log(1.0 - actAndPred._2))
        l.map(x => (1, x)).sum(1).map(ll => ll._2 * (- 1.0 / n))
      }
    }
  }

  /**
   * Mean Zero One Loss Function
   *
   * Assigns 1 if outputs differ and 0 if they are equal, and returns the mean.
   *
   * @return a Loss object
   */
  def zeroOneLoss = {
    new MeanScore[Double]((y1, y2) =>
      if (y1.approximatelyEquals(y2)) Some(0) else Some(1)) with Loss
  }
}

object RecommendationScores extends java.io.Serializable {

  private def precisionAtK(k: Int, actual: Array[Int], predicted: Array[Int]): Double = {
    //val pred = predicted.take(k)
    val a = actual.length
    //actual.intersect(pred).length / a : Double
    a
  }

  def averagePrecisionAtK(k: Int) = {
    new Score[Array[Int]] with PerformanceScore {
      //TODO
      override def evaluate(data: DataSet[(Array[Int], Array[Int])]): DataSet[Double] = {
        val perUser = data.map { trueAndPred => precisionAtK(k, trueAndPred._1, trueAndPred._2)
          //val pred = trueAndPred._2.take(k)
          //val a = trueAndPred._1.length
          //trueAndPred._1.intersect(pred).length / a : Double}
          //a
        }
          //val perUser = data.map(trueAndPred => 2)
        perUser.mean()
      }
    }
  }

  def meanAveragePrecisionAtK(k: Int) = {
    new Score[Array[Int]] with PerformanceScore {
      //TODO
      override def evaluate(data: DataSet[(Array[Int], Array[Int])]): DataSet[Double] = {
        val perUser = data.map{ trueAndPred =>
          val precisions = 1 to k map (_k => precisionAtK(_k, trueAndPred._1, trueAndPred._2))
          precisions.sum / k
        }
        perUser.mean()
      }
    }
  }

  def averageRecallAtK(k: Int) = {
    new Score[Array[Int]] with PerformanceScore {
      //TODO
      override def evaluate(data: DataSet[(Array[Int], Array[Int])]): DataSet[Double] = {
        val perUser = data.map { trueAndPred =>
          val act = trueAndPred._1.toSet
          val pred = trueAndPred._2.take(k).toSet
          act.intersect(pred).size / k : Double
        }
        perUser.mean()
      }
    }
  }

  private def iDCG(k: Int) = {

  }
}

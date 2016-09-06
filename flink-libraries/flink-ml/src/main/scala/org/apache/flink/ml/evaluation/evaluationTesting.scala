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

import org.apache.flink.api.common.operators.Order
import org.apache.flink.api.scala._
import org.apache.flink.ml.recommendation._

/**
  * Created by mani on 12/07/16.
  */
object evaluationTesting {

  def main(args: Array[String]) {

    val env = ExecutionEnvironment.getExecutionEnvironment

    val inputDS: DataSet[(Int, Int, Double)] = env.readCsvFile[(Int, Int, Double)](
          "/Users/mani/Downloads/batch/nmusic_recoded_max_test.csv",
          lineDelimiter = "\n",
          fieldDelimiter = "|",
          includedFields = Array(0, 1, 2))

    val als = ALS()
      .setIterations(10)
      .setNumFactors(10)

    als.fit(inputDS)

    val users = inputDS.map(x => x._1).distinct().collect()
    val items = inputDS.map(x => x._2).distinct().collect()

    val dataToPredict = for {u <- users; i <- items} yield (u, i)

    val dataSet2Predict: DataSet[(Int, Int)] = env.fromCollection(dataToPredict)

    val predictions = als
        .predict(dataSet2Predict)
        .groupBy(x => x._1)
        .sortGroup(x => x._3, Order.DESCENDING)
        .reduceGroup(x => x.toArray)
        .map(x => (x.head._1, x.map(y => y._2)))

    val actuallyviewed = inputDS
      .filter(x => x._3 > 0.9)
      .groupBy(x => x._1)
      .reduceGroup(x => x.toArray)
      .map(x => (x.head._1, x.map(y => y._2)))

    val dataToTest = actuallyviewed.join(predictions)

//    //val resultsInNumbers: DataSet[(Double, Double)] =
//    // env.readCsvFile[(Double, Double)]("/Users/mani/Downloads/eredm_stathoz.csv",
//    val resultsInNumbers: DataSet[(Int, Int)] =
//      env.readCsvFile[(Int, Int)]("/Users/mani/flink_measurement/classes.tsv",
//      lineDelimiter = "\n",
//      fieldDelimiter = "\t",
//      includedFields = Array(0, 1))
//    val resultsInNumbers7: DataSet[(Double, Double)] =
//      env.readCsvFile[(Double, Double)]("/Users/mani/flink_measurement/probabilities.tsv",
//      lineDelimiter = "\n",
//      fieldDelimiter = "\t",
//      includedFields = Array(0, 1))
//
//    val smallTest: DataSet[(Double, Double)] =
//      env.fromElements((0.0, 0.1), (0.0, 0.2), (1.0, 0.9), (1.0, 0.8))
//
//    def getBool(x: Int) = {
//      !(x == 0)
//    }
//
//
//    val resultsInBools = resultsInNumbers.map(x => (getBool(x._1), getBool(x._2)))
//
//    val accuracyScore = ClassificationScores.accuracyScore
//    val precisionScore = ClassificationScores.precisionScore
//    val recallScore = ClassificationScores.recallScore
//    val trueNegativeRateScore = ClassificationScores.trueNegativeRateScore
//    val falsePositiveRateScore = ClassificationScores.falsePositiveRateScore
//    val negativePredictiveValueScore = ClassificationScores.negativePredictiveValueScore
//    val fMeasureScore = ClassificationScores.fMeasureScore
//    val aucScore = ClassificationScores.AUCScore
//    val loglossScore = ClassificationScores.logLossScore
//
//
//    val _acc = accuracyScore.evaluate(resultsInNumbers)
//    val _pre = precisionScore.evaluate(resultsInBools)
//    val _rec = recallScore.evaluate(resultsInBools)
//    val _tru = trueNegativeRateScore.evaluate(resultsInBools)
//    val _fal = falsePositiveRateScore.evaluate(resultsInBools)
//    val _neg = negativePredictiveValueScore.evaluate(resultsInBools)
//    val _fMe = fMeasureScore.evaluate(resultsInBools)
//    val _auc = aucScore.evaluate(resultsInNumbers7 )
//    val _log = loglossScore.evaluate(resultsInNumbers7)
//
//    val acc = _acc.collect()
//    val pre = _pre.collect()
//    val rec = _rec.collect()
//    val tru = _tru.collect()
//    val fal = _fal.collect()
//    val neg = _neg.collect()
//    val fMe = _fMe.collect()
//    val auc = _auc.collect()
//    val log = _log.collect()
//
//    println("accuracyScore " + acc)
//    println("precisionScore " + pre)
//    println("recallScore " + rec)
//    println("trueNegativeRateScore " + tru)
//    println("falsePositiveRateScore " + fal)
//    println("negativePredictiveValueScore " + neg)
//    println("fMeasureScore " + fMe)
//    println("AUCScore " + auc)
//    println("logLossScore " + log)

  }
}

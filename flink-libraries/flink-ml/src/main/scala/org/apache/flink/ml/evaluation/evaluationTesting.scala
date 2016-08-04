package org.apache.flink.ml.evaluation

import org.apache.flink.api.java.LocalEnvironment
import org.apache.flink.api.scala._


/**
  * Created by mani on 12/07/16.
  */
object evaluationTesting {

  def main(args: Array[String]) {

    val env = ExecutionEnvironment.getExecutionEnvironment
    val env1 = ExecutionEnvironment.getExecutionEnvironment
    val env2 = ExecutionEnvironment.getExecutionEnvironment
    val env3 = ExecutionEnvironment.getExecutionEnvironment
    val env4 = ExecutionEnvironment.getExecutionEnvironment
    val env5 = ExecutionEnvironment.getExecutionEnvironment
    val env6 = ExecutionEnvironment.getExecutionEnvironment
    val env7 = ExecutionEnvironment.getExecutionEnvironment

    //val resultsInNumbers: DataSet[(Double, Double)] = env.readCsvFile[(Double, Double)]("/Users/mani/Downloads/eredm_stathoz.csv",
    val resultsInNumbers: DataSet[(Int, Int)] = env.readCsvFile[(Int, Int)]("/Users/mani/flink_measurement/classes.tsv",
      lineDelimiter = "\n",
      fieldDelimiter = "\t",
      includedFields = Array(0, 1))
    val resultsInNumbers1: DataSet[(Int, Int)] = env1.readCsvFile[(Int, Int)]("/Users/mani/flink_measurement/classes.tsv",
      lineDelimiter = "\n",
      fieldDelimiter = "\t",
      includedFields = Array(0, 1))
    val resultsInNumbers2: DataSet[(Int, Int)] = env2.readCsvFile[(Int, Int)]("/Users/mani/flink_measurement/classes.tsv",
      lineDelimiter = "\n",
      fieldDelimiter = "\t",
      includedFields = Array(0, 1))
    val resultsInNumbers3: DataSet[(Int, Int)] = env3.readCsvFile[(Int, Int)]("/Users/mani/flink_measurement/classes.tsv",
      lineDelimiter = "\n",
      fieldDelimiter = "\t",
      includedFields = Array(0, 1))
    val resultsInNumbers4: DataSet[(Int, Int)] = env4.readCsvFile[(Int, Int)]("/Users/mani/flink_measurement/classes.tsv",
      lineDelimiter = "\n",
      fieldDelimiter = "\t",
      includedFields = Array(0, 1))
    val resultsInNumbers5: DataSet[(Int, Int)] = env5.readCsvFile[(Int, Int)]("/Users/mani/flink_measurement/classes.tsv",
      lineDelimiter = "\n",
      fieldDelimiter = "\t",
      includedFields = Array(0, 1))
    val resultsInNumbers6: DataSet[(Int, Int)] = env6.readCsvFile[(Int, Int)]("/Users/mani/flink_measurement/classes.tsv",
      lineDelimiter = "\n",
      fieldDelimiter = "\t",
      includedFields = Array(0, 1))
    val resultsInNumbers7: DataSet[(Double, Double)] = env7.readCsvFile[(Double, Double)]("/Users/mani/flink_measurement/probabilities.tsv",
      lineDelimiter = "\n",
      fieldDelimiter = "\t",
      includedFields = Array(0, 1))

    def getBool(x: Int) = {
      !(x == 0)
    }


    val resultsInBools = resultsInNumbers.map(x => (getBool(x._1), getBool(x._2)))
    val resultsInBools1 = resultsInNumbers1.map(x => (getBool(x._1), getBool(x._2)))
    val resultsInBools2 = resultsInNumbers2.map(x => (getBool(x._1), getBool(x._2)))
    val resultsInBools3 = resultsInNumbers3.map(x => (getBool(x._1), getBool(x._2)))
    val resultsInBools4 = resultsInNumbers4.map(x => (getBool(x._1), getBool(x._2)))
    val resultsInBools5 = resultsInNumbers5.map(x => (getBool(x._1), getBool(x._2)))
    //val resultsInBools6 = resultsInNumbers6.map(x => (getBool(x._1), getBool(x._2)))

    val accuracyScore = ClassificationScores.accuracyScore
    val precisionScore = ClassificationScores.precisionScore
    val recallScore = ClassificationScores.recallScore
    val trueNegativeRateScore = ClassificationScores.trueNegativeRateScore
    val falsePositiveRateScore = ClassificationScores.falsePositiveRateScore
    val negativePredictiveValueScore = ClassificationScores.negativePredictiveValueScore
    //val fMeasureScore = ClassificationScores.fMeasureScore
    val loglossScore = ClassificationScores.logLossScore


    val _acc = accuracyScore.evaluate(resultsInNumbers)
    val _pre = precisionScore.evaluate(resultsInBools1)
    val _rec = recallScore.evaluate(resultsInBools2)
    val _tru = trueNegativeRateScore.evaluate(resultsInBools3)
    val _fal = falsePositiveRateScore.evaluate(resultsInBools4)
    val _neg = negativePredictiveValueScore.evaluate(resultsInBools5)
    //val _fMe = fMeasureScore.evaluate(resultsInBools6)
    val _log = loglossScore.evaluate(resultsInNumbers7)

    val acc = _acc.collect()
    val pre = _pre.collect()
    val rec = _rec.collect()
    val tru = _tru.collect()
    val fal = _fal.collect()
    val neg = _neg.collect()
    //val fMe = _fMe.collect()
    val log = _log.collect()

    println("accuracyScore " + acc)
    println("precisionScore " + pre)
    println("recallScore " + rec)
    println("trueNegativeRateScore " + tru)
    println("falsePositiveRateScore " + fal)
    println("negativePredictiveValueScore " + neg)
    //println("fMeasureScore " + fMe)
    println("logLossScore " + log)

  }
}

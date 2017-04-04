package com.needine.spark

import org.apache.spark._
import org.apache.spark.SparkContext._
import org.apache.spark.rdd.RDD
  
import org.apache.commons.math3.random.MersenneTwister
import org.apache.commons.math3.stat.regression.OLSMultipleLinearRegression
import org.apache.commons.math3.stat.correlation.Covariance
import org.apache.commons.math3.distribution.MultivariateNormalDistribution
import org.apache.commons.math3.stat.correlation.PearsonsCorrelation

import java.text.SimpleDateFormat
import java.io.File
import java.util.Locale

import scala.io.Source
import scala.collection.mutable.ArrayBuffer

import com.github.nscala_time.time.Imports._

import breeze.plot._
import org.apache.spark.mllib.stat.correlation.PearsonCorrelation

object Prog1 {
  
  def readInvestingHistory(file: File): Array[(DateTime, Double)] = {
    val format = new SimpleDateFormat("MMM d, yyyy",  Locale.ENGLISH)
    val lines = Source.fromFile(file).getLines().toSeq
    lines.map { line => 
      val cols = line.split("\t")
      //println("|"+cols(0)+"|")
      val date = new DateTime(format.parse(cols(0)))
      val value = cols(1).replace(",", "").toDouble
      //val value = cols(1).toDouble
      
      (date, value)
    }.reverse.toArray
  }
  
  def readYahoo(file: File): Array[(DateTime, Double)] ={
    val format = new SimpleDateFormat("yyyy-MM-dd",  Locale.ENGLISH)
    val lines = Source.fromFile(file).getLines().toSeq
    lines.tail.map { line => 
      val cols = line.split(",")
      val date = new DateTime(format.parse(cols(0)))
      val value = cols(1).toDouble
      (date, value)
    }.reverse.toArray
    
  }
  
  def trimToRegion(history: Array[(DateTime, Double)], start: DateTime, end: DateTime):Array[(DateTime, Double)] = {
    var trimmed = history.dropWhile(_._1< start).takeWhile(_._1<= end)
    if (trimmed.head._1 != start) {
      trimmed = Array((start , trimmed.head._2)) ++ trimmed
    }
    if (trimmed.last._1 != end){
      trimmed = trimmed ++  Array((end , trimmed.last._2)) 
    }
    trimmed
  }
  
  def fillInHistory(history: Array[(DateTime, Double)], start: DateTime, end: DateTime): Array[(DateTime, Double)] = {
    var cur = history
    val filled = new ArrayBuffer[((DateTime, Double))]()
    
    var curDate = start
    
    while (curDate < end) {
      if(cur.tail.nonEmpty && cur.tail.head._1 == curDate){
        cur = cur.tail
      }
      filled +=((curDate, cur.head._2))  
      curDate += 1.days
      
      // Skip weekends
      if(curDate.dayOfWeek().get > 5) curDate += 2.days
    }
    filled.toArray
  }
  
  def twoWeeksReturn(history : Array[(DateTime, Double)]): Array[Double] ={
    history.sliding(10).map{ window =>
      val next = window.last._2
      val prev = window.head._2
      (next-prev)/prev
    }.toArray
  }

  def factorMatrix(histories: Seq[Array[Double]]): Array[Array[Double]] = {
    val mat = new Array[Array[Double]](histories.head.length)
    for(i<-0 until histories.head.length ){
      mat(i) = histories.map(_(i)).toArray
    }
    mat
  }
  
  def featurize (factorReturns:Array[Double]):Array[Double]  = {
    val squareReturns = factorReturns.map { x => math.signum(x)*x*x}
    val squareRootReturns = factorReturns.map { x => math.signum(x)* math.sqrt(math.abs(x))}
    squareReturns++squareRootReturns++factorReturns
  }
  
  def linearModel(instrument: Array[Double], factorMatrix: Array[Array[Double]]):OLSMultipleLinearRegression = {
    val regression = new OLSMultipleLinearRegression()
    regression.newSampleData(instrument, factorMatrix)
    regression
  }
  
  def plotDistribution(samples: Array[Double]) {
    val min = samples.min
    val max = samples.max
    val domain = Range.Double(min, max, (max-min)/100).toList.toArray
    val densities = KernelDensity.estimate(samples, domain)
    
    val f = Figure()
    val p = f.subplot(0)
    p += plot(domain, densities)
    p.xlabel = "Two week return ($)"
    p.ylabel = "Density"
  }
 
  def instrumentTrialReturn(instrument: Array[Double], trial: Array[Double]):Double = {
    var instrumentTrialReturn = instrument(0)
    var i = 0
    while(i < trial.length){
      instrumentTrialReturn += trial(i) * instrument(i+1)
      i += 1
    }
    instrumentTrialReturn  
  }
  
  def trialReturn(trial : Array[Double], instruments:Seq[Array[Double]]):Double = {
    var totalReturn = 0.0
    for(instr <- instruments){
      totalReturn += instrumentTrialReturn(instr, trial)
    }
    totalReturn/instruments.size
  }
  
  def trialReturns(seed: Long, numTrials: Int, instruments: Seq[Array[Double]], factorMeans: Array[Double], factorCovariances: Array[Array[Double]]): Seq[Double] = {
    val rand = new MersenneTwister(seed)
    val multivariateNormal = new MultivariateNormalDistribution(rand, factorMeans, factorCovariances)
    
    val trialReturns = new Array[Double](numTrials)
    
    for( i<- 0 until numTrials){
      val trialFactorReturns = multivariateNormal.sample
      val trialFeatures = featurize(trialFactorReturns)
      trialReturns(i) = trialReturn(trialFeatures, instruments)
    }
    trialReturns
  }
  
  def fivePercentVaR(trials: RDD[Double]): Double = {
    val topLosses = trials.takeOrdered(math.max(trials.count().toInt/20, 1))
    topLosses.last
    
  }
  
  def plotDistribution(samples: RDD[Double]){
    val stats = samples.stats()
    val min = stats.min
    val max = stats.max
    val domain = Range.Double(min, max, (max-min)/100).toList.toArray
    val densities = KernelDensity.estimate(samples, domain)
    val f = Figure()
    val p = f.subplot(0)
    p += plot(domain, densities)
    p.xlabel = "Two Weeks Return ($)"
    p.ylabel = "Density"
  }
  
  
  // MAIN ---------------------------------  
  def main(args: Array[String]) = {
    
    val appName = "Monte Carlo 1.6"
    val conf    = new SparkConf()

    conf.setAppName(appName).setMaster("local[*]").setExecutorEnv("driver-memory", "12G")
    val sc = new SparkContext(conf)
    
    // Primera tarea
    
    val start = new DateTime(2009, 10, 27, 0, 0)
    val end = new DateTime(2016, 10, 27, 0, 0)
    
    val factorsPrefix = "C:/Users/juani/Documents/ml/MonteCarlo/factors/"
    
    val factors1: Seq[Array[(DateTime, Double)]] = Array("CrudeOil.data", "TreasuryBonds.data").map { f => new File(factorsPrefix+f) }.map(readInvestingHistory(_))     
    val factors2: Seq[Array[(DateTime, Double)]] = Array("GSPC.csv", "IXIC.csv").map { f => new File(factorsPrefix+f) }.map(readInvestingHistory(_))     
    
    val files = new File("C:/Users/juani/Documents/ml/MonteCarlo/stocks/").listFiles()
    val rawStocks:Seq[Array[(DateTime, Double)]] = files.flatMap { file =>  
      try{
        Some(readYahoo(file))
      }catch{
        case e : Exception => None
      }
    }.filter(_.size >= 260*5+10)
    
    val stocks = rawStocks.map(trimToRegion(_, start, end)).map(fillInHistory(_, start, end))  
    val factors = (factors1 ++ factors2).map(trimToRegion(_, start, end)).map(fillInHistory(_, start, end))
    
    println((stocks ++ factors).forall(_.size == stocks(0).size))
    
    val stocksReturns = stocks.map(twoWeeksReturn(_))
    val factorsReturns = factors.map(twoWeeksReturn(_))
    
    val factorMat = factorMatrix(factorsReturns)
    val factorFeatures = factorMat.map { featurize(_) }
    val models = stocksReturns.map(linearModel(_, factorFeatures))
    val factorWeights = models.map(_.estimateRegressionParameters()).toArray
       
    // Segunda tarea
    plotDistribution(factorsReturns(0))
    plotDistribution(factorsReturns(1))
    
    //val factorCor = new PearsonCorrelation(factorMat).getCorrelationMatrix().getData()
    //println()
    
    val factorCov = new  Covariance(factorMat).getCovarianceMatrix.getData
    val factorMeans = factorsReturns.map(factor => factor.sum/factor.size).toArray
    val factorsDist = new MultivariateNormalDistribution(factorMeans, factorCov)
    
    
    // Tercera tarea
    val parallelism = 48
    val baseSeed = 1496
    val seeds = (baseSeed until baseSeed + parallelism)
    val seedRDD = sc.parallelize(seeds, parallelism)
    
    
    val numTrials = 10000000
    val bFactorWeights = sc.broadcast(factorWeights)
    
    val trials = seedRDD.flatMap (trialReturns(_, numTrials/parallelism, bFactorWeights.value, factorMeans, factorCov))
    val valueAtRisk = fivePercentVaR(trials)
    
    println(valueAtRisk)
    plotDistribution(trials)
    
    println("AAA")
    
  }
  
}
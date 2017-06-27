package com.needine.spark


import org.apache.spark._
import org.apache.spark.SparkContext._
import org.apache.spark.rdd.RDD


object Tmp {
  
  def main(args: Array[String]) ={
    val appName = "Monte Carlo 1.6"
    val conf    = new SparkConf()

    conf.setAppName(appName).setMaster("local[*]").setExecutorEnv("driver-memory", "12G")
    val sc = new SparkContext(conf)
    
    val rawRDD = sc.parallelize(Array(0,1,2,3,4,5,6,7,8,9))
    
    rawRDD.collect().foreach { println }

    println("SAMPLE")

    val sampleRDD = rawRDD.sample(true, 1.0)    
    sampleRDD.collect().foreach { println }

    println("Top Losses")
    
    //var topLossesList =  new Array[Int](100)
    val topLossesList = (0 until 100).map{ i =>
      val sampleRDD2 = rawRDD.sample(true, 1.0)    
      val topLosses = sampleRDD2.takeOrdered(rawRDD.count().toInt/5)  
      topLosses 
    }
    
    println(topLossesList)
    
    println("AAA")
    
  }
}
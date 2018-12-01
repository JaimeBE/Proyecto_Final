 //LIMPIEZA
 import org.apache.spark.ml.feature.{VectorAssembler, VectorSizeHint}
 val Dataset = spark.read.option("header","true").option("inferSchema","true").format("csv").load("bank-full.csv")
  Dataset.show()

  val Dataset = spark.read.option("header","true").option("inferSchema","true").option("delimiter",";").format("csv").load("bank-full.csv")
  Dataset.show()

    val cam1 = Data.withColumn("y",when(col("y").equalTo("yes"),1).otherwise(col("y")))
  val cam2 = cam1.withColumn("y",when(col("y").equalTo("no"),2).otherwise(col("y")))
 

  val cam3 = cam2.withColumn("y",'y.cast("Int"))
cam3.show(1)

val assemFeatures = (new VectorAssembler().setInputCols(Array("balance","day","duration","pdays","previous")).setOutputCol("features"))
  val Ldata = assemFeatures.transform(cam3)
  Ldata.show(1)

  val cambio = Ldata.withColumnRenamed("y", "label")
  val feat = cambio.select("label","features")

  feat.show()

//LIMPIEZA
///SVM
import org.apache.spark.ml.classification.LinearSVC

val c1 = feat.withColumn("label",when(col("label").equalTo("1"),0).otherwise(col("label")))
val c2 = c1.withColumn("label",when(col("label").equalTo("2"),1).otherwise(col("label")))
val c3 = c2.withColumn("label",'label.cast("Int"))
val linsvc = new LinearSVC().setMaxIter(10).setRegParam(0.1)
// Fit the model
val linsvcModel = linsvc.fit(c3)
// Imprime los coeficientes y la linea de svm
println(s"Coefficients: ${linsvcModel.coefficients} Intercept: ${linsvcModel.intercept}")
///SVM

///MULTILAYER PARCEPTRON

//Multilayer
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator



val splits = feat.randomSplit(Array(0.6, 0.4), seed = 1234L)
val train = splits(0)
val test = splits(1)
// specify layers for the neural network:
// input layer of size 4 (features), two intermediate of size 5 and 4
// and output of size 3 (classes)
val layers = Array[Int](5, 2, 2, 4)
// create the trainer and set its parameters
val trainer = new MultilayerPerceptronClassifier().setLayers(layers).setBlockSize(128).setSeed(1234L).setMaxIter(100)
// train the model
val model = trainer.fit(train)
// compute accuracy on the test set
val result = model.transform(test)
val predictionAndLabels = result.select("prediction", "label")
val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")
println(s"Test set accuracy = ${evaluator.evaluate(predictionAndLabels)}")


/// MULTILAYER PERCEPTRON

///regrecion logistica
import org.apache.spark.ml.classification.LogisticRegression
val lr = new LogisticRegression().setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8)
// Fit the model
val lrModel = lr.fit(feat)
// Print the coefficients and intercept for logistic regression
println(s"Coefficients: ${lrModel.coefficientMatrix} Intercept: ${lrModel.interceptVector}")
// We can also use the multinomial family for binary classification
val mlr = new LogisticRegression().setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8).setFamily("multinomial")
val mlrModel = mlr.fit(feat)
// Print the coefficients and intercepts for logistic regression with multinomial family
println(s"Multinomial coefficients: ${mlrModel.coefficientMatrix}")
println(s"Multinomial intercepts: ${mlrModel.interceptVector}")

///regrecion logistica


///ARBOL
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.feature.IndexToString 
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator


val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(feat)

val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4) // features with > 4 distinct values are treated as continuous.  .fit(data)

val Array(trainingData, testData) = feat.randomSplit(Array(0.7, 0.3))

val dt = new DecisionTreeClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures")

val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)

val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, dt, labelConverter))

val model = pipeline.fit(trainingData)

val predictions = model.transform(testData)

predictions.select("predictedLabel", "label", "features").show(5)

val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")
val accuracy = evaluator.evaluate(predictions)
println(s"Test Error = ${(1.0 - accuracy)}")

val treeModel = model.stages(2).asInstanceOf[DecisionTreeClassificationModel]
println(s"Learned classification tree model:\n ${treeModel.toDebugString}")


///ARBOL




package project1

import scalation.*
import scalation.mathstat._
import scalation.modeling.SimpleRegression

/** Project 1: run SimpleRegression (ScalaTion) on the top-2 predictor features
 *  (chosen by |correlation| with the target, per the Python/statsmodels analysis)
 *  for each of the three UCI datasets.
 */
object Project1EDA:

    private val DATA_DIR = "../project 1/data/"
    private val RESULTS_DIR = "../project 1/results/"
    
    /** Load a cleaned, numeric CSV (with header) and run SimpleRegression of
     *  target ~ feature for each feature in topFeatures.  Writes (x, y, yp)
     *  for each feature to a CSV under results/<datasetDir>/ so a plot of
     *  ScalaTion's actual-vs-predicted values can be rendered.
     *  @param fileName     csv file name (relative to DATA_DIR)
     *  @param datasetDir   results subdirectory name (matches the Python results folder)
     *  @param datasetName  label used in banners/output
     *  @param target       name of the target/response column
     *  @param topFeatures  the two predictor column names to regress on individually
     */
    def runDataset (fileName: String, datasetDir: String, datasetName: String, target: String,
                     topFeatures: Array [String]): Unit =
        val (mat, hdr) = MatrixD.loadH (DATA_DIR + fileName, fullPath = true)
        val colIdx     = hdr.zipWithIndex.toMap

        val y = mat (?, colIdx (target))

        banner (s"Project 1 - $datasetName: SimpleRegression on top-2 features (ScalaTion)")
        println (s"dataset shape = ${mat.dim} x ${mat.dim2}, target = $target")

        for feat <- topFeatures do
            val x   = mat (?, colIdx (feat))
            val mod = SimpleRegression (x, y, Array ("one", feat))
            mod.train ()
            val (yp, qof) = mod.test ()
            println (mod.report (qof))
            println (mod.summary ())

            val out = MatrixD (x, y, yp).ᵀ
            out.write (s"$RESULTS_DIR$datasetDir/scalation_pred_$feat.csv",
                       Array (feat, target, "yp"), fullPath = true)
    end runDataset

end Project1EDA


/** > runMain project1.project1_autoMPG */
@main def project1_autoMPG (): Unit =
    Project1EDA.runDataset ("auto_mpg.csv", "auto_mpg", "Auto MPG", "mpg",
                            Array ("weight", "displacement"))

/** > runMain project1.project1_concrete */
@main def project1_concrete (): Unit =
    Project1EDA.runDataset ("concrete.csv", "concrete", "Concrete Compressive Strength",
                            "concrete_compressive_strength", Array ("cement", "superplasticizer"))

/** > runMain project1.project1_airfoil */
@main def project1_airfoil (): Unit =
    Project1EDA.runDataset ("airfoil.csv", "airfoil", "Airfoil Self-Noise",
                            "scaled_sound_pressure_level",
                            Array ("frequency", "suction_side_displacement_thickness"))

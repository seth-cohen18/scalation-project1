
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Sun Jun 30 13:27:00 EDT 2024
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model: Auto-Regressive on lagged y and xe (ARX) using OLS
 *
 *  @see `scalation.modeling.RidgeRegression`
 *  @see `scalation.modeling.forecasting.ARY` when no exogenous variable are needed
 */

package scalation
package modeling
package forecasting

import scala.collection.mutable.{ArrayBuffer => VEC, LinkedHashSet => LSET}
import scala.runtime.ScalaRunTime.stringOf

import scalation.mathstat._

import MakeMatrix4TS._
import TransformT._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `ARX` class provides basic time series analysis capabilities for ARX models.
 *  ARX models build on `ARY` by including one or more exogenous (xe) variables.
 *  Given time series data stored in vector y, its next value y_t = combination of
 *  last p values of y and the last q values of each exogenous variable xe_j.
 *
 *      y_t = b dot x_t + e_t
 *
 *  where y_t is the value of y at time t and e_t is the residual/error term.
 *  @param x        the data/input matrix (lagged columns of y and xe) @see `ARX.apply`
 *  @param y        the response/output vector (time series data) 
 *  @param hh       the maximum forecasting horizon (h = 1 to hh)
 *  @param n_exo    the number of exogenous variables
 *  @param fname    the feature/variable names
 *  @param tRng     the time range, if relevant (time index may suffice)
 *  @param hparam   the hyper-parameters (defaults to `MakeMatrix4TS.hp`)
 *  @param bakcast  whether a backcasted value is prepended to the time series (defaults to false)
 *  @param tForms   the map of transformations applied
 */
class ARX (x: MatrixD, y: VectorD, hh: Int, n_exo: Int, fname: Array [String],
           tRng: Range = null, hparam: HyperParameter = hp,
           bakcast: Boolean = false,
           tForms: TransformMap = Map ("tForm_y" -> null))
      extends Forecaster_Reg (x, y, hh, fname, tRng, hparam, bakcast):

    private   val debug = debugf ("ARX", false)                         // debug function
    protected val p     = hparam("p").toInt                             // use the last p endogenous values (p lags)
    protected val q     = hparam("q").toInt                             // use the last q exogenous values (q lags)
    protected val spec  = hparam("spec").toInt                          // trend terms: 0 - implicit constant, 1 - linear, 2 - quadratic
                                                                        //              3 - sine, 4 cosine
    _modelName = s"ARX_${p}_${q}_$n_exo"
    yForm      = tForms("tForm_y").asInstanceOf [Transform]

    debug ("init", s"$modelName with $n_exo exogenous variables and additional trend spec = $spec, x.dims = ${x.dims}")
//  debug ("init", s"[ x | y ] = ${x :^+ y}")

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Forge a new vector from the first spec values of x, the last p-h+1 values
     *  of x (past values), values 1 to h-1 from the forecasts, and available values
     *  from exogenous variables.
     *  @param xx  the t-th row of the input matrix (lagged actual values)
     *  @param yy  the t-th row of the forecast matrix (forecasted future values)
     *  @param h   the forecasting horizon, number of steps ahead to produce forecasts
     */
    def forge (xx: VectorD, yy: VectorD, h: Int): VectorD =
        // add terms for the endogenous variable
        val n_endo  = spec + p                                       // number of trend + endogenous values
        val x_act   = xx(n_endo-(p+1-h) until n_endo)                // get actual lagged y-values (endogenous)
        val nyy     = p - x_act.dim                                  // number of forecasted values needed
        val x_fcast = yy(h-nyy until h)                              // get forecasted y-values

        var xy = x_act ++ x_fcast
        if n_exo > 0 and q > 0 then
            for j <- 0 until n_exo do                                // for the j-th exogenous variable
                xy = xy ++ hide (xx(n_endo + j*q until n_endo + (j+1)*q), h)
        xx(0 until spec) ++ xy
    end forge

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Hide values at the end of vector z (last h-1 values) as the increasing horizon  
     *  turns them in future values (hence unavailable).  Set these values to either
     *  zero (the default) or the last available value.
     *  @param z     the vector to shift
     *  @param h     the current horizon (number of steps ahead to forecast)
     *  @param fill  whether to backfill with the rightmost value (true) or with 0 (false)
     */
    def hide (z: VectorD, h: Int, fill: Boolean = true): VectorD =
        val zl = z(z.dim - 1)                                        // last available z value per horizon
        val z_ = new VectorD (z.dim)
        for k <- z.indices do
            z_(k) = if k <= z.dim - h then z(k+h-1) else if fill then zl else 0.0
        z_
    end hide

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Build a `ARX` model using the cols with the selected features.
     *  @param cols  the cols of the input matrix with selected features
     */
    def convertReg2Forc (cols: LSET [Int] = mcols): ARX =
        new ARX (getX(?, cols), getY, hh, n_exo, cols.toArray.map (fname (_)), tRng, hparam, bakcast, tForms)
    end convertReg2Forc

end ARX


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `ARX` companion object provides factory methods for the `ARX` class.
 */
object ARX extends MakeMatrix4TS:

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create an `ARX` object by building an input matrix xy and then calling the
     *  `ARX` constructor.
     *  @param xe       the matrix of exogenous variable values
     *  @param y        the endogenous/response vector (main time series data)
     *  @param hh       the maximum forecasting horizon (h = 1 to hh)
     *  @param fname_   the feature/variable names
     *  @param tRng     the time range, if relevant (time index may suffice)
     *  @param hparam   the hyper-parameters
     *  @param fEndo    the set of transforms to be used for the endogenous
     *  @param fExo     the array containing the sets of transforms to be used for the exogenous
     *  @param bakcast  whether a backcasted value is prepended to the time series (defaults to false)
     */
    def apply (xe: MatrixD, y: VectorD, hh: Int, fname_ : Array [String] = null,
               tRng: Range = null, hparam: HyperParameter = hp,
               fEndo: LSET [Transform] = null,
               fExo: Array [LSET [Transform]] = Array (),
               bakcast: Boolean = false): ARX =

        var xe_bfill: MatrixD = null
        if xe.dim2 > 0 and hparam("q").toInt > 0 then
            xe_bfill = new MatrixD (xe.dim, xe.dim2)
            for j <- xe.indices2 do xe_bfill(?, j) = backfill (xe(?, j))    // backfill each exogenous variable

        val xy = buildMatrix (xe_bfill, y, hparam, bakcast)
        val fname = if fname_ == null then formNames (xe.dim2, hparam) else fname_
        new ARX (xy, y, hh, xe.dim2, fname, tRng, hparam, bakcast)
    end apply

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create an `ARX` object by building an input matrix xy and then calling the
     *  `ARX` constructor, with rescaling of endogneous and exogenous variable values.
     *  @param xe       the matrix of exogenous variable values
     *  @param y        the endogenous/response vector (main time series data)
     *  @param hh       the maximum forecasting horizon (h = 1 to hh)
     *  @param fname_   the feature/variable names
     *  @param tRng     the time range, if relevant (time index may suffice)
     *  @param hparam   the hyper-parameters
     *  @param fEndo    the set of transforms to be used for the endogenous
     *  @param fExo     the array containing the sets of transforms to be used for the exogenous
     *  @param bakcast  whether a backcasted value is prepended to the time series (defaults to false)
     *  @param tFormT   the transform for rescaling endogenous and exogenous
     */
    def rescale (xe: MatrixD, y: VectorD, hh: Int, fname_ : Array [String] = null,
                 tRng: Range = null, hparam: HyperParameter = hp,
//               fEndo: LSET [Transform] = null,
//               fExo: Array [LSET [Transform]] = Array (),
                 bakcast: Boolean = false,
                 tFormT: TransformT = MinMax): ARX =

        if tFormT.name == "NormForm" then hparam("nneg") = 0

        // rescale y
        val tFormScale = tFormT.form
        val tr_size = Model.trSize (y.dim)
        val tForm_y = tFormScale (y(0 until tr_size))                       // use (mean, std) of training set for both In-sample and TnT
        val y_scl   = tForm_y.f(y)

        var xe_bfill: MatrixD = null
        if xe.dim2 > 0 and hparam("q").toInt > 0 then
            xe_bfill = new MatrixD (xe.dim, xe.dim2)
            for j <- xe.indices2 do xe_bfill(?, j) = backfill (xe(?, j))    // backfill each exogenous variable
            if tFormScale != null then
                val tForm_exo = tFormScale (xe_bfill(0 until tr_size))
                xe_bfill      = tForm_exo.f (xe_bfill)                      // rescale the backfilled exogenous variable

        val tForms = Map ("tForm_y" -> tForm_y)
        val xy     = buildMatrix (xe_bfill, y_scl, hparam, bakcast)
        val fname  = if fname_ == null then formNames (xe.dim2, hparam) else fname_
        new ARX (xy, y_scl, hh, xe.dim2, fname, tRng, hparam, bakcast, tForms)
    end rescale

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create an `ARX` object by building an input matrix xy and then calling the
     *  `ARX` constructor.
     *  @param xe       the matrix of exogenous variable values
     *  @param y        the endogenous/response vector (main time series data)
     *  @param hh       the maximum forecasting horizon (h = 1 to hh)
     *  @param fname_   the feature/variable names
     *  @param tRng     the time range, if relevant (time index may suffice)
     *  @param hparam   the hyper-parameters
     *  @param bakcast  whether a backcasted value is prepended to the time series (defaults to false)
     */
    def trend (xe: MatrixD, y: VectorD, hh: Int, fname_ : Array [String] = null,
               tRng: Range = null, hparam: HyperParameter = hp,
               bakcast: Boolean = false): ARX =

        val hp2 = hparam.updateReturn (("p", 0), ("q", 0))
        val xe_bfill: MatrixD = null

        val xy = buildMatrix (xe_bfill, y, hparam, bakcast)
        val fname = if fname_ == null then formNames (xe.dim2, hparam) else fname_
        new ARX (xy, y, hh, xe.dim2, fname, tRng, hp2, bakcast)
    end trend

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Build the input matrix by combining the spec + p columns for the trend and
     *  endogenous variable with the q * xe.dim2 columns for the exogenous variables.
     *  When cross = true, additional cross terms will be added.  Columns produced
     *  by transformations will be added as well.
     *  @param xe_bfill  the matrix of exogenous variable values
     *  @param y         the endogenous/response vector (main time series data)
     *  @param hp_       the hyper-parameters
     *  @param bakcast   whether a backcasted value is prepended to the time series (defaults to false)
     */
    def buildMatrix (xe_bfill: MatrixD, y: VectorD, hp_ : HyperParameter, bakcast: Boolean): MatrixD =

        val (p, q, spec, lwave) = (hp_("p").toInt, hp_("q").toInt, hp_("spec").toInt, hp_("lwave").toDouble)

        // make matrix xy for trend terms and lagged terms of the endogenous variable
        var xy = makeMatrix4T (y, spec, lwave, bakcast) ++^                 // trend terms
                 makeMatrix4L (y, p, bakcast)                               // lagged linear terms

        // apply transformations fExo to the exogenous variables and add there columns to x_exo
        if xe_bfill != null and q > 0 then
            xy = xy ++^ makeMatrix4L (xe_bfill, q, bakcast)                 // add lagged exogenous term to xy

        println (s"xy.dims = ${xy.dims}")
        xy
    end buildMatrix

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Form an array of names for the features included in the model.
     *  @param n_exo     the number of exogenous variable
     *  @param hp_       the hyper-parameters
     *  @param n_fEn     the number of functions used to map endogenous variables (none for `ARX`)
     *  @param n_fExArr  the number of functions used to map exogenous variables (none for `ARX`)
     */
    def formNames (n_exo: Int, hp_ : HyperParameter, n_fEn: Int = 0, n_fExArr: Array [Int] = null): Array [String] =

        val (p, q, spec) = (hp_("p").toInt, hp_("q").toInt, hp_("spec").toInt)
        val names = VEC [String] ()
        for j <- 0 until n_exo; k <- q to 1 by -1 do names += s"xe${j}l$k"
        MakeMatrix4TS.formNames (spec, p) ++ names.toArray
    end formNames

end ARX

import Example_LakeLevels.y

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRXTest` main function tests the `ARX` class on real data:
 *  Forecasting Lake Levels using In-Sample Testing (In-ST).
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  @see cran.r-project.org/web/packages/fpp/fpp.pdf
 *  > runMain scalation.modeling.forecasting.aRXTest
 */
@main def aRXTest (): Unit =

    val hh = 3                                                          // maximum forecasting horizon

    val mod = ARX (EMPTY_EXO, y, hh)                                    // create model for time series data
    banner (s"In-ST Forecasts: ${mod.modelName} on LakeLevels Dataset")
    mod.inSample_Test ()                                                // In-sample Testing
    println (mod.summary ())                                            // statistical summary of fit

end aRXTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRXTest2` main function tests the `ARX` class on real data:
 *  Forecasting Lake Levels using Train-n-Test Split (TnT) with Rolling Validation.
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  @see cran.r-project.org/web/packages/fpp/fpp.pdf
 *  > runMain scalation.modeling.forecasting.aRXTest2
 */
@main def aRXTest2 (): Unit =

    val hh = 3                                                          // maximum forecasting horizon

    val mod = ARX (EMPTY_EXO, y, hh)                                    // create model for time series data
    banner (s"TnT Forecasts: ${mod.modelName} on LakeLevels Dataset")
    mod.tnT_Test ()                                                     // Train and Test with Rolling Validation

end aRXTest2

import Example_Covid._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRXTest3` main function tests the `ARX` class on real data:
 *  Forecasting COVID-19 using In-Sample Testing (In-ST).
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  Comparison of sMAPE for AR(p), ARX(p), ARX_D(p), ARX_RD_p),
 *  ARX_Quad(p), ARX_Quad_D, ARX_Quad_RD, ARX_SR
 *
 *  19.0371,    29.5797,    39.0740,    47.4638,    55.1785,    62.1818  RW
 *
 *  18.7298,    28.4908,    37.4800,    46.3173,    53.3245,    59.5733  AR(1)
 *  17.9570,    27.8389,    38.2202,    46.8801,    53.9360,    60.0898  ARX(1)
 *  17.9618,    28.2405,    37.9195,    46.3640,    52.7653,    57.5828  ARX_D(1)
 *  18.8784,    29.6315,    38.6593,    45.3175,    50.2741,    52.6499  ARX_RD(1)
 *  18.1510,    28.9302,    39.1532,    47.4220,    53.9358,    59.8057  ARX_Quad(1) @1.8
 *  18.1307,    28.8176,    39.3560,    47.6758,    53.5738,    58.1790  ARX_Quad_D(1)
 *  19.0236,    30.0517,    38.7271,    44.9628,    49.8305,    52.0660  ARX_Quad_RD(1)
 *  18.1286,    29.9886,    39.4455,    46.9350,    52.5416,    56.9551  ARX_SR(1) with DQuad (.1mx)
 *  18.3111,    30.0877,    39.8794,    47.4267,    53.2857,    58.0669  ARX_SR(1) with DQuad (.2mx)
 *  19.5105,    30.7110,    40.7739,    48.1174,    53.2060,    56.8004  ARX_SR_D(1) with DQuad (.2mx)
 *  18.8796,    29.6807,    37.3175,    43.4480,    47.3335,    49.8975  ARX_SR_RD(1) with DQuad (.2mx)
 *
 *  16.3579,    24.7155,    33.0480,    40.1643,    46.8762,    53.2178  AR(2)
 *  15.6434,    23.1377,    31.4817,    39.7497,    47.1402,    54.4322  ARX(2)
 *  15.6462,    22.9254,    30.2138,    37.9797,    45.5425,    53.3514  ARX_D(2)
 *  16.2990,    23.8779,    31.1601,    36.5220,    41.4915,    46.4906  ARX_RD(2)
 *  15.7432,    23.3563,    32.3720,    40.6980,    47.5430,    54.6544  ARX_Quad(2)
 *  15.7573,    23.2011,    31.3754,    39.0022,    45.8773,    52.7765  ARX_Quad_D(2)
 *  16.3329,    23.3762,    30.7640,    36.2625,    41.1872,    46.0774  ARX_Quad_RD(2)
 *  14.3090,    23.2857,    31.2506,    37.4457,    43.7610,    48.8745  ARX_SR(2) with DQuad (.1mx)
 *  15.2542,    25.6917,    32.9542,    39.0075,    43.7206,    49.5221  ARX_SR(2) with DQuad (.2mx)
 *  15.4270,    24.1621,    31.6682,    38.9184,    43.2447,    49.5650  ARX_SR_D(2) with DQuad (.2mx)
 *  15.4608,    23.7257,    28.3966,    32.8648,    38.0295,    44.0833  ARX_SR_RD(2) with DQuad (.2mx)
 *
 *  16.0114,    22.7408,    29.5631,    35.2773,    41.5856,    47.5716  AR(3)
 *  15.2466,    21.9209,    28.9117,    36.8809,    44.3765,    51.6056  ARX(3)
 *  15.2577,    21.7428,    29.0617,    36.9863,    45.2127,    53.1253  ARX_D(3)
 *  15.9398,    22.7960,    29.2415,    34.0072,    39.8081,    44.2387  ARX_RD(3)
 *  15.2283,    21.7942,    29.3575,    37.5463,    43.4328,    50.1804  ARX_Quad(3)
 *  15.2930,    21.9746,    30.1124,    37.6141,    44.8848,    51.6602  ARX_Quad_D(3)
 *  15.7469,    22.2693,    29.2256,    33.8352,    40.1248,    44.2459  ARX_Quad_RD(3)
 *  14.0271,    21.6754,    27.8942,    33.1746,    38.7099,    44.2036  ARX_SR(3) with DQuad (.1mx)
 *  14.0003,    23.6392,    29.8588,    37.7099,    42.9960,    49.2562  ARX_SR(3) with DQuad (.2mx)
 *  14.0545,    22.2943,    29.5713,    36.5834,    41.8106,    47.2197  ARX_SR_D(3) with DQuad (.2mx)
 *  14.7697,    21.6841,    26.2704,    30.6927,    36.6370,    41.7517  ARX_SR_RD(3) with DQuad (.2mx)
 *
 *  15.8988,    22.5738,    28.5298,    33.3360,    39.1586,    44.3459  AR(4)
 *  15.2111,    21.6626,    28.2550,    36.4412,    43.5074,    50.8344  ARX(4)
 *  15.2240,    21.7621,    29.0603,    37.3265,    45.6671,    53.4636  ARX_D(4)
 *  15.8299,    22.6619,    28.8738,    33.7431,    39.5870,    43.4979  ARX_RD(4)
 *  15.1834,    21.7630,    29.9297,    37.2337,    43.2314,    48.1949  ARX_Quad(4)
 *  15.2944,    21.9995,    30.3745,    37.7328,    43.9640,    49.9208  ARX_Quad_D(4)
 *  15.7572,    22.5344,    29.3159,    34.5332,    39.9878,    43.8639  ARX_Quad_RD(4)
 *  14.0509,    21.8374,    27.6873,    32.5337,    38.1541,    43.2345  ARX_SR(4) with DQuad (.1mx)
 *  13.7250,    22.6431,    28.3822,    35.6001,    39.1553,    44.1104  ARX_SR(4) with DQuad (.2mx)
 *  13.7942,    21.0103,    28.2074,    34.3365,    40.6701,    46.2519  ARX_SR_D(4) with DQuad (.2mx)
 *  14.3905,    20.4449,    24.7036,    30.0186,    35.9514,    40.9009  ARX_SR_RD(4) with DQuad (.2mx)
 *
 *  15.9279,    22.5769,    28.5035,    33.3019,    39.1381,    43.0520  AR(5)
 *  14.9614,    21.5226,    28.7885,    36.8389,    44.4641,    51.8397  ARX(5)
 *  15.0044,    21.6113,    29.3029,    37.7081,    45.7620,    53.3911  ARX_D(5)
 *  15.8402,    22.6580,    28.8706,    33.7429,    39.6313,    43.6473  ARX_RD(5)
 *  14.6721,    22.0108,    31.0667,    38.7504,    44.8782,    49.7450  ARX_Quad(5)
 *  14.8070,    21.8999,    30.4074,    37.6566,    43.9429,    49.6169  ARX_Quad_D(5)
 *  15.4832,    22.5912,    29.5305,    34.7435,    40.0628,    44.2391  ARX_Quad_RD(5)
 *  13.7285,    20.7820,    26.4331,    30.8550,    36.9136,    42.2655  ARX_SR(5) with DQuad (.1mx)
 *  12.5208,    20.4209,    27.1075,    33.2648,    36.6577,    40.2518  ARX_SR(5) with DQuad (.2mx)
 *  12.6476,    19.7826,    27.0071,    32.9171,    38.2884,    43.9415  ARX_SR_D(5) with DQuad (.2mx)
 *  12.4809,    19.6497,    24.2383,    29.6165,    34.7276,    39.7801  ARX_SR_RD(5) with DQuad (.2mx)
 *
 *  > runMain scalation.modeling.forecasting.aRXTest3
 */
@main def aRXTest3 (): Unit =

    val (xe, y) = clip (loadData (Example_Covid.NO_EXO))                // for testing against `ARY, should be the same
//  val (xe, y) = clip (loadData (BEST_EXO))
    val hh      = 6                                                     // maximum forecasting horizon
    hp("lwave") = 20                                                    // wavelength (distance between peaks)
    RidgeRegression.hp("lambda") = 1.0                                  // regularization/shrinkage parameter

//  new Plot (null, y, null, s"y (new_deaths) vs. t", lines = true)
//  if xe.dim2 > 0 then new Plot (null, xe(?, 0), null, s"x_0 (${BEST_EXO(0)}) vs. t", lines = true)

    for p <- 1 to 5; q <- 0 to 0; s <- 0 to 0 do
        hp.set (("p", p), ("q", q), ("spec", s))                        // # endo lags, # exo lags, trend specification: 0, 1, 2, 3, 4
        val mod = ARX (xe, y, hh)                                       // create model for time series data
//      val mod = ARX.rescale (xe, y, hh, tFormT = Log1p)
        banner (s"In-ST Forecasts: ${mod.modelName} on COVID-19 Dataset")
        mod.inSample_Test ()                                            // In-sample Testing
        println (mod.summary ())                                        // statistical summary of fit
    end for

/*
    banner ("Test NormForm Transform")
    val yform  = NormForm (y)
    val y_     = yform.f (y)
    val xeform = NormForm (xe(?, 0))
    val xe_    = xeform.f (xe(?, 0))
    new Plot (null, y_, xe_, s"y (new_deaths) vs. t", lines = true)
*/

    for s <- 0 to 5 do
        hp("spec") = s                                                  // trend specification: 0, 1, 2, 3, 4
        val mod = ARX.trend (xe, y, hh)                                 // create model for time series data
        banner (s"In-ST Trend (s = $s) Forecasts: ${mod.modelName} on COVID-19 Dataset")
        mod.inSample_Test ()                                            // In-sample Testing
//      println (mod.summary ())                                        // statistical summary of fit
    end for

    //ZZ

end aRXTest3


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRXTest4` main function tests the `ARX` class on real data:
 *  Forecasting COVID-19 using Train-n-Test Split (TnT) with Rolling Validation.
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  Comparison of sMAPE for AR(p), ARX(p), ARX_D(p), ARX_RD_p),
 *  ARX_Quad(p), ARX_Quad_D, ARX_Quad_RD, ARX_SR
 *
 *  18.1532,    27.2211,    40.3519,    52.3739,    62.5276,    73.6424  RW
 *
 *  18.4029,    29.5363,    41.9465,    53.3411,    62.7636,    73.1441  AR(1)
 *  18.3339,    28.2821,    41.4751,    53.3526,    63.1629,    73.7484  ARX(1)
 *  18.3350,    29.0193,    42.6355,    54.5750,    63.9505,    73.4665  ARX_D(1)
 *  18.6587,    31.6942,    45.4174,    56.8690,    63.8777,    70.1173  ARX_RD(1)
 *  18.3998,    28.5716,    41.4413,    53.1146,    62.1894,    71.9894  ARX_Quad(1) @1.8
 *  18.3994,    29.5500,    42.5357,    53.7233,    60.8267,    67.7903  ARX_Quad_D(1)
 *  18.9221,    31.3191,    46.1923,    58.5189,    66.0335,    72.6569  ARX_Quad_RD(1)
 *  18.5031,    29.0971,    42.0843,    53.7853,    62.5209,    71.7928  ARX_SR(1) with DQuad (.6mx)
 *  18.7286,    31.8397,    45.9592,    57.6019,    64.5924,    70.8347  ARX_SR_D(1) with DQuad (.6mx)
 *  18.9804,    31.2814,    46.5726,    59.3222,    67.1032,    73.8360  ARX_SR_RD(1) with DQuad (.6mx)
 *
 *  18.0433,    26.2445,    38.7122,    49.5094,    58.1561,    69.7676  AR(2)
 *  17.7747,    26.0295,    36.9176,    49.3267,    59.5828,    72.3176  ARX(2)
 *  17.7814,    25.4928,    32.4529,    45.8697,    52.5594,    67.1803  ARX_D(2)
 *  17.6815,    25.7698,    35.5386,    44.6552,    49.7403,    59.3480  ARX_RD(2)
 *  17.8052,    27.9975,    40.9754,    53.0931,    63.3634,    74.3260  ARX_Quad(2)
 *  17.8197,    25.7513,    34.4427,    43.6920,    50.2977,    62.0854  ARX_Quad_D(2)
 *  18.4329,    26.8344,    36.8511,    47.6132,    53.2063,    63.8937  ARX_Quad_RD(2)
 *  17.5166,    28.2235,    42.2793,    55.1799,    66.4042,    76.4367  ARX_SR(2) with DQuad (.6mx)
 *  18.4715,    24.6244,    33.8790,    45.2572,    50.6426,    60.7419  ARX_SR_D(2) with DQuad (.6mx)
 *  18.1256,    27.1247,    35.8479,    47.8026,    53.0220,    64.4222  ARX_SR_RD(2) with DQuad (.6mx)
 *
 *  16.4218,    23.5256,    32.6385,    44.4111,    51.8210,    63.7353  AR(3)
 *  16.5613,    22.8588,    28.4131,    42.0704,    48.1937,    61.5939  ARX(3)
 *  16.5696,    21.8566,    28.0666,    40.7382,    49.2025,    63.8777  ARX_D(3)
 *  16.1730,    21.4435,    28.3385,    36.9935,    40.1137,    51.1967  ARX_RD(3)
 *  15.6404,    21.4870,    31.7588,    41.3497,    50.9569,    62.9595  ARX_Quad(3)
 *  15.6594,    20.3493,    27.3164,    34.9166,    43.3263,    56.0600  ARX_Quad_D(3)
 *  16.3157,    20.8165,    28.8194,    36.3519,    42.9462,    54.0447  ARX_Quad_RD(3)
 *  15.4467,    21.2986,    31.1643,    40.9819,    50.0013,    61.3770  ARX_SR(3) with DQuad (.6mx)
 *  16.6709,    20.5994,    28.5948,    37.6549,    43.7447,    54.5306  ARX_SR_D(3) with DQuad (.6mx)
 *  16.0799,    20.9558,    26.9962,    37.7361,    40.9038,    54.0283  ARX_SR_RD(3) with DQuad (.6mx)
 *
 *  14.8153,    21.8989,    29.9364,    39.3910,    47.8124,    53.5577  AR(4)
 *  15.9556,    22.2629,    26.7569,    40.0194,    43.4279,    57.3312  ARX(4)  
 *  15.9498,    22.0629,    27.7412,    41.0443,    50.1630,    66.4739  ARX_D(4)
 *  15.4216,    21.0296,    28.6235,    36.8928,    41.0351,    49.3166  ARX_RD(4)
 *  14.2151,    18.8819,    24.9838,    33.9136,    36.1594,    45.1952  ARX_Quad(4)
 *  14.1950,    18.5632,    25.0109,    34.8623,    40.1918,    53.2637  ARX_Quad_D(4)
 *  14.4044,    19.7075,    26.5422,    35.9619,    41.2083,    50.2606  ARX_Quad_RD(4)
 *  12.1334,    18.4577,    22.3107,    32.5163,    35.4627,    43.5519  ARX_SR(4) with DQuad (.6mx)
 *  15.3903,    18.7594,    25.2308,    36.7434,    42.8923,    53.6184  ARX_SR_D(4) with DQuad (.6mx)
 *  12.4424,    18.5479,    24.1643,    36.0140,    38.8692,    49.8293  ARX_SR_RD(4) with DQuad (.6mx)
 *
 *  14.3593,    21.9036,    29.1791,    40.6232,    46.1989,    55.2531  AR(5)
 *  15.5574,    21.3477,    26.6765,    39.4747,    44.1251,    55.0460  ARX(5)
 *  15.5692,    21.9239,    27.4043,    40.8279,    53.9456,    67.9537  ARX_D(5)
 *  15.8055,    20.9975,    28.7802,    37.3350,    41.3091,    49.4173  ARX_RD(5)
 *  14.3814,    18.6770,    25.0875,    33.9770,    34.4636,    45.3979  ARX_Quad(5)  
 *  14.3806,    19.3271,    25.2710,    34.6718,    39.8364,    51.1628  ARX_Quad_D(5)
 *  15.0146,    20.6906,    26.3743,    37.4673,    42.1440,    51.1924  ARX_Quad_RD(5)
 *  11.8667,    18.4592,    22.4543,    32.2853,    35.0223,    44.6631  ARX_SR(5) with DQuad (.6mx)
 *  15.5139,    18.6594,    24.1764,    35.9191,    41.7688,    52.6745  ARX_SR_D(5) with DQuad (.6mx)
 *  12.1503,    19.6588,    24.3819,    36.7356,    39.9623,    49.4441  ARX_SR_RD(5) with DQuad (.6mx)
 *
 *  > runMain scalation.modeling.forecasting.aRXTest4
 */
@main def aRXTest4 (): Unit =

    val (xe, y) = clip (loadData (Example_Covid.NO_EXO))                // for testing against `ARY, should be the same
//  val (xe, y) = clip (loadData (BEST_EXO))
    val hh      = 6                                                     // maximum forecasting horizon
    hp("lwave") = 20                                                    // wavelength (distance between peaks)
    Fit.hp("omega") = 0.0                                               // blending parameter WAPE (0) <-> NMAE (1)
    RidgeRegression.hp("lambda") = 1.0                                  // regularization/shrinkage parameter

    for p <- 1 to 5; q <- 0 to 0; s <- 0 to 0 do
        hp.set (("p", p), ("q", q), ("spec", s))                        // # endo lags, # exo lags, trend specification: 0, 1, 2, 3, 4
        val mod = ARX (xe, y, hh)                                       // create model for time series data
//      val mod = ARX.rescale (xe, y, hh, tFormT = Log1p)
        banner (s"TnT Forecasts: ${mod.modelName} on COVID-19 Dataset")
        mod.tnT_Test ()                                                 // Train and Test with Rolling Validation
    end for

end aRXTest4


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRXTest5` main function tests the `ARX` class on real data:
 *  Forecasting COVID-19 using In-Sample Testing (In-ST).
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  This version performs Feature Selection.
 *  > runMain scalation.modeling.forecasting.aRXTest5
 */
@main def aRXTest5 (): Unit =

    val (xe, y) = clip (loadData (BEST2_EXO))
    val hh      = 6                                                     // maximum forecasting horizon
    hp("p")     = 10                                                    // endo lags
    hp("q")     = 5                                                     // exo lags
    hp("spec")  = 2                                                     // trend specification: 0, 1, 2, 3, 4
    hp("lwave") = 20                                                    // wavelength (distance between peaks)
    RidgeRegression.hp("lambda") = 1.0                                  // regularization/shrinkage parameter
    FeatureSelection.fullset_FS = true                                  // use full dataset for FS

    val mod = ARX (xe, y, hh)                                           // create model for time series data
    banner (s"In-ST Forecasts: ${mod.modelName} on COVID-19 Dataset")
    mod.inSample_Test ()                                                // In-sample Testing
    println (mod.summary ())                                            // statistical summary of fit

    import SelectionTech._                                              // one of Forward, Backward, Stepwise, Beam

//  for tech <- values do                                               // try all feature selection techniques
//  for tech <- values if tech == Forward do                            // try a particular one
//  for tech <- values if tech == Backward do                           // try a particular one
//  for tech <- values if tech == Stepwise do                           // try a particular one
    for tech <- values if tech == Beam do                               // try a particular one
        banner (s"Feature Selection Technique: $tech")
        val (cols, rSq) = mod.selectFeatures (tech, "none")             // R^2, R^2 bar, sMAPE, sMAPEC
        val k = cols.size
        println (s"k = $k")

        val modBest = mod.getBest.mod                                   // regress on this x
        println (s"full model: ${stringOf (mod.getFname)}")             // feature names for full model
        println (s"best model: ${stringOf (modBest.getFname)}")         // feature names for best model

        new PlotM (null, rSq, Regression.metrics, s"R^2 vs k for ${mod.modelName} with $tech", lines = true)
        banner (s"Feature Selection Key Metrics using $tech")
        for r_ <- rSq do println (s"$tech: rSq = $r_")

//      val imp = mod.importance (cols.toArray, rSq.ᵀ)
//      println (s"feature importance imp = $imp")
//      for (c, r) <- imp do println (s"col = $c, \t ${ox_fname(c)}, \t importance = $r")
    end for

end aRXTest5


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRXTest6` main function tests the `ARX` class on real data:
 *  Forecasting COVID-19 using Train-n-Test Split (TnT) with Rolling Validation.
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  This version performs Feature Selection (training-set, training-set).
 *  > runMain scalation.modeling.forecasting.aRXTest6
 */
@main def aRXTest6 (): Unit =

    val (xe, y) = clip (loadData (BEST2_EXO))
    val hh      = 6                                                     // maximum forecasting horizon
    hp("p")     = 10                                                    // endo lags
    hp("q")     = 5                                                     // exo lags
    hp("spec")  = 2                                                     // trend specification: 0, 1, 2, 3, 4
    hp("lwave") = 20                                                    // wavelength (distance between peaks)
    RidgeRegression.hp("lambda") = 1.0                                  // regularization/shrinkage parameter

    val mod = ARX (xe, y, hh)                                           // create model for time series data
    banner (s"Training-only Forecasts: ${mod.modelName} on COVID-19 Dataset")
    mod.trainNtest_x ()()                                               // initially train and test on full dataset

//  mod.setSkip (p)                                                     // full AR-formula available when t >= p
    mod.forecastAll ()                                                  // forecast h-steps ahead (h = 1 to hh) for all y
    mod.diagnoseAll (mod.getY, mod.getYf)                               // QoF for each horizon
//  println (s"Training-only Forecast Matrix yf = ${mod.getYf}")

/*
    mod.setSkip (0)
    mod.rollValidate ()                                                 // TnT with Rolling Validation
    mod.diagnoseAll (mod.getY, mod.getYf, Forecaster.teRng(y.dim))
*/

    import SelectionTech._                                              // one of Forward, Backward, Stepwise, Beam

//  for tech <- values do                                               // try all feature selection techniques
//  for tech <- values if tech == Forward do                            // try a particular one
//  for tech <- values if tech == Backward do                           // try a particular one
//  for tech <- values if tech == Stepwise do                           // try a particular one
    for tech <- values if tech == Beam do                               // try a particular one
        banner (s"Feature Selection Technique: $tech")
        val (cols, rSq) = mod.selectFeatures (tech, "none")             // R^2, R^2 bar, sMAPE, sMAPEC
        val k = cols.size
        println (s"k = $k")

        val modBest = mod.getBest.mod                                   // regress on this x
        println (s"full model: ${stringOf (mod.getFname)}")             // feature names for full model
        println (s"best model: ${stringOf (modBest.getFname)}")         // feature names for best model

        new PlotM (null, rSq, Regression.metrics, s"R^2 vs k for ${mod.modelName} with $tech", lines = true)
        banner (s"Feature Selection Key Metrics using $tech")
        for r_ <- rSq do println (s"$tech: rSq = $r_")

//      val imp = mod.importance (cols.toArray, rSq.ᵀ)
//      println (s"feature importance imp = $imp")
//      for (c, r) <- imp do println (s"col = $c, \t ${ox_fname(c)}, \t importance = $r")
    end for

end aRXTest6


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRXTest7` main function tests the `ARX` class on real data:
 *  Forecasting COVID-19 using Train-n-Test Split (TnT) with Rolling Validation.
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  This version performs Feature Selection (training-set, testing-set).
 *  > runMain scalation.modeling.forecasting.aRXTest7
 */
@main def aRXTest7 (): Unit =

    val (xe, y) = clip (loadData (BEST2_EXO))
    val hh      = 1  // 6                                               // maximum forecasting horizon
    hp("p")     = 10                                                    // endo lags
    hp("q")     = 5                                                     // exo lags
    hp("spec")  = 2                                                     // trend specification: 0, 1, 2, 3, 4
    hp("lwave") = 20                                                    // wavelength (distance between peaks)
    RidgeRegression.hp("lambda") = 1.0                                  // regularization/shrinkage parameter

    val mod = ARX (xe, y, hh)                                           // create model for time series data
    banner (s"Training-Test Forecasts: ${mod.modelName} on COVID-19 Dataset")
    mod.trainNtest_x ()()                                               // initially train and test on full dataset

    import SelectionTech._                                              // one of Forward, Backward, Stepwise, Beam

//  for tech <- values do                                               // try all feature selection techniques
//  for tech <- values if tech == Forward do                            // try a particular one
//  for tech <- values if tech == Backward do                           // try a particular one
//  for tech <- values if tech == Stepwise do                           // try a particular one
    for tech <- values if tech == Beam do                               // try a particular one
        banner (s"Feature Selection Technique: $tech")
        val (cols, rSq) = mod.selectFeatures (tech, "none")             // R^2, R^2 bar, sMAPE, sMAPEC
        val k = cols.size
        println (s"k = $k")

        val modBest = mod.convertReg2Forc (mod.getBest.mod_cols)        // regress on this x
        println (s"full model: ${stringOf (mod.getFname)}")             // feature names for full model
        println (s"best model: ${stringOf (modBest.getFname)}")         // feature names for best model

        new PlotM (null, rSq, Regression.metrics, s"R^2 vs k for ${modBest.modelName} with $tech", lines = true)
        banner (s"Feature Selection Key Metrics using $tech")
        for r_ <- rSq do println (s"$tech: rSq = $r_")

        modBest.setSkip (0)
        modBest.rollValidate ()                                         // TnT with Rolling Validation
        mod.diagnoseAll (modBest.getY, modBest.getYf, Forecaster.teRng(y.dim))
//      println (s"Training-Test Forecast Matrix yf = ${modBest.getYf}")

//      val imp = mod.importance (cols.toArray, rSq.ᵀ)
//      println (s"feature importance imp = $imp")
//      for (c, r) <- imp do println (s"col = $c, \t ${ox_fname(c)}, \t importance = $r")
    end for

end aRXTest7


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRXTest8` main function tests the `ARX` class in terms of model specification
 *  formulas/equations.  Nore:  y~2 means y(t-2)
 *  > runMain scalation.modeling.forecasting.aRXTest8
 */
@main def aRXTest8 (): Unit =

    // Setup mock historic data vectors
    val y = VectorD (12.0, 15.0, 14.0, 16.0, 18.0, 21.0, 19.0, 22.0)
    val x = VectorD (4.0,    5.2,    4.8,    5.5,    6.0,    6.2,    5.9,    6.5)

    // Model Coefficients (Estimated via Ordinary Least Squares or similar)
    val c  = 1.5
    val φ = VectorD (-0.2, 0.6)
    val β = VectorD (1.1, 0.8)
    val r = (2, 1)                               // back 2, back 1

    println ("--- Step-by-Step ARX Prediction Loop ---")
    for timestamp <- 2 until y.dim do

        given t: Int = timestamp                 // activates contextual time lookup

        banner (s"timestamp t = $t")

        val y_pred = c + φ(0) * y~2 + φ(1) * y~1 + β(0) * x~2 + β(1) * x~1
        val y_pre2 = c + φ ∙ (y~r) + β ∙ (x~r)
        
        println (s"Scalar Formula: Time $t | Actual: ${y(t)} | Predicted: $y_pred")
        println (s"Vector Formula: Time $t | Actual: ${y(t)} | Predicte2: $y_pre2")

end aRXTest8


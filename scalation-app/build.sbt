// build.sbt
lazy val scalation = project.in(file("."))
  .settings(
    scalaVersion  := "3.8.4",
    scalacOptions ++= Seq(
        "-deprecation",             // emit warning and location for usages of deprecated APIs
        "-explain",                 // explain errors in more detail
        "-new-syntax",              // require `then` and `do` in control expressions.
        "-feature",                 // cause language feature usage warning to be detailed
        "-Wunused:imports",         // warn of unused imports, ...
        "-Wunused:all",             // warn of unused imports, ...
        "-Werror")                  // was "-Xfatal-warnings" fail the compilation if there are any warnings
//  javacOptions  += "--add-modules jdk.incubator.vector"
  )
fork := true
outputStrategy := Some(StdoutOutput)
// Parallel Collections
libraryDependencies += "org.scala-lang.modules" %% "scala-parallel-collections" % "1.2.0"

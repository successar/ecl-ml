IMPORT ML;
IMPORT ML.Types as Types;

NumericField := Types.NumericField;

EXPORT Logistic (DATASET(NumericField) X, DATASET(NumericField) Y, Types.t_FieldNumber iterations=5, BOOLEAN doNormalize=FALSE) :=
                  MODULE(ML.GradientBoosting.Classification.IClassification(X, Y, iterations, doNormalize))

  SHARED DATASET(NumericField) ComputeBetas(DATASET(NumericField) indeps, DATASET(NumericField) deps) := FUNCTION
    model := ML.Regression.sparse.OLS_Cholesky(indeps, deps);
    return model.Betas;
  END;

  SHARED DATASET(NumericField) PredictLocal(DATASET(NumericField) indeps, DATASET(NumericField) betas) := FUNCTION
    return ML.Regression.Predict(indeps, betas);
  END;

END;
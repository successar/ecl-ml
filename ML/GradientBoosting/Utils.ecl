IMPORT ML;
IMPORT ML.GradientBoosting.Losses as Losses;
IMPORT ML.Types as Types;

NumericField := Types.NumericField;
Regress := ML.Regression.sparse.OLS_Cholesky;
LossRecord := Losses.LossRecord;

EXPORT Utils := MODULE

  EXPORT DATASET(LossRecord) ComputeErrors(DATASET(NumericField) actuals, DATASET(NumericField) predicted) := FUNCTION
    return JOIN(actuals, predicted, LEFT.id=RIGHT.id, Losses.EstimateError(LEFT, RIGHT));
  END;

  EXPORT DATASET(NumericField) ComputeBetas(DATASET(NumericField) indeps, DATASET(NumericField) deps) := FUNCTION
    model := Regress(indeps, deps);
    return model.Betas;
  END;

  EXPORT DATASET(NumericField) PredictLocal(DATASET(NumericField) indeps, DATASET(NumericField) betas) := FUNCTION
    return ML.Regression.Predict(indeps, betas);
  END;

  EXPORT DATASET(LossRecord) ComputeLocalErrors(DATASET(NumericField) indeps, DATASET(NumericField) deps, DATASET(NumericField) betas) := FUNCTION
    predicted := PredictLocal(indeps, betas);
    return ComputeErrors(deps, predicted);
  END;

  SHARED Re := Types.t_FieldReal;
  EXPORT Re Norm(Re val, Re high, Re low) := FUNCTION
    RETURN (val - low) / (high - low);
  END;

  EXPORT Re DeNorm(Re val, Re high, Re low) := FUNCTION
    RETURN val * (high - low) + low;
  END;

END;
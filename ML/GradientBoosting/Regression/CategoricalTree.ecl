IMPORT ML;
IMPORT ML.Types as Types;
IMPORT ML.GradientBoosting.Losses as Losses;
IMPORT ML.GradientBoosting.GBTypes as GBTypes;

NumericField := Types.NumericField;

EXPORT CategoricalTree(DATASET(NumericField) X, DATASET(NumericField) Y,
            Types.t_Count min_NumObj=2,
            Types.t_level max_Level=6,
            Types.t_FieldNumber iterations=5,
            BOOLEAN doNormalize=FALSE) :=
              MODULE(ML.GradientBoosting.Regression.IRegression(X, Y, iterations, doNormalize))

  SHARED baseTree := ML.DecisionTree.Regression.ID3(min_NumObj, max_Level);

  SHARED DATASET(NumericField) ComputeBetas(DATASET(NumericField) indeps, DATASET(NumericField) deps) := FUNCTION
    return baseTree.LearnD(indeps, deps);
  END;

  SHARED DATASET(NumericField) PredictLocal(DATASET(NumericField) indeps, DATASET(NumericField) betas) := FUNCTION
    return ML.DecisionTree.Regression.Predict.PredictD(indeps, betas);
  END;

END;
IMPORT ML;
IMPORT ML.Types as Types;
IMPORT ML.GradientBoosting.Losses as Losses;
IMPORT ML.GradientBoosting.GBTypes as GBTypes;

NumericField := Types.NumericField;
GBUtils := ML.GradientBoosting.Utils;
LossRecord := Losses.LossRecord;
GBRecord := GBTypes.GBRegressionRecord;

EXPORT IRegression(DATASET(NumericField) X, DATASET(NumericField) Y,
                    Types.t_FieldNumber iterations, BOOLEAN doNormalize=FALSE) := MODULE, VIRTUAL

  SHARED maxY := MAX(Y, Y.value);
  SHARED minY := MIN(Y, Y.value);

  SHARED AggRecord := RECORD
    X.number;
    max_val:=MAX(GROUP, X.value);
    min_val:=MIN(GROUP, X.value);
  END;

  EXPORT Extremes := TABLE(X, AggRecord, number);

  SHARED VIRTUAL DATASET(NumericField) NormalizeX(DATASET(NumericField) indeps) := FUNCTION
    RETURN IF(doNormalize, JOIN(indeps, Extremes, LEFT.number=RIGHT.number, TRANSFORM(NumericField,
                           SELF.id:=LEFT.id, SELF.number:=LEFT.number, SELF.value:=GBUtils.Norm(LEFT.value,
                                    RIGHT.max_val, RIGHT.min_val))),
              indeps);
  END;

  SHARED DATASET(NumericField) NormalizeY(DATASET(NumericField) deps) := FUNCTION
    RETURN PROJECT(deps, TRANSFORM(NumericField, SELF.id:=LEFT.id, SELF.number:=LEFT.number,
                            SELF.value:=GBUtils.Norm(LEFT.value, maxY, minY)));
  END;

  SHARED DATASET(NumericField) DeNormalizeY(DATASET(NumericField) deps) := FUNCTION
    RETURN PROJECT(deps, TRANSFORM(NumericField, SELF.id:=LEFT.id, SELF.number:=LEFT.number,
                            SELF.value:=GBUtils.DeNorm(LEFT.value, maxY, minY)));
  END;

  EXPORT DATASET(NumericField) Independents := NormalizeX(X);
  EXPORT DATASET(NumericField) Dependents := NormalizeY(Y);

  SHARED Types.t_FieldNumber repeats := iterations;

  SHARED ComputeErrors := GBUtils.ComputeErrors;

  SHARED VIRTUAL DATASET(NumericField) ComputeBetas (DATASET(NumericField) indeps, DATASET(NumericField) deps) := FUNCTION
    RETURN DATASET([], NumericField);
  END;

  SHARED VIRTUAL DATASET(NumericField) PredictLocal (DATASET(NumericField) indeps, DATASET(NumericField) betas) := FUNCTION
    RETURN DATASET([], NumericField);
  END;

  SHARED DATASET(LossRecord) ComputeLocalErrors(DATASET(NumericField) indeps, DATASET(NumericField) deps, DATASET(NumericField) betas) := FUNCTION
    predicted := PredictLocal(indeps, betas);
    return ComputeErrors(deps, predicted);
  END;

  EXPORT DATASET(LossRecord) BaseModelPredictions() := FUNCTION
    mdl := ComputeBetas(Independents, Dependents);
    predicted := DeNormalizeY(PredictLocal(Independents, mdl));
    RETURN ComputeErrors(DeNormalizeY(Dependents), predicted);
  END;

  EXPORT DATASET(GBRecord) Learn() := FUNCTION
    records := PROJECT(Dependents, TRANSFORM(GBRecord, SELF.id:=LEFT.id, SELF.iteration:=1, SELF.isBeta:=FALSE,
                                    SELF.number:=LEFT.number, SELF.value:=LEFT.value));
    depIndex := MAX(Independents, number) + 1;
    GBRecord loopBody(DATASET(GBRecord) recs, INTEGER c) := FUNCTION
      actuals := PROJECT(recs(isBeta=FALSE, iteration=c),
                          TRANSFORM(NumericField, SELF.id:=LEFT.id, SELF.number:=LEFT.number, SELF.value:=LEFT.value));

      computed_betas := ComputeBetas(Independents, actuals);
      errors := ComputeLocalErrors(Independents, actuals, computed_betas);
      new_betas := PROJECT(computed_betas, TRANSFORM(GBRecord, SELF.id:=LEFT.id, SELF.iteration:=c,
                                                    SELF.isBeta:=TRUE, SELF.number:=LEFT.number,
                                                    SELF.value:=LEFT.value));
      new_actuals := PROJECT(errors, TRANSFORM(GBRecord, SELF.id:=LEFT.id, SELF.iteration:=c+1,
                                                SELF.isBeta:=FALSE, SELF.number:=depIndex,
                                                SELF.value:=LEFT.loss));
      RETURN recs(isBeta=TRUE) + new_betas + new_actuals;
    END;
    looped := LOOP(records, repeats, loopBody(ROWS(LEFT), COUNTER));
    RETURN looped(isBeta=TRUE);
  END;

  EXPORT DATASET(NumericField) Predict(DATASET(NumericField) indeps, DATASET(GBRecord) weights) := FUNCTION
    depIndex := MAX(Independents, number) + 1;
    norm_indeps := NormalizeX(indeps);
    GBRecord loopBody(DATASET(GBRecord) recs, INTEGER c) := FUNCTION
      betas := PROJECT(weights(isBeta=TRUE, iteration=c),
                       TRANSFORM(NumericField, SELF.id:=LEFT.id,
                       SELF.number:=LEFT.number, SELF.value:=LEFT.value));
      predicted := PredictLocal(norm_indeps, betas);
      predicted_records := PROJECT(predicted, TRANSFORM(GBRecord, SELF.id:=LEFT.id, SELF.iteration:=c,
                                              SELF.isBeta:=FALSE, SELF.number:=depIndex,
                                              SELF.value:=LEFT.value));
      RETURN recs + predicted_records;
    END;
    all_predicted := LOOP(DATASET([], GBRecord), repeats, loopBody(ROWS(LEFT), COUNTER));
    grped := RECORD
      all_predicted.id;
      pred:=SUM(GROUP, all_predicted.value);
    END;
    RETURN DeNormalizeY(PROJECT(TABLE(all_predicted, grped, id), TRANSFORM(NumericField,
                    SELF.id:=LEFT.id, SELF.number:=depIndex, SELF.value:=LEFT.pred)));
  END;
END;
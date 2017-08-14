IMPORT ML;
IMPORT ML.Types as Types;
IMPORT ML.GradientBoosting.Losses as Losses;
IMPORT ML.GradientBoosting.GBTypes as GBTypes;

NumericField := Types.NumericField;
GBUtils := ML.GradientBoosting.Utils;
LossRecord := Losses.LossRecord;
GBRecord := GBTypes.GBClassificationRecord;
ValueRecord := GBTypes.ValueRecord;
DependentRecord := GBTypes.DependentClassifierRecord;
LabeledNumericField := GBTypes.LabeledNumericField;

EXPORT IClassification (DATASET(NumericField) X, DATASET(NumericField) Y,
                    Types.t_FieldNumber iterations, BOOLEAN doNormalize=FALSE) := MODULE, VIRTUAL

  SHARED AggRecord := RECORD
    X.number;
    max_val:=MAX(GROUP, X.value);
    min_val:=MIN(GROUP, X.value);
  END;

  EXPORT Extremes := TABLE(X, AggRecord, number);

  SHARED VIRTUAL DATASET(DependentRecord) NormalizeX(DATASET(NumericField) indeps, DATASET(ValueRecord) classifiers) := FUNCTION
    normalized := IF(doNormalize, JOIN(indeps, Extremes, LEFT.number=RIGHT.number, TRANSFORM(NumericField,
                           SELF.id:=LEFT.id, SELF.number:=LEFT.number, SELF.value:=GBUtils.Norm(LEFT.value,
                                    RIGHT.max_val, RIGHT.min_val))),
              indeps);
    RETURN DISTRIBUTE(JOIN(normalized, classifiers, TRUE,
                      TRANSFORM(DependentRecord, SELF.classifier_ID:=RIGHT.id, SELF:=LEFT), ALL),
                      HASH(classifier_ID));
  END;

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

  // Metadata for classes.
  EXPORT DATASET(ValueRecord) Classes := PROJECT(DEDUP(SORT(TABLE(Y, {value}), value),value),
                                         TRANSFORM(ValueRecord, SELF.value:=LEFT.value, SELF.id:=COUNTER));

  // Generate depenedent variables
  // Dependents = #classes * #rows
  SHARED DATASET(DependentRecord) getDependents() := FUNCTION
    DependentRecord loopBody(DATASET(DependentRecord) recs, Types.t_FieldNumber c) := FUNCTION
      class := MAX(Classes(id=c), value);
      deps := PROJECT(Y, TRANSFORM(DependentRecord, SELF.id:=LEFT.id,
                      SELF.number:=LEFT.number, SELF.classifier_ID:=c,
                      SELF.value:=IF(LEFT.value=class, 1.0, 0.0)));
      RETURN recs + deps;
    END;
    unhashed_deps := LOOP(DATASET([], DependentRecord), COUNT(Classes), loopBody(ROWS(LEFT), COUNTER));
    hashed_deps := DISTRIBUTE(unhashed_deps, HASH(classifier_ID));
    RETURN hashed_deps;
  END;

  // Independent Variables
  EXPORT DATASET(DependentRecord) Independents := NormalizeX(X, Classes);
  // Dependent Variables
  EXPORT DATASET(DependentRecord) Dependents := getDependents();
  // Number of Iterations
  SHARED Types.t_FieldNumber repeats := iterations;

  // Perform Gradient Boosting Regression for one classifier
  EXPORT DATASET(GBRecord) LearnClassifier(Types.t_FieldNumber c_id) := FUNCTION
    records := PROJECT(Dependents(classifier_ID=c_id),
                      TRANSFORM(GBRecord, SELF.id:=LEFT.id, SELF.iteration:=1, SELF.isBeta:=FALSE,
                      SELF.number:=LEFT.number, SELF.value:=LEFT.value, SELF.classifier_ID:=LEFT.classifier_ID));
    classifier_indeps := PROJECT(Independents(classifier_ID=c_id), TRANSFORM(NumericField, SELF:=LEFT));
    depIndex := MAX(classifier_indeps, number) + 1;
    GBRecord loopBody(DATASET(GBRecord) recs, INTEGER c) := FUNCTION
      actuals := PROJECT(recs(isBeta=FALSE, iteration=c),
                          TRANSFORM(NumericField, SELF.id:=LEFT.id, SELF.number:=LEFT.number, SELF.value:=LEFT.value));
      computed_betas := ComputeBetas(classifier_indeps, actuals);
      errors := ComputeLocalErrors(classifier_indeps, actuals, computed_betas);
      new_betas := PROJECT(computed_betas, TRANSFORM(GBRecord, SELF.id:=LEFT.id, SELF.iteration:=c,
                                                    SELF.isBeta:=TRUE, SELF.number:=LEFT.number,
                                                    SELF.value:=LEFT.value, SELF.classifier_ID:=c_id));
      new_actuals := PROJECT(errors, TRANSFORM(GBRecord, SELF.id:=LEFT.id, SELF.iteration:=c+1,
                                                SELF.isBeta:=FALSE, SELF.number:=depIndex,
                                                SELF.value:=LEFT.loss, SELF.classifier_ID:=c_id));
      RETURN recs(isBeta=TRUE) + new_betas + new_actuals;
    END;
    looped := LOOP(records, repeats, loopBody(ROWS(LEFT), COUNTER));
    RETURN looped(isBeta=TRUE);
  END;

  // Perform Gradient Boosting Classification
  // Returns weights for each local regressor for each iteration.
  EXPORT DATASET(GBRecord) Learn() := FUNCTION
    GBRecord loopBody(DATASET(GBRecord) rs, Types.t_FieldNumber c_id) := FUNCTION
      new_rs := LearnClassifier(c_id);
      RETURN rs + new_rs;
    END;
    looped := LOOP(DATASET([], GBRecord), COUNT(Classes), loopBody(ROWS(LEFT), COUNTER));
    RETURN looped;
  END;

  SHARED DATASET(LabeledNumericField) PredictClassifier(DATASET(NumericField) indeps,
              DATASET(GBRecord) weights, Types.t_FieldNumber c_id) := FUNCTION
      depIndex := MAX(X, number) + 1;
      GBRecord loopBody(DATASET(GBRecord) recs, Types.t_FieldNumber c) := FUNCTION
        betas := PROJECT(weights(isBeta=TRUE, iteration=c, classifier_id=c_id),
                         TRANSFORM(NumericField, SELF.id:=LEFT.id,
                         SELF.number:=LEFT.number, SELF.value:=LEFT.value));
        predicted := PredictLocal(indeps, betas);
        predicted_records := PROJECT(predicted, TRANSFORM(GBRecord, SELF.id:=LEFT.id, SELF.iteration:=c,
                                                SELF.isBeta:=FALSE, SELF.number:=depIndex,
                                                SELF.value:=LEFT.value, SELF.classifier_ID:=c_id));
        RETURN recs + predicted_records;
      END;
      all_predicted := LOOP(DATASET([], GBRecord), repeats, loopBody(ROWS(LEFT), COUNTER));
      grped := RECORD
        all_predicted.id;
        all_predicted.classifier_ID;
        pred:=SUM(GROUP, all_predicted.value);
      END;
      RETURN PROJECT(TABLE(all_predicted, grped, id, classifier_ID),
                    TRANSFORM(LabeledNumericField, SELF.id:=LEFT.id,
                    SELF.number:=depIndex, SELF.value:=LEFT.pred,
                    SELF.label:=LEFT.classifier_ID));
  END;

  EXPORT DATASET(LabeledNumericField) Predict(DATASET(NumericField) indeps, DATASET(GBRecord) weights) := FUNCTION
    norm_indeps := NormalizeX(indeps, Classes);
    LabeledNumericField loopBody(DATASET(LabeledNumericField) recs, Types.t_FieldNumber c_id) := FUNCTION
      new_recs := PredictClassifier(PROJECT(norm_indeps(classifier_ID=c_id), TRANSFORM(NumericField, SELF:=LEFT)), weights, c_id);
      RETURN recs + new_recs;
    END;
    looped := SORT(
                LOOP(DATASET([], LabeledNumericField), COUNT(Classes), loopBody(ROWS(LEFT), COUNTER)),
                id, -value
              );
    top_scores := TABLE(
                looped,
                {looped.id, best_score:=MAX(GROUP, looped.value)},
                id);
    pred_joined := JOIN(looped, top_scores, LEFT.id=RIGHT.id AND LEFT.value=RIGHT.best_score);
    predictions := JOIN(pred_joined, Classes,
                  LEFT.label=RIGHT.id,
                  TRANSFORM(LabeledNumericField, SELF.id:=LEFT.id,
                  SELF.number:=LEFT.number, SELF.value:=LEFT.value,
                  SELF.label:=RIGHT.value));
    RETURN predictions;
  END;

END;
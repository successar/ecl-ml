IMPORT ML;
IMPORT ML.Types as Types;
IMPORT ML.GradientBoosting.Losses as Losses;
IMPORT ML.GradientBoosting.GBTypes as GBTypes;

NumericField := Types.NumericField;
FieldType := ML.DecisionTree.Utils.FieldType;
GBUtils := ML.GradientBoosting.Utils;
DependentRecord := GBTypes.DependentClassifierRecord;
ValueRecord := GBTypes.ValueRecord;

EXPORT MixedTree(DATASET(NumericField) X, DATASET(NumericField) Y,
                 DATASET(FieldType) f_types,
                 Types.t_Count min_NumObj=2,
                 Types.t_level max_Level=6,
                 Types.t_FieldNumber iterations=5,
                 BOOLEAN doNormalize=FALSE) :=
              MODULE(ML.GradientBoosting.Classification.IClassification(X, Y, iterations, doNormalize))

  SHARED Type_Extremes := JOIN(Extremes, f_types, LEFT.number=RIGHT.number);

  SHARED DATASET(DependentRecord) NormalizeX(DATASET(NumericField) indeps, DATASET(ValueRecord) classifiers) := FUNCTION
    normalized := IF(doNormalize, JOIN(indeps, Type_Extremes, LEFT.number=RIGHT.number, TRANSFORM(NumericField,
                           SELF.id:=LEFT.id, SELF.number:=LEFT.number,
                           SELF.value:=IF(RIGHT.is_continuous, GBUtils.Norm(LEFT.value,RIGHT.max_val, RIGHT.min_val), LEFT.value))),
              indeps);
    RETURN DISTRIBUTE(JOIN(normalized, classifiers, TRUE,
                      TRANSFORM(DependentRecord, SELF.classifier_ID:=RIGHT.id, SELF:=LEFT), ALL),
                      HASH(classifier_ID));
  END;

  SHARED baseTree := ML.DecisionTree.Regression.ID3(min_NumObj, max_Level);

  SHARED DATASET(NumericField) ComputeBetas(DATASET(NumericField) indeps, DATASET(NumericField) deps) := FUNCTION
    return baseTree.LearnC(indeps, deps);
  END;

  SHARED DATASET(NumericField) PredictLocal(DATASET(NumericField) indeps, DATASET(NumericField) betas) := FUNCTION
    return ML.DecisionTree.Regression.Predict.PredictC(indeps, betas);
  END;

END;
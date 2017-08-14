IMPORT ML;
IMPORT ML.Types as Types;

EXPORT GBTypes := MODULE

  // General Gradient Boosting Regression
  // Record used to store new dependents and betas
  EXPORT GBRegressionRecord := RECORD
    Types.t_FieldNumber id;
    Types.t_FieldNumber iteration;
    BOOLEAN isBeta; // TRUE indicates weight, FALSE indicates a dependent value.
    Types.t_FieldNumber number; // Index of weight/dependent value.
    Types.t_FieldReal value; // Value stored in record
  END;

  // General Gradient Boosting Regression Record
  // used to store new dependents and betas
  EXPORT GBClassificationRecord := RECORD(GBRegressionRecord)
    Types.t_FieldNumber classifier_ID;
  END;

  // Record used to store the Class Metadata
  EXPORT ValueRecord := RECORD
    Types.t_FieldNumber id;
    Types.t_FieldReal value;
  END;

  // Record used to store the Depenedent Variables
  EXPORT DependentClassifierRecord := RECORD(Types.NumericField)
    Types.t_FieldNumber classifier_ID;
  END;

  // Extension of NumericField adding a label.
  EXPORT LabeledNumericField := RECORD(Types.NumericField)
    Types.t_FieldNumber label;
  END;

END;

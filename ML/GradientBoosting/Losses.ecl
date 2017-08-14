IMPORT ML;
IMPORT ML.Types as Types;

EXPORT Losses := MODULE

  EXPORT SquareLoss(Types.t_FieldReal expected, Types.t_FieldReal predicted) := FUNCTION
    RETURN POWER(expected - predicted, 2) / 2;
  END;

  EXPORT HuberLoss(Types.t_FieldReal expected, Types.t_FieldReal predicted, REAL delta=0.5) := FUNCTION
    diff := ABS(expected - predicted);
    RETURN IF(diff <= delta, 0.5 * diff * diff, delta * (diff - delta / 2));
  END;

  EXPORT AbsLoss(Types.t_FieldReal expected, Types.t_FieldReal predicted) := FUNCTION
    RETURN ABS(expected - predicted);
  END;

  EXPORT LossRecord := RECORD
    Types.t_RecordID id;
    Types.t_FieldReal expected;
    Types.t_FieldReal predicted;
    Types.t_FieldReal loss;
  END;

  EXPORT LossRecord EstimateError(ML.Types.NumericField expected, ML.Types.NumericField predicted, REAL delta=0.5) := TRANSFORM
    SELF.id := expected.id;
    SELF.expected := expected.value;
    SELF.predicted := predicted.value;
    SELF.loss := HuberLoss(expected.value, predicted.value, delta);
    //SELF.loss := AbsLoss(expected.value, predicted.value);
  END;
END;
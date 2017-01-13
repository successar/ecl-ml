IMPORT ML;
IMPORT ML.Types AS Types;
IMPORT ML.Utils AS Utils;
IMPORT ML.Classify AS Classify;
IMPORT ML.StepwiseLogistic.TypesSL AS TypesSL;
IMPORT Std.Str;

EXPORT StepLogistic(REAL8 Ridge=0.00001, REAL8 Epsilon=0.000000001, UNSIGNED2 MaxIter=200) 
                                          := MODULE(Classify.Logistic_sparse(Ridge,Epsilon,MaxIter))
																				
  SHARED Parameter := TypesSL.Parameter;
  SHARED ParamRec := TypesSL.ParamRec;
  SHARED StepRec := TypesSL.StepRec;

  SHARED RebaseX(DATASET(Types.NumericField) X) := FUNCTION
    RebaseX := Utils.RebaseNumericField(X);
    X_Map := RebaseX.Mapping(1);
    RETURN RebaseX.ToNew(X_Map);
  END;

  EXPORT ExtractX(DATASET(Types.NumericField) X, DATASET(Parameter) M) := FUNCTION
    x_subset := X(number in SET(M, number));
    RETURN RebaseX(x_subset);
  END;
	
END;
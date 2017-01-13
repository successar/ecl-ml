IMPORT ML;
IMPORT ML.Types AS Types;
IMPORT ML.Utils AS Utils;
IMPORT ML.Classify AS Classify;
IMPORT ML.StepwiseLogistic.TypesSL AS TypesSL;
IMPORT Std.Str;

EXPORT Backward(REAL8 Ridge=0.00001, REAL8 Epsilon=0.000000001, UNSIGNED2 MaxIter=200) 
                                  := MODULE(ML.StepwiseLogistic.StepLogistic(Ridge,Epsilon,MaxIter))
  EXPORT Regression(DATASET(Types.NumericField) X,DATASET(Types.DiscreteField) Y) := MODULE

    SHARED DATASET(Parameter) Indices := NORMALIZE(DATASET([{0}], Parameter), COUNT(ML.FieldAggregates(X).Cardinality), 
                                                      TRANSFORM(Parameter, SELF.number := COUNTER));
    InitMod := LearnCS(X, Y);
    AIC := DevianceC(X, Y, InitMod).AIC[1].AIC;

    SHARED DATASET(StepRec) InitialStep := DATASET([{DATASET([], Parameter), DATASET([], ParamRec), Indices, AIC}], StepRec);

    DATASET(StepRec) Step_Backward(DATASET(StepRec) recs, INTEGER c) := FUNCTION

      le := recs[c];			
      Selected := SET(le.Final, number);

      SelectList := Indices(number IN Selected) + DATASET([{0}], Parameter);			
      NumSelect := COUNT(SelectList);

      DATASET(ParamRec) T_Select(DATASET(ParamRec) precs, INTEGER paramNum) := FUNCTION
        x_subset := X(number IN Selected AND number NOT IN [paramNum]);
        X_0 := RebaseX(x_subset);
        reg := LearnCS(X_0, Y);
        Dev := DevianceC(IF(EXISTS(X_0), X_0, X), Y, reg);
        AIC := Dev.AIC[1].AIC;
        Deviance := Dev.ResidDev[1].Deviance;
        Op := '-';
        RETURN precs + ROW({Op, paramNum, Deviance, AIC}, ParamRec);
      END;		

      SelectCalculated := LOOP(DATASET([], ParamRec), COUNTER <= NumSelect, T_Select(ROWS(LEFT), SelectList[COUNTER].number));
      bestSR := TOPN(SelectCalculated, 1, AIC);			

      Initial := le.Final;
      StepRecs := SelectCalculated;
      Final := Indices(number IN Selected AND number NOT IN [bestSR[1].ParamNum]);
      AIC := bestSR[1].AIC;

      RETURN recs + ROW({Initial, StepRecs, Final, AIC}, Steprec);
    END;

    EXPORT DATASET(StepRec) Steps := LOOP(InitialStep, 
                              COUNTER = 1 OR ROWS(LEFT)[COUNTER].Initial != ROWS(LEFT)[COUNTER].Final,
                              Step_Backward(ROWS(LEFT), COUNTER));

    SHARED BestStep := Steps[COUNT(Steps)];
    var_subset := SET(BestStep.Final, number);
    x_subset := X(number IN var_subset);
    X_0 := RebaseX(x_subset);
    EXPORT BestModel := LearnCS(X_0, Y);
    EXPORT MapX := BestStep.Final;
  END;

END;
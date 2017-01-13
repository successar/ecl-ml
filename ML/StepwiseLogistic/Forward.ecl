IMPORT ML;
IMPORT ML.Types AS Types;
IMPORT ML.Utils AS Utils;
IMPORT ML.Classify AS Classify;
IMPORT ML.StepwiseLogistic.TypesSL AS TypesSL;
IMPORT Std.Str;

EXPORT Forward(REAL8 Ridge=0.00001, REAL8 Epsilon=0.000000001, UNSIGNED2 MaxIter=200) 
                                                      := MODULE(ML.StepwiseLogistic.StepLogistic(Ridge,Epsilon,MaxIter))

  EXPORT Regression(DATASET(Types.NumericField) X,DATASET(Types.DiscreteField) Y) := MODULE

    SHARED DATASET(Parameter) Indices := NORMALIZE(DATASET([{0}], Parameter), COUNT(ML.FieldAggregates(X).Cardinality), 
                                                TRANSFORM(Parameter, SELF.number := COUNTER));
    X_0 := DATASET([], Types.NumericField);
    InitMod := LearnCS(X_0, Y);
    AIC := DevianceC(IF(EXISTS(X_0), X_0, X), Y, InitMod).AIC[1].AIC;

    SHARED DATASET(StepRec) InitialStep := DATASET([{DATASET([], Parameter), DATASET([], ParamRec), DATASET([], Parameter), AIC}], StepRec);

    DATASET(StepRec) Step_Forward(DATASET(StepRec) recs, INTEGER c) := FUNCTION

      le := recs[c];			
      Selected := SET(le.Final, number);

      NotChosen := Indices(number NOT IN Selected) + DATASET([{0}], Parameter);
      NumChosen := COUNT(NotChosen);

      DATASET(ParamRec) T_Choose(DATASET(ParamRec) precs, INTEGER paramNum) := FUNCTION
        x_subset := X(number IN (Selected + [paramNum]));
        X_0 := RebaseX(x_subset);
        reg := LearnCS(X_0, Y);
        Dev := DevianceC(IF(EXISTS(X_0), X_0, X), Y, reg);
        AIC := Dev.AIC[1].AIC;
        Deviance := Dev.ResidDev[1].Deviance;
        Op := '+';
        RETURN precs + ROW({Op, paramNum, Deviance, AIC}, ParamRec);
      END;		

      ChooseCalculated := LOOP(DATASET([], ParamRec), COUNTER <= NumChosen, T_Choose(ROWS(LEFT), NotChosen[COUNTER].number));
      bestCR := TOPN(ChooseCalculated, 1, AIC);			

      Initial := le.Final;
      StepRecs := ChooseCalculated;
      Final := Indices(number IN Selected OR number IN [bestCR[1].ParamNum]);
      AIC := bestCR[1].AIC;

      RETURN recs + ROW({Initial, StepRecs, Final, AIC}, Steprec);
    END;

      
    EXPORT DATASET(StepRec) Steps := LOOP(InitialStep, 
                    COUNTER = 1 OR ROWS(LEFT)[COUNTER].Initial != ROWS(LEFT)[COUNTER].Final,
                    Step_Forward(ROWS(LEFT), COUNTER));

    SHARED BestStep := Steps[COUNT(Steps)];
    var_subset := SET(BestStep.Final, number);
    x_subset := X(number IN var_subset);
    X_0 := RebaseX(x_subset);
    EXPORT BestModel := LearnCS(X_0, Y);
    EXPORT MapX := BestStep.Final;
  END;	


END;
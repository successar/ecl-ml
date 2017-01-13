IMPORT ML;
IMPORT ML.StepRegression AS Step;
IMPORT ML.Types AS Types;
IMPORT ML.Utils AS Utils;
IMPORT Std.Str;
OLS2Use := ML.Regression.Sparse.OLS_LU;

EXPORT ForwardRegression(DATASET(Types.NumericField) X,
			DATASET(Types.NumericField) Y) := MODULE(Step.StepRegression(X,Y))
											

	AIC := OLS2Use(X(number IN [0]), Y).AIC[1].AIC;
	SHARED DATASET(StepRec) InitialStep := DATASET([{DATASET([], Parameter), DATASET([], ParamRec), DATASET([], Parameter), AIC}], StepRec);
		
	DATASET(StepRec) Step_Forward(DATASET(StepRec) recs, INTEGER c) := FUNCTION
	
		le := recs[c];			
		Selected := SET(le.Final, number);
					
		NotChosen := Indices(number NOT IN Selected) + DATASET([{0}], Parameter);
		NumChosen := COUNT(NotChosen);
		 
		DATASET(ParamRec) T_Choose(DATASET(ParamRec) precs, INTEGER paramNum) := FUNCTION
			x_subset := X(number IN (Selected + [paramNum]));
			reg := OLS2Use(x_subset, Y);
			RSS := (reg.Anova)[1].Error_SS;
			AIC := (reg.AIC)[1].AIC;
			Op := '+';
			RETURN precs + ROW({Op, paramNum, RSS, AIC}, ParamRec);
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
	
END;
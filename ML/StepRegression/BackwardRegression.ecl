IMPORT ML;
IMPORT ML.StepRegression AS Step;
IMPORT ML.Types AS Types;
IMPORT ML.Utils AS Utils;
IMPORT Std.Str;
OLS2Use := ML.Regression.Sparse.OLS_LU;

EXPORT BackwardRegression(DATASET(Types.NumericField) X,
			DATASET(Types.NumericField) Y) := MODULE(Step.StepRegression(X,Y))
											

	AIC := OLS2Use(X, Y).AIC[1].AIC;
	SHARED DATASET(StepRec) InitialStep := DATASET([{DATASET([], Parameter), DATASET([], ParamRec), Indices, AIC}], StepRec);
		
	DATASET(StepRec) Step_Backward(DATASET(StepRec) recs, INTEGER c) := FUNCTION
	
		le := recs[c];			
		Selected := SET(le.Final, number);
		
		SelectList := Indices(number IN Selected) + DATASET([{0}], Parameter);			
		NumSelect := COUNT(SelectList);
		
		DATASET(ParamRec) T_Select(DATASET(ParamRec) precs, INTEGER paramNum) := FUNCTION
			x_subset := X(number IN Selected AND number NOT IN [ParamNum]);
			reg := OLS2Use(x_subset, Y);
			RSS := (reg.Anova)[1].Error_SS;
			AIC := (reg.AIC)[1].AIC;
			Op := '-';
			RETURN precs + ROW({Op, paramNum, RSS, AIC}, ParamRec);
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
	
END;
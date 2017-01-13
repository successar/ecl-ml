IMPORT ML;
IMPORT ML.StepRegression AS Step;
IMPORT ML.Types AS Types;
IMPORT ML.Utils AS Utils;
IMPORT Std.Str;
OLS2Use := ML.Regression.Sparse.OLS_LU;

EXPORT BidirecRegression(DATASET(Types.NumericField) X,
			DATASET(Types.NumericField) Y,
			DATASET({UNSIGNED4 number}) InputVars) := MODULE(Step.StepRegression(X,Y))
											
	AIC := OLS2Use(X(number IN SET(InputVars, number)), Y).AIC[1].AIC;
	SHARED DATASET(StepRec) InitialStep := DATASET([{DATASET([], Parameter), DATASET([], ParamRec), InputVars, AIC}], StepRec);
		
	DATASET(StepRec) Step_Bidirec(DATASET(StepRec) recs, INTEGER c) := FUNCTION
	
		le := recs[c];			
		Selected := SET(le.Final, number);
			
		SelectList := Indices(number IN Selected);			
		NotChosen := Indices(number NOT IN Selected) + DATASET([{0}], Parameter);
		
		NumChosen := COUNT(NotChosen);
		NumSelect := COUNT(SelectList);
		 
		DATASET(ParamRec) T_Choose(DATASET(ParamRec) precs, INTEGER paramNum) := FUNCTION
			x_subset := X(number IN (Selected + [paramNum]));
			reg := OLS2Use(x_subset, Y);
			RSS := (reg.Anova)[1].Error_SS;
			AIC := (reg.AIC)[1].AIC;
			Op := '+';
			RETURN precs + ROW({Op, paramNum, RSS, AIC}, ParamRec);
		END;		
		
		DATASET(ParamRec) T_Select(DATASET(ParamRec) precs, INTEGER paramNum) := FUNCTION
			x_subset := X(number IN Selected AND number NOT IN [ParamNum]);
			reg := OLS2Use(x_subset, Y);
			RSS := (reg.Anova)[1].Error_SS;
			AIC := (reg.AIC)[1].AIC;
			Op := '-';
			RETURN precs + ROW({Op, paramNum, RSS, AIC}, ParamRec);
		END;			
		
		ChooseCalculated := LOOP(DATASET([], ParamRec), COUNTER <= NumChosen, T_Choose(ROWS(LEFT), NotChosen[COUNTER].number));
		bestCR := TOPN(ChooseCalculated, 1, AIC)[1];			
		SelectCalculated := LOOP(DATASET([], ParamRec), COUNTER <= NumSelect, T_Select(ROWS(LEFT), SelectList[COUNTER].number));
		bestSR := TOPN(SelectCalculated, 1, AIC)[1];
		
		Initial := le.Final;
		StepRecs := SelectCalculated + ChooseCalculated;
		Final := IF(bestSR.AIC < bestCR.AIC, Indices(number IN Selected AND number NOT IN [bestSR.ParamNum]),
						Indices(number IN Selected OR number IN [bestCR.ParamNum]));
		AIC := IF(bestSR.AIC < bestCR.AIC, bestSR.AIC, bestCR.AIC);
		
		RETURN recs + ROW({Initial, StepRecs, Final, AIC}, Steprec);
	END;

	EXPORT DATASET(StepRec) Steps := LOOP(InitialStep, 
						COUNTER = 1 OR ROWS(LEFT)[COUNTER].Initial != ROWS(LEFT)[COUNTER].Final,
						Step_Bidirec(ROWS(LEFT), COUNTER));
	
END;
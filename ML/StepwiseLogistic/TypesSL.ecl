IMPORT ML;

EXPORT TypesSL := MODULE
EXPORT Parameter := RECORD
		UNSIGNED4 number;
END;

EXPORT ParamRec := RECORD
	STRING Op := '+';
	UNSIGNED1 ParamNum;
	REAL Deviance := 0;
	REAL AIC := 0;
END;

EXPORT StepRec := RECORD
		DATASET(Parameter) Initial;
		DATASET(ParamRec) ParamSteps;
		DATASET(Parameter) Final;
		REAL AIC := 0;
END;

END;
	
/*
Perform Forward Stepwise Regression
*/

IMPORT ML;
IMPORT ML.Types AS Types;
IMPORT ML.Utils AS Utils;
IMPORT Std.Str;
OLS2Use := ML.Regression.Sparse.OLS_LU;

EXPORT StepRegression(DATASET(Types.NumericField) X,
                      DATASET(Types.NumericField) Y) := MODULE, VIRTUAL
	
	//Numeric Index of Parameter
	SHARED Parameter := RECORD
		UNSIGNED4 number;
	END;
	
	//Numeric Index of all the Parameters in X
	SHARED DATASET(Parameter) Indices := NORMALIZE(DATASET([{0}], Parameter), COUNT(ML.FieldAggregates(X).Cardinality), 
							TRANSFORM(Parameter, SELF.number := COUNTER));
	
	//Record for Each Parameter tested at each Step
	SHARED ParamRec := RECORD
		//Denotes if Parameter was added or removed
		STRING Op := '+';
		//Parameters's Numeric Index
		UNSIGNED1 ParamNum;
		//AIC and RSS obtained after adding this Parameter
		REAL RSS := 0;
		REAL AIC := 0;
	END;
	
	//Record for Each Step Taken
	EXPORT StepRec := RECORD
		//Parameters in model before in this Step
		DATASET(Parameter) Initial;
		//Records of Parameters Tested
		DATASET(ParamRec) ParamSteps;
		//Selected Parameters at end of this Step
		DATASET(Parameter) Final;
		//Best AIC obtained at End of this Step
		REAL AIC := 0;
	END;
	
	//Dataset of All Steps Taken
	EXPORT DATASET(StepRec) Steps;
	
	//Choose best Model among all Steps
	BestStep := Steps[COUNT(Steps)];
	var_subset := SET(BestStep.Final, number);
	x_subset := X(number IN var_subset);
	EXPORT BestModel := OLS2Use(x_subset, Y);
	
END;

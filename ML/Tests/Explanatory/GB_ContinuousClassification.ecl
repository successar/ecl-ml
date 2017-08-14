IMPORT ML;
IMPORT continious_ecoliDS FROM TestingSuite.Classification.Datasets;
IMPORT ML.GradientBoosting as GB;
IMPORT ML.Types as Types;

NumericField := Types.NumericField;
DiscreteField := Types.DiscreteField;
L_Result := Types.l_result;
Classification := GB.Classification.Logistic;
Tree := GB.Classification.ContinuousTree;
LabeledNumericField := GB.GBTypes.LabeledNumericField;

// Dataset
content := continious_ecoliDS.content;

// Independent and Dependent Variables
ML.ToField(content, fielded);

class_index := COUNT(fielded)/COUNT(content);
indep := fielded(number < class_index);
dep := fielded(number = class_index);

// Function to convert to DiscreteField
DATASET(DiscreteField) toDiscrete(DATASET(NumericField) predicteds) := FUNCTION
	RETURN PROJECT(predicteds, TRANSFORM(DiscreteField, SELF.id:=LEFT.id, SELF.number:=LEFT.number, SELF.value:=LEFT.value));
END;

// Function to convert to l_result
DATASET(L_Result) toLResult(DATASET(LabeledNumericField) predicteds) := FUNCTION
	RETURN PROJECT(predicteds, TRANSFORM(L_Result, SELF.id:=LEFT.id, SELF.number:=LEFT.number, SELF.value:=LEFT.label, SELF.conf:=LEFT.value));
END;


base_model := Classification(indep, dep, 1, TRUE);
base_betas := base_model.Learn();
base_pred := SORT(base_model.Predict(indep, base_betas), id);
base_stats := ML.Classify.Compare(toDiscrete(dep), toLResult(base_pred));
OUTPUT(base_stats.RecallByClass, NAMED('Base_Recall'), ALL);
OUTPUT(base_stats.PrecisionByClass, NAMED('Base_Precision'), ALL);
OUTPUT(base_stats.FP_Rate_ByClass, NAMED('Base_False_Positive'), ALL);
OUTPUT(base_stats.Accuracy, NAMED('Base_Accuracy'), ALL);


grad_model := Classification(indep, dep, 5, TRUE);
grad_betas := grad_model.Learn();
grad_pred := SORT(grad_model.Predict(indep, grad_betas), id);
grad_stats := ML.Classify.Compare(toDiscrete(dep), toLResult(grad_pred));
OUTPUT(grad_stats.RecallByClass, NAMED('Gradient_Recall'), ALL);
OUTPUT(grad_stats.PrecisionByClass, NAMED('Gradient_Precision'), ALL);
OUTPUT(grad_stats.FP_Rate_ByClass, NAMED('Gradient_False_Positive'), ALL);
OUTPUT(grad_stats.Accuracy, NAMED('Gradient_Accuracy'), ALL);

grad_tree_model := Tree(indep, dep, iterations:=5, doNormalize:=TRUE);
grad_tree_betas := grad_tree_model.Learn();
grad_tree_pred := SORT(grad_tree_model.Predict(indep, grad_tree_betas), id);
grad_tree_stats := ML.Classify.Compare(toDiscrete(dep), toLResult(grad_tree_pred));
OUTPUT(grad_tree_stats.RecallByClass, NAMED('Gradient_Tree_Recall'), ALL);
OUTPUT(grad_tree_stats.PrecisionByClass, NAMED('Gradient_Tree_Precision'), ALL);
OUTPUT(grad_tree_stats.FP_Rate_ByClass, NAMED('Gradient_Tree_False_Positive'), ALL);
OUTPUT(grad_tree_stats.Accuracy, NAMED('Gradient_Tree_Accuracy'), ALL);

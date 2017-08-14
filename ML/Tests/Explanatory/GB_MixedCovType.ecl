IMPORT ML;
IMPORT mixed_covTypeDS FROM TestingSuite.Classification.Datasets;
IMPORT ML.GradientBoosting as GB;
IMPORT ML.Types as Types;

NumericField := Types.NumericField;
DiscreteField := Types.DiscreteField;
L_Result := Types.l_result;
Classification := GB.Classification.Logistic;
Tree := GB.Classification.MixedTree;
// Tree := GB.Classification.ContinuousTree;
LabeledNumericField := GB.GBTypes.LabeledNumericField;
FieldType := ML.DecisionTree.Utils.FieldType;

// Datasets
train := mixed_covTypeDS.trainRecs;
test := mixed_covTypeDS.testRecs;

// Training Independent and Dependent Variables
ML.ToField(train, train_fielded);
train_class_index := COUNT(train_fielded)/COUNT(train);
train_indep := train_fielded(number < train_class_index);
train_dep := train_fielded(number = train_class_index);
f_types := ML.DecisionTree.Utils.GetFieldTypes(train_indep, DATASET([
        {1, True},{2, True},
        {3, True},{4, True},
        {5, True},{6, True},
        {7, True},{8, True},
        {9, True},{10, True}
  ], FieldType), is_cont_default:=FALSE);


// Testing Independent and Dependent Variables
ML.ToField(test, test_fielded);
test_class_index := COUNT(test_fielded)/COUNT(test);
test_indep := test_fielded(number < test_class_index);
test_dep := test_fielded(number = test_class_index);


// Function to convert to DiscreteField
DATASET(DiscreteField) toDiscrete(DATASET(NumericField) predicteds) := FUNCTION
	RETURN PROJECT(predicteds, TRANSFORM(DiscreteField, SELF.id:=LEFT.id, SELF.number:=LEFT.number, SELF.value:=LEFT.value));
END;

// Function to convert to l_result
DATASET(L_Result) toLResult(DATASET(LabeledNumericField) predicteds) := FUNCTION
	RETURN PROJECT(predicteds, TRANSFORM(L_Result, SELF.id:=LEFT.id, SELF.number:=LEFT.number, SELF.value:=LEFT.label, SELF.conf:=LEFT.value));
END;

grad_tree_model := Tree(train_indep, train_dep, f_types, max_level:=3, iterations:=5, doNormalize:=TRUE);
grad_tree_betas := grad_tree_model.Learn();
grad_tree_pred := SORT(grad_tree_model.Predict(test_indep, grad_tree_betas), id);
grad_tree_stats := ML.Classify.Compare(toDiscrete(test_dep), toLResult(grad_tree_pred));
OUTPUT(grad_tree_stats.RecallByClass, NAMED('Gradient_Tree_Recall'), ALL);
OUTPUT(grad_tree_stats.PrecisionByClass, NAMED('Gradient_Tree_Precision'), ALL);
OUTPUT(grad_tree_stats.FP_Rate_ByClass, NAMED('Gradient_Tree_False_Positive'), ALL);
OUTPUT(grad_tree_stats.Accuracy, NAMED('Gradient_Tree_Accuracy'), ALL);

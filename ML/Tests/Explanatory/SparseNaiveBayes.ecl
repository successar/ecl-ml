IMPORT ML;
IMPORT ML.Utils AS Utils;

// The train dataset has 80,000 instances x 109,735 attributes 1 class attribute.
// The dataset is represented using Weka Sparse ARFF format, we assume default value as "0"
// You can download the ARFF file from:
// https://www.dropbox.com/s/3ss8ibhi9ftvvn5/sparsearfffile.arff?dl=0

TrainDS   := Utils.SparseARFFfileToDiscreteFieldCounted('~vherrara::datasets::sparsearfffile.arff');
indepData := TrainDS(Number<109736);
depData   := TrainDS(Number=109736);
//SparseNaiveBayes classifier
trainer   := ML.Classify.SparseNaiveBayes(TRUE, 0); // IgnoreMissing = TRUE and defValue = 0
// Learning Phase
D_Model   := trainer.LearnD(indepData, depData);
dmodel    := trainer.ModelD(D_model);
// Classification Phase
D_classDist := trainer.ClassProbDistribD(indepData, D_Model);           // Class Probalility Distribution
D_results   := trainer.ClassifyD(indepData, D_Model);                   // Classification results
// Performance Metrics
D_compare   := ML.Classify.Compare(depData, D_results);                 // Comparing results with original class
AUC_D0      := SORT(ML.Classify.AUC_ROC(D_ClassDist, 0, depData), -id); //Area under ROC Curve for class "0"
AUC_D4      := SORT(ML.Classify.AUC_ROC(D_ClassDist, 4, depData), -id); //Area under ROC Curve for class "1"
// OUPUTS
OUTPUT(indepData, NAMED('indepData'));
OUTPUT(depData  , NAMED('depData'));
OUTPUT(SORT(dmodel, id), ALL, NAMED('DiscModel'));
OUTPUT(D_classDist, ALL, NAMED('DisClassDist'));
OUTPUT(D_results, NAMED('DiscClassifResults'), ALL);
OUTPUT(SORT(D_compare.CrossAssignments, c_actual, c_modeled), NAMED('DiscCrossAssig'), ALL); // Confusion Matrix
OUTPUT(AUC_D0, ALL, NAMED('AUC_0'));
OUTPUT(AUC_D4, ALL, NAMED('AUC_4'));
OUTPUT(D_compare.RecallByClass, NAMED('RecallByClassD'));
OUTPUT(D_compare.PrecisionByClass, NAMED('PrecByClassD'));
OUTPUT(SORT(D_compare.FP_Rate_ByClass, classifier, c_modeled), NAMED('FPR_ByClassD'));
OUTPUT(D_compare.Accuracy, NAMED('AccuracyD'));
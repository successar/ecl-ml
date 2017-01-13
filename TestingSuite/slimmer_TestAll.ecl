IMPORT Std;
IMPORT TS;
IMPORT ML;
IMPORT ML.Types;
IMPORT TestingSuite;
IMPORT TestingSuite.BenchmarkResults AS BenchmarkResults;
IMPORT TestingSuite.Utils AS Utils;
IMPORT TestingSuite.Classification as Classification;
IMPORT TestingSuite.Clustering as Clustering;
IMPORT TestingSuite.Regression as Regression;

dataset_record := RECORD
	INTEGER dataset_id;
	STRING dataset_name;
	REAL ecl_performance;
        REAL scikit_learn_performance;
END;

QualifiedName(prefix, datasetname):=  FUNCTIONMACRO
        RETURN prefix + datasetname + '.content';
ENDMACRO;

SET OF STRING classificationDatasetNamesD := ['discrete_houseVoteDS'];

// For Testing KMeans        
SET OF STRING clusteringDatasetNames := ['ionek_f_two_c_twoDS' ];    
SET OF INTEGER ClusterNumbers := [2]; 

SET OF STRING classificationDatasetNamesC := ['continious_ecoliDS'];                                                  
SET OF STRING regressionDatasetNames := ['AbaloneDS'];   


timeseriesDatasetNames := ['default'];
ts_no_of_elements := COUNT(timeseriesDatasetNames);

INTEGER c_no_of_elementsD := COUNT(classificationDatasetNamesD);
INTEGER c_no_of_elementsC := COUNT(classificationDatasetNamesC);
INTEGER cluster_no_of_elements := COUNT(clusteringDatasetNames);
INTEGER r_no_of_elements := COUNT(regressionDatasetNames);

SEQUENTIAL(
        OUTPUT(Utils.GenerateCode('Classification.TestRandomForestClassificationC',  classificationDatasetNamesC, BenchmarkResults.rfc_performance_scores_c, c_no_of_elementsC, 1), NAMED('Classification_RandomForestC')),
        OUTPUT(Utils.GenerateCode('Classification.TestRandomForestClassificationD',  classificationDatasetNamesD, BenchmarkResults.rfc_performance_scores_d, c_no_of_elementsD, 1), NAMED('Classification_RandomForestD')),
        OUTPUT(Utils.GenerateCode('Classification.TestDecisionTreeClassifier',  classificationDatasetNamesD, BenchmarkResults.dtc_performance_scores, c_no_of_elementsD, 1), NAMED('Classification_DecisionTree')),
        OUTPUT(Utils.GenerateCode_K('Clustering.TestKmeans', clusteringDatasetNames, ClusterNumbers, BenchmarkResults.kmeans_performance_scores, 1), NAMED('Clustering_KMeans')),
        OUTPUT(Utils.GenerateCode_R('Regression.TestLinearRegression', regressionDatasetNames), NAMED('Regression_LR')),
        OUTPUT(Utils.GenerateCode('Classification.TestLogisticRegression',  classificationDatasetNamesD, BenchmarkResults.logistic_regression_performance_scores, c_no_of_elementsD, 1), NAMED('Classification_LogisticRegression'));
);



                                                                              
                                        

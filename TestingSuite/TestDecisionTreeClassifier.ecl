IMPORT Std;
IMPORT TS;
IMPORT ML;
IMPORT ML.Types;
IMPORT TestingSuite;
IMPORT TestingSuite.BenchmarkResults AS BenchmarkResults;
IMPORT TestingSuite.Utils AS Utils;
IMPORT TestingSuite.Classification as Classification;
IMPORT TestingSuite.BenchmarkResults AS BenchmarkResults;


QualifiedName(prefix, datasetname):=  FUNCTIONMACRO
        RETURN prefix + datasetname + '.content';
ENDMACRO;


SET OF STRING classificationDatasetNames := ['discrete_GermanDS', 'discrete_houseVoteDS',
        'discrete_letterrecognitionDS','discrete_liverDS', 'discrete_satimagesDS',
        'discrete_soybeanDS', 'discrete_VehicleDS']; 
       
INTEGER c_no_of_elements := COUNT(classificationDatasetNames);

dt_results := Utils.GenerateCode('Classification.TestDecisionTreeClassifier',  classificationDatasetNames, BenchmarkResults.dtc_performance_scores, c_no_of_elements, 10);

OUTPUT(dt_results, NAMED('Classification_DecisionTree'));



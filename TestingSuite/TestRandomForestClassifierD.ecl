    IMPORT Std;
IMPORT TS;
IMPORT ML;
IMPORT ML.Types;
IMPORT TestingSuite;
IMPORT TestingSuite.BenchmarkResults AS BenchmarkResults;
IMPORT TestingSuite.Utils;
IMPORT TestingSuite.Classification as Classification;

QualifiedName(prefix, datasetname):=  FUNCTIONMACRO
        RETURN prefix + datasetname + '.content';
ENDMACRO;


SET OF STRING classificationDatasetNamesD := ['discrete_GermanDS',
        , 'discrete_houseVoteDS',
        'discrete_letterrecognitionDS','discrete_liverDS', 'discrete_satimagesDS',
        'discrete_soybeanDS', 'discrete_VehicleDS'];  
                                               

INTEGER c_no_of_elements := COUNT(classificationDatasetNamesD);

rf_results := Utils.GenerateCode('Classification.TestRandomForestClassificationD',  classificationDatasetNamesD, BenchmarkResults.rfc_performance_scores_d, c_no_of_elements);

OUTPUT(rf_results, NAMED('Classification_RandomForestD'));



IMPORT Std;
IMPORT TS;
IMPORT ML;
IMPORT ML.Types;
IMPORT TestingSuite;
IMPORT TestingSuite.BenchmarkResults AS BenchmarkResults;
IMPORT TestingSuite.Utils AS Utils;
IMPORT TestingSuite.Regression as Regression;
IMPORT TestingSuite.BenchmarkResults AS BenchmarkResults;

QualifiedName(prefix, datasetname):=  FUNCTIONMACRO
        RETURN prefix + datasetname + '.content';
ENDMACRO;

SET OF STRING regressionDatasetNames := ['AbaloneDS', 'friedman1DS', 'friedman2DS', 'friedman3DS', 'housingDS', 'servoDS'];                                                                                

INTEGER r_no_of_elements := COUNT(regressionDatasetNames);

lr_results := Utils.GenerateCode_R('Regression.TestLinearRegression', regressionDatasetNames);

OUTPUT(lr_results, NAMED('Regression_LR'));


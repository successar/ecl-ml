IMPORT Std;
IMPORT TS;
IMPORT ML;
IMPORT ML.Types;
IMPORT TestingSuite;
IMPORT TestingSuite.BenchmarkResults AS BenchmarkResults;
IMPORT TestingSuite.Utils;
IMPORT TestingSuite.Clustering as Clustering;


QualifiedName(prefix, datasetname):=  FUNCTIONMACRO
        RETURN prefix + datasetname + '.content';
ENDMACRO;

// For Testing KMeans        
SET OF STRING clusteringDatasetNames := ['ionek_f_eight_c_sixDS',  
                                        'ionek_f_four_c_fourDS', 
                                        'ionek_f_sixteen_c_eightDS', 
                                        'ionek_f_thirty_two_c_eightDS', 
                                        'ionek_f_two_c_twoDS' ];    
SET OF INTEGER ClusterNumbers := [6, 4, 8, 8, 2]; 

INTEGER cluster_no_of_elements := COUNT(clusteringDatasetNames);

kmeans_results := Utils.GenerateCode_K('Clustering.TestKmeans', clusteringDatasetNames, ClusterNumbers, BenchmarkResults.kmeans_performance_scores, 10);

OUTPUT(kmeans_results, NAMED('Clustering_KMeans'));



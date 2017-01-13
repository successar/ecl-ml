EXPORT GenerateCode_K(algorithm, datasetNames, clusterNumbers, performance_scores, repeat_no=1):= FUNCTIONMACRO
        #DECLARE(source_code)
        #SET(source_code, '');
        #DECLARE(indexs);
        #SET(indexs, 1);
        

        #LOOP
        	#IF(%indexs%> cluster_no_of_elements)	
        		#BREAK
        	#ELSE
                        #APPEND(source_code, 'dataset_' + datasetNames[%indexs%] + ' := ' + algorithm + '(' + QualifiedName('Clustering.Datasets.', datasetNames[%indexs%]));
                        #APPEND(source_code,','+ repeat_no +',' + clusterNumbers[%indexs%] + ');\n');
                        #SET(indexs,%indexs%+1);
                #END
        #END
        
        #APPEND(source_code, 'final_results := DATASET([');
        #SET(indexs, 1);
        #LOOP
        	#IF(%indexs%>cluster_no_of_elements)	
        		#BREAK
        	#ELSE
                        #APPEND(source_code, '{' + %indexs% + ',\'' + datasetNames[%indexs%] + '\', AVE(dataset_' + datasetNames[%indexs%] + ', scores),' + 'AVE(' + #TEXT(performance_scores) + '(dataset_id=' + %indexs% + '),'+ #TEXT(performance_scores)+'.scores),' + 'TestingSuite.Utils.Student_T_Test(dataset_' + datasetNames[%indexs%]+ ','+ #TEXT(performance_scores) + '(dataset_id=' + %indexs% + '))}');
                        #IF(%indexs%<cluster_no_of_elements)
                                #APPEND(source_code, ',\n');
                        #ELSE
                                #APPEND(source_code, '\n');
                        #END
                        #SET(indexs,%indexs%+1);
                #END
        #END
        #APPEND(source_code, '], TestingSuite.Utils.Types.dataset_record);\n');
        
        %source_code%;
        RETURN final_results;
        
ENDMACRO;   

/*
QualifiedName(prefix, datasetname):=  FUNCTIONMACRO
        RETURN prefix + datasetname + '.content';
ENDMACRO;

SET OF STRING clusteringDatasetNames := ['ionek_f_eight_c_sixDS'];   
SET OF INTEGER ClusterNumbers := [6, 2, 4, 8, 8, 2, 8];  
SET OF REAL km_performance_scores := [1, 1, 1, 1, 1, 1, 1, 1]; 
INTEGER cluster_no_of_elements := COUNT(clusteringDatasetNames); 
GenerateCode_K('Clustering.TestKmeans', clusteringDatasetNames, ClusterNumbers, km_performance_scores)
*/

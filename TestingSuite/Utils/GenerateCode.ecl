EXPORT GenerateCode(algorithm, datasetNames, performance_scores, c_no_of_elements, repeat_no=1):= FUNCTIONMACRO
        #DECLARE(source_code)
        #SET(source_code, '');
        #DECLARE(indexs);
        #SET(indexs, 1);
        #LOOP
        	#IF(%indexs%> c_no_of_elements)	
        		#BREAK
        	#ELSE
                        #APPEND(source_code, 'dataset_' + datasetNames[%indexs%] + ' := ' + algorithm + '(' + QualifiedName('Classification.Datasets.', datasetNames[%indexs%]) + ','+ repeat_no + ');\n');
                        #SET(indexs,%indexs%+1);
                #END
        #END
        
        #APPEND(source_code, 'final_results := DATASET([');
        #SET(indexs, 1);
        #LOOP
        	#IF(%indexs%>c_no_of_elements)	
        		#BREAK
        	#ELSE
                        #APPEND(source_code, '{' + %indexs% + ',\'' + datasetNames[%indexs%] + '\', AVE(dataset_' + datasetNames[%indexs%] + ', scores),' + 'AVE(' + #TEXT(performance_scores) + '(dataset_id=' + %indexs% + '),'+ #TEXT(performance_scores)+'.scores),' + 'TestingSuite.Utils.Student_T_Test(dataset_' + datasetNames[%indexs%]+ ','+ #TEXT(performance_scores) + '(dataset_id=' + %indexs% + '))}');
                        #IF(%indexs%<c_no_of_elements)
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
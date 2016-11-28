EXPORT GenerateCode_R(algorithm, datasetNames):= FUNCTIONMACRO
        #DECLARE(source_code)
        #SET(source_code, '');
        #DECLARE(indexs);
        #SET(indexs, 1);
        

        #LOOP
        	#IF(%indexs%> r_no_of_elements)	
        		#BREAK
        	#ELSE
                        #APPEND(source_code, 'result_' + datasetNames[%indexs%] + ':= ' + algorithm + '(' + QualifiedName('Regression.Datasets.', datasetNames[%indexs%]));
                        #APPEND(source_code, ', Regression.Datasets.' + datasetNames[%indexs%] + '.betas);\n');
                        #SET(indexs,%indexs%+1);
                #END
        #END
        
        #APPEND(source_code, 'final_results := DATASET([');
        #SET(indexs, 1);
        #LOOP
        	#IF(%indexs%>r_no_of_elements)	
        		#BREAK
        	#ELSE
                        #APPEND(source_code, '{' + %indexs% + ',\'' + datasetNames[%indexs%] + '\', result_' + datasetNames[%indexs%] + ', 1, \'FAIL\''  + '}');
                        #IF(%indexs%<r_no_of_elements)
                                #APPEND(source_code, ',\n');
                        #ELSE
                                #APPEND(source_code, '\n');
                        #END
                        #SET(indexs,%indexs%+1);
                #END
        #END
        #APPEND(source_code, '], Utils.Types.dataset_record);\n');
        
        // RETURN %'source_code'%;
        
        // #APPEND(source_code, 'transormed_data_set_record := RECORD\n');
        // #APPEND(source_code, 'final_results;\n');
        // #APPEND(source_code, 'STRING Status;\n');
        // #APPEND(source_code, 'END;\n');
        #APPEND(source_code, 'Utils.Types.dataset_record assign_status(Utils.Types.dataset_record L) := TRANSFORM\n');
        #APPEND(source_code, 'SELF.Status := IF( L.ecl_performance < 1, \'FAIL\', \'PASS\');\n');
        #APPEND(source_code, 'SELF:= L;\n');
        #APPEND(source_code, 'END;\n');
        %source_code%;
        t_final_results := PROJECT(final_results, assign_status(LEFT));
        RETURN t_final_results;
        
ENDMACRO;   

/*
QualifiedName(prefix, datasetname):=  FUNCTIONMACRO
        RETURN prefix + datasetname + '.content';
ENDMACRO;

SET OF STRING regressionDatasetNames := ['AbaloneDS', 'friedman1DS', 'friedman2DS', 'friedman3DS', 'housingDS', 'servoDS'];  
INTEGER r_no_of_elements := COUNT(regressionDatasetNames);
GenerateCode_R('Regression.TestLinearRegression', regressionDatasetNames);*/


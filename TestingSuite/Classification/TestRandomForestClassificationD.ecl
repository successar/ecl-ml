//RandomForest.ecl
IMPORT Std;
IMPORT ML;
IMPORT ML.Tests.Explanatory as TE;
IMPORT ML.Types;
IMPORT TestingSuite.Utils AS Utils;
IMPORT TestingSuite.Classification as Classification;

EXPORT TestRandomForestClassificationD(raw_dataset_name, repeats) := FUNCTIONMACRO
	AnyDataSet :=  TABLE(raw_dataset_name);

	RunRandomForestClassfier(DATASET(Types.DiscreteField) trainIndepData, 
                                DATASET(Types.DiscreteField) trainDepData, 
                                DATASET(Types.DiscreteField) testIndepData, 
                                DATASET(Types.DiscreteField) testDepData,
                                Types.t_Count treeNum, Types.t_Count fsNum, 
                                REAL Purity=1.0, Types.t_level maxLevel=32) := FUNCTION
        learner := ML.Classify.RandomForest(treeNum, fsNum, Purity, maxLevel);  
        result := learner.LearnD(trainIndepData, trainDepData); 
        model:= learner.model(result);  
        class:= learner.classifyD(testIndepData, result); 
        performance:= ML.Classify.Compare(testDepData, class);
        return performance.Accuracy[1].accuracy;
    END;



	WrapperRunRandomForestClassfier(DATASET(RECORDOF(AnyDataSet)) AnyDataSet):= FUNCTION

		// To create training and testing sets
		new_data_set := TABLE(AnyDataSet, {AnyDataSet, select_number := RANDOM()%100});


        t_raw_train_data := new_data_set(select_number <= 40);
        raw_train_data := PROJECT(t_raw_train_data, TRANSFORM(RECORDOF(t_raw_train_data),
                                                                            SELF.id := COUNTER;
                                                                            SELF := LEFT));
        t_raw_test_data := new_data_set(select_number > 40);
        raw_test_data := PROJECT(t_raw_test_data, TRANSFORM(RECORDOF(t_raw_train_data),
                                                                            SELF.id := COUNTER;
                                                                            SELF := LEFT));

		// Splitting data into train and test	
		Utils.ToTraining(raw_train_data, train_data_independent);
		Utils.ToTesting(raw_train_data, train_data_dependent);
		Utils.ToTraining(raw_test_data, test_data_independent);
		Utils.ToTesting(raw_test_data, test_data_dependent);

		ML.ToField(train_data_independent, pr_indep);
		trainIndepData := ML.Discretize.ByRounding(pr_indep);
		ML.ToField(train_data_dependent, pr_dep);
		trainDepData := ML.Discretize.ByRounding(pr_dep);

		ML.ToField(test_data_independent, tr_indep);
		testIndepData := ML.Discretize.ByRounding(tr_indep);
		ML.ToField(test_data_dependent, tr_dep);
		testDepData := ML.Discretize.ByRounding(tr_dep);
		
		result := RunRandomForestClassfier(trainIndepData, trainDepData, testIndepData, testDepData, 100, 4, 1.0, 100);
		return result;
	END;


	IMPORT Std;

	results1 := DATASET([],Utils.Types.result_rec);
    #DECLARE(source_code)
    #SET(source_code, '');
    #DECLARE(indexs);
    #SET(indexs, 1);
    #LOOP
        #IF(%indexs%> repeats)	
            #BREAK;
        #ELSEIF(%indexs% = repeats)
            #APPEND(source_code, 'results := results' + %indexs% + '+ DATASET([{'+%indexs% + ', WrapperRunRandomForestClassfier(AnyDataSet)}], Utils.Types.result_rec) : INDEPENDENT;\n');
            #SET(indexs,%indexs%+1);
        #ELSE
            #APPEND(source_code, 'results'+(%indexs%+1) + ':= results' + %indexs% + '+ DATASET([{'+%indexs% + ', WrapperRunRandomForestClassfier(AnyDataSet)}], Utils.Types.result_rec) : INDEPENDENT;\n');
            #SET(indexs,%indexs%+1);
        #END
    #END
    %source_code%;
	RETURN results;
ENDMACRO;

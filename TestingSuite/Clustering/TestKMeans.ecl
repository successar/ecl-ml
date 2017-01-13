IMPORT Std;
IMPORT ML;
IMPORT ML.Tests.Explanatory as TE;
IMPORT ML.Types;
IMPORT TestingSuite.Utils AS Utils;
IMPORT TestingSuite.Clustering as Clustering;

EXPORT TestKMeans(raw_dataset_name, repeats, no_clusters) := FUNCTIONMACRO
        LOCAL AnyDataSet :=  raw_dataset_name;
		
		LOCAL anyDataWithRandom := PROJECT
					(
						AnyDataSet,
						TRANSFORM
							(
								{
									RECORDOF(AnyDataSet),
									UNSIGNED	_random_num
								},
								SELF._random_num := RANDOM(),
								SELF := LEFT
							)
					);
		
		LOCAL anyDataSubset := PROJECT
			(
				TOPN(anyDataWithRandom, repeats * no_clusters, _random_num),
				TRANSFORM
					(
						RECORDOF(AnyDataSet),
						SELF := LEFT
					)
			);
 
        LOCAL RunKMeans(DATASET(Types.NumericField) dDocuments, DATASET(Types.NumericField) dCentroids):= FUNCTION
                learner := ML.Cluster.KMeans(dDocuments,dCentroids, 100);  
                result := learner.Allegiances();
                RETURN SUM(result, value);
        END;

        LOCAL WrapperRunKmeansClusterer(DATASET(RECORDOF(AnyDataSet)) documentData,
										DATASET(RECORDOF(AnyDataSet)) centroidData):= FUNCTION
				Utils.ToTraining(documentData, dDocumentMatrix);
                Utils.ToTraining(centroidData, dCentroidMatrix);                                                     
                                                       
                ML.ToField(dDocumentMatrix,dDocuments);
                ML.ToField(dCentroidMatrix,dCentroids);
                accuracy := RunKMeans(dDocuments, dCentroids);
                RETURN accuracy;    
        END; 
        
        IMPORT Std;
		
		#DECLARE(indexs);
        #SET(indexs, 1);

		LOCAL results :=
			#LOOP
				#IF(%indexs% <= repeats)
					#IF(%indexs% > 1) + #END
					DATASET
						(
							[
								{
									%indexs%,
									WrapperRunKmeansClusterer(AnyDataSet, CHOOSEN(anyDataSubset, no_clusters, ((%indexs% - 1) * no_clusters + 1)))
								}
							],
							Utils.Types.result_rec
						)
					#SET(indexs,%indexs%+1)
				#ELSE
					#BREAK
				#END
			#END;
		
        RETURN results;
ENDMACRO;
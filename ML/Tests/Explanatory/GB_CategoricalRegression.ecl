IMPORT ML;
IMPORT discrete_houseVoteDS FROM TestingSuite.Classification.Datasets;
IMPORT ML.GradientBoosting as GB;
IMPORT ML.Types;

NumericField := ML.Types.NumericField;

DATASET(NumericField) FormatInputAggregate(DATASET(GB.Losses.LossRecord) losses) := FUNCTION
	RETURN PROJECT(losses, TRANSFORM(NumericField, SELF.id:=LEFT.id, SELF.number:=1, SELF.value:=LEFT.loss));
END;

DummyAggregates := ML.FieldAggregates(DATASET([], NumericField)).Simple;

NamedStat := RECORD(RECORDOF(DummyAggregates))
	STRING name;
END;

AddName(DATASET(RECORDOF(DummyAggregates)) stats, STRING stat_name) := FUNCTION
	RETURN PROJECT(stats, TRANSFORM(NamedStat,
					SELF.name:=stat_name, SELF:=LEFT));
END;


// Dataset
content := discrete_houseVoteDS.content;

// Independent and Dependent Variables
ML.ToField(content, fielded);
class_index := COUNT(fielded) / COUNT(content);
indep := fielded(number < class_index);
dep := fielded(number = class_index);

// Base Linear Regression
model := GB.Regression.Linear(indep, dep, 5, TRUE);
base_errors := SORT(model.BaseModelPredictions(), id);
base_stats := AddName(ML.FieldAggregates(FormatInputAggregate(base_errors)).Simple,'base');

// Gradient Boosting Linear Regression
weights := model.Learn();
predicted := model.Predict(indep, weights);
gradient_errors := SORT(GB.Utils.ComputeErrors(dep, predicted), id);
gradient_stats := AddName(ML.FieldAggregates(FormatInputAggregate(gradient_errors)).Simple, 'grad_linear');

// Gradient Boosting Regression Tree
gb_id3 := ML.GradientBoosting.Regression.CategoricalTree(indep, dep, iterations:=5, doNormalize:=TRUE);
id3_weights := gb_id3.Learn();
id3_predicted := gb_id3.Predict(indep, id3_weights);
id3_gradient_errors := SORT(GB.Utils.ComputeErrors(dep, id3_predicted), id);
id3_gradient_stats := AddName(ML.FieldAggregates(FormatInputAggregate(id3_gradient_errors)).Simple, 'grad_tree');

OUTPUT(base_stats + gradient_stats + id3_gradient_stats);

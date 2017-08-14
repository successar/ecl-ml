IMPORT ML;
IMPORT ML.Types as Types;

EXPORT Predict := MODULE

  SHARED SplitC := ML.DecisionTree.Utils.SplitC;
  SHARED SplitC_Map := ML.DecisionTree.Utils.SplitC_Map;

  SHARED SplitCInstances(DATASET(SplitC) mod, DATASET(Types.NumericField) indep) := FUNCTION
    splits:= mod(new_node_id <> 0); // Get split nodes (branches)
    join0 := JOIN(indep, splits, LEFT.number = RIGHT.number AND RIGHT.high_fork = IF(LEFT.value > RIGHT.value, 1, 0), LOOKUP, MANY);
    sort0 := SORT(join0, id, level, number, node_id, new_node_id, LOCAL);
    dedup0:= DEDUP(sort0, LEFT.id = RIGHT.id AND LEFT.new_node_id != RIGHT.node_id, KEEP 1, LEFT, LOCAL);
    dedup1:= DEDUP(dedup0, LEFT.id = RIGHT.id AND LEFT.new_node_id = RIGHT.node_id, KEEP 1, RIGHT, LOCAL);
    RETURN dedup1;
  END;

  // Predict Dependent variables given indep and decision tree as mod
  EXPORT PredictC(DATASET(Types.NumericField) indep, DATASET(Types.NumericField) mod) := FUNCTION
    ML.FromField(mod, SplitC, nodes, ML.Trees.modelC_Map);
    leaves := nodes(new_node_id = 0);
    // Locate instances into deepest split node based upon independent values
    splitData := SplitCInstances(nodes, indep);
    Types.NumericField predictTransform(RECORDOF(splitData) l, RECORDOF(leaves) r):= TRANSFORM
      SELF.id := l.id;
      SELF.number := 1;
      SELF.value := r.value;
    END;
    RETURN JOIN(splitData, leaves, LEFT.new_node_id = RIGHT.node_id, predictTransform(LEFT, RIGHT), LOOKUP);
  END;

  SHARED SplitD := ML.DecisionTree.Utils.SplitD;
  SHARED SplitD_Map := ML.DecisionTree.Utils.SplitD_Map;

  SHARED SplitDInstances(DATASET(SplitD) mod, DATASET(Types.NumericField) indep) := FUNCTION
    splits:= mod(new_node_id <> 0); // Get split nodes (branches)
    join0 := JOIN(indep, splits, LEFT.number = RIGHT.number AND LEFT.value = RIGHT.value, LOOKUP, MANY);
    sort0 := SORT(join0, id, level, number, node_id, new_node_id, LOCAL);
    dedup0:= DEDUP(sort0, LEFT.id = RIGHT.id AND LEFT.new_node_id != RIGHT.node_id, KEEP 1, LEFT, LOCAL);
    dedup1:= DEDUP(dedup0, LEFT.id = RIGHT.id AND LEFT.new_node_id = RIGHT.node_id, KEEP 1, RIGHT, LOCAL);
    RETURN dedup1;
  END;

  // Predict Dependent variables given indep and decision tree as mod
  EXPORT PredictD(DATASET(Types.NumericField) indep, DATASET(Types.NumericField) mod) := FUNCTION
    ML.FromField(mod, SplitD, nodes, SplitD_Map);
    leaves := nodes(new_node_id = 0);
    // Locate instances into deepest split node based upon independent values
    splitData := SplitDInstances(nodes, indep);
    Types.NumericField predictTransform(RECORDOF(splitData) l, RECORDOF(leaves) r):= TRANSFORM
      SELF.id := l.id;
      SELF.number := 1;
      SELF.value := r.depend;
    END;
    RETURN JOIN(splitData, leaves, LEFT.new_node_id = RIGHT.node_id, predictTransform(LEFT, RIGHT), LOOKUP);
  END;

  SHARED SplitCD := ML.DecisionTree.Utils.SplitCD;
  SHARED SplitCD_Map := ML.DecisionTree.Utils.SplitCD_Map;

  SHARED SplitCDInstances(DATASET(SplitCD) mod, DATASET(Types.NumericField) indep) := FUNCTION
    splits:= mod(new_node_id <> 0); // Get split nodes (branches)
    join0 := JOIN(indep, splits, LEFT.number = RIGHT.number AND
                  ((RIGHT.is_continuous=0 AND LEFT.value=RIGHT.value) OR
                   (RIGHT.is_continuous=1 AND RIGHT.high_fork=IF(LEFT.value > RIGHT.value, 1, 0))),
                 LOOKUP, MANY);
    sort0 := SORT(join0, id, level, number, node_id, new_node_id, LOCAL);
    dedup0:= DEDUP(sort0, LEFT.id = RIGHT.id AND LEFT.new_node_id != RIGHT.node_id, KEEP 1, LEFT, LOCAL);
    dedup1:= DEDUP(dedup0, LEFT.id = RIGHT.id AND LEFT.new_node_id = RIGHT.node_id, KEEP 1, RIGHT, LOCAL);
    RETURN dedup1;
  END;

  // Predict Dependent variables given indep and decision tree as mod
  EXPORT PredictCD(DATASET(Types.NumericField) indep, DATASET(Types.NumericField) mod) := FUNCTION
    ML.FromField(mod, SplitCD, nodes, SplitCD_Map);
    leaves := nodes(new_node_id = 0);
    // Locate instances into deepest split node based upon independent values
    splitData := SplitCDInstances(nodes, indep);
    Types.NumericField predictTransform(RECORDOF(splitData) l, RECORDOF(leaves) r):= TRANSFORM
      SELF.id := l.id;
      SELF.number := 1;
      SELF.value := r.depend;
    END;
    RETURN JOIN(splitData, leaves, LEFT.new_node_id = RIGHT.node_id, predictTransform(LEFT, RIGHT), LOOKUP);
  END;

END;
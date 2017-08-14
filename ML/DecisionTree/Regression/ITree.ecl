IMPORT ML;
IMPORT ML.Types as Types;

EXPORT ITree(Types.t_Count min_NumObj=2, Types.t_level max_Level=32) := MODULE, VIRTUAL

  SHARED minNumObj := min_NumObj;
  SHARED maxLevel := max_Level;

  SHARED StdDev(REAL tot, REAL tot_sq, INTEGER cnt) := FUNCTION
    RETURN SQRT((tot_sq - tot * tot / cnt) / cnt);
  END;

  // Continuous Independent Variables
  SHARED SplitC := ML.Trees.SplitC;
  SHARED TreeNodeC := ML.DecisionTree.Utils.ContinuousTreeNode;

  SHARED VIRTUAL SplitC BinarySplitC(DATASET(Types.NumericField) indep, DATASET(Types.NumericField) dep) := FUNCTION
    RETURN DATASET([], SplitC);
  END;

  // Learn decision tree from Independent and Dependent variables
  // and return the tree as a Dataset of NumericField
  EXPORT LearnC(DATASET(Types.NumericField) indep, DATASET(Types.NumericField) dep) := FUNCTION
    nodes := BinarySplitC(indep, dep);
    RETURN ML.DecisionTree.Utils.ToNumericTree(nodes);
  END;

  // Convert a model from numeric to table of SplitC Records.
  EXPORT getModelC(DATASET(Types.NumericField) mod) := FUNCTION
    ML.FromField(mod, SplitC, nodes,ML.DecisionTree.Utils.SplitC_Map);
    RETURN nodes;
  END;

  // Categorical Independent Variables
  SHARED SplitD := ML.DecisionTree.Utils.SplitD;
  SHARED TreeNodeD := ML.DecisionTree.Utils.CategoricalTreeNode;

  SHARED VIRTUAL SplitD BinarySplitD(DATASET(Types.NumericField) indep, DATASET(Types.NumericField) dep) := FUNCTION
    RETURN DATASET([], SplitD);
  END;

  EXPORT LearnD(DATASET(Types.NumericField) indep, DATASET(Types.NumericField) dep) := FUNCTION
    nodes := BinarySplitD(indep, dep);
    RETURN ML.DecisionTree.Utils.ToOrdinalTree(nodes);
  END;

  // Convert a model from numeric to table of SplitC Records.
  EXPORT getModelD(DATASET(Types.NumericField) mod) := FUNCTION
    ML.FromField(mod, SplitD, nodes,ML.DecisionTree.Utils.SplitD_Map);
    RETURN nodes;
  END;

  // Categorical and Continuous Variables
  SHARED SplitCD := ML.DecisionTree.Utils.SplitCD;
  SHARED TreeNodeCD := ML.DecisionTree.Utils.MixedTreeNode;
  SHARED FieldType := ML.DecisionTree.Utils.FieldType;

  SHARED VIRTUAL SplitCD BinarySplitCD(DATASET(Types.NumericField) indep, DATASET(Types.NumericField) dep, DATASET(FieldType) f_types) := FUNCTION
    RETURN DATASET([], SplitCD);
  END;

  EXPORT LearnCD(DATASET(Types.NumericField) indep, DATASET(Types.NumericField) dep, DATASET(FieldType) f_types) := FUNCTION
    nodes := BinarySplitCD(indep, dep, f_types);
    RETURN ML.DecisionTree.Utils.ToMixedTree(nodes);
  END;

  // Convert a model from numeric to table of SplitCD Records.
  EXPORT getModelCD(DATASET(Types.NumericField) mod) := FUNCTION
    ML.FromField(mod, SplitCD, nodes,ML.DecisionTree.Utils.SplitCD_Map);
    RETURN nodes;
  END;

END;
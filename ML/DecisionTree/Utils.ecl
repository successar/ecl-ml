IMPORT ML;
IMPORT ML.Types as Types;

NumericField := ML.Types.NumericField;

EXPORT Utils := MODULE

  EXPORT SplitC := ML.Trees.SplitC;
  EXPORT STRING SplitC_Fields := ML.Trees.modelC_fields;
  EXPORT SplitC_Map :=  DATASET([{'id','ID'},{'node_id','1'},{'level','2'},{'number','3'},
                                  {'value','4'},{'high_fork','5'},{'new_node_id','6'}],
                                {STRING orig_name; STRING assigned_name;});

  EXPORT SplitD := RECORD   // data structure for splitting results
    Types.NodeID;
    Types.t_FieldNumber number;       // The attribute used to split
    Types.t_FieldReal    value;        // The discrete value for the attribute in question
    Types.t_FieldReal   depend;       // Dependent value
    Types.t_node        new_node_id;  // The new node identifier this branch links to
  END;
  EXPORT STRING SplitD_Fields := 'node_id,level,number,value,depend,new_node_id';
  EXPORT SplitD_Map :=  DATASET([{'id','ID'},{'node_id','1'},{'level','2'},{'number','3'},
                                  {'value','4'},{'depend','5'},{'new_node_id','6'}],
                                {STRING orig_name; STRING assigned_name;});

  EXPORT SplitCD := RECORD   // data structure for splitting results
    Types.NodeID;
    Types.t_FieldNumber number;       // The attribute used to split
    Types.t_FieldReal    value;       // The discrete or continuous value for the attribute in question
    Types.t_FieldReal   depend;       // Dependent value
    INTEGER1 is_continuous := 0;    // Is Continuous attribute or discrete?
    INTEGER1 high_fork:=0;
    Types.t_node        new_node_id;  // The new node identifier this branch links to
  END;
  EXPORT STRING SplitCD_Fields := 'node_id,level,number,value,depend,is_continuous,high_fork,new_node_id';
  EXPORT SplitCD_Map :=  DATASET([{'id','ID'},{'node_id','1'},{'level','2'},{'number','3'},{'value','4'},
                                 {'depend','5'},{'is_continuous','6'},{'high_fork','7'},{'new_node_id','8'}],
                                {STRING orig_name; STRING assigned_name;});

  EXPORT ToNumericTree(DATASET(SplitC) nodes) := FUNCTION
    ML.AppendID(nodes, id, model);
    ML.ToField(model, out_model, id, SplitC_Fields);
    RETURN out_model;
  END;

  EXPORT ToOrdinalTree(DATASET(SplitD) nodes) := FUNCTION
    ML.AppendID(nodes, id, model);
    ML.ToField(model, out_model, id, SplitD_Fields);
    RETURN out_model;
  END;

  EXPORT ToMixedTree(DATASET(SplitCD) nodes) := FUNCTION
    ML.AppendID(nodes, id, model);
    ML.ToField(model, out_model, id, SplitCD_Fields);
    RETURN out_model;
  END;

  EXPORT ContinuousTreeNode := RECORD
    Types.NodeID;
    Types.NumericField;
    Types.t_FieldReal depend;
    BOOLEAN high_fork := FALSE;
    Types.t_Node child_id := 0;
  END;

  EXPORT CategoricalTreeNode := RECORD
    Types.NodeID;
    Types.NumericField;
    Types.t_FieldReal depend;
    Types.t_Node child_id := 0;
  END;

  EXPORT FieldType := RECORD
    Types.t_FieldNumber number;
    BOOLEAN is_continuous := TRUE;
  END;

  EXPORT MixedTreeNode := RECORD
    Types.NodeID;
    Types.NumericField;
    Types.t_FieldReal depend;
    Types.t_Node child_id := 0;
    BOOLEAN is_continuous := TRUE;
    BOOLEAN high_fork := FALSE;
  END;

  EXPORT GetFieldTypes(DATASET(NumericField) indeps, DATASET(FieldType) presets, BOOLEAN is_cont_default=TRUE) := FUNCTION
    attrs := DEDUP(indeps, number, HASH);
    unsets := JOIN(attrs, presets, LEFT.number=RIGHT.number, TRANSFORM(FieldType, SELF.number:=LEFT.number, SELF.is_continuous:=is_cont_default), LEFT ONLY);
    RETURN SORT(presets + unsets, number);
  END;
END;
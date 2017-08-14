IMPORT ML;
IMPORT ML.Types as Types;

EXPORT ID3(Types.t_Count min_NumObj=2, Types.t_level max_Level=32) := MODULE(ML.DecisionTree.Regression.ITree(min_NumObj, max_Level))

  // Function that splits all nodes at a level based on minimizing std deviation
  SHARED BinaryPartitionC(DATASET(TreeNodeC) nodes, Types.t_level p_level) := FUNCTION
    node_base := MAX(nodes, node_id);
    nodes_level := nodes(level = p_level);
    nodes_level_distrib := DISTRIBUTE(nodes_level, HASH(node_id, number));
    root_sorted := SORT(nodes_level_distrib, node_id, number, value, depend, LOCAL);
    attrib1 := root_sorted(number=1);
    node_dep := TABLE(attrib1, {node_id, tot := COUNT(GROUP), depend_sum := SUM(GROUP, depend),
                      depend_sum_sq:=SUM(GROUP, depend*depend), std_dev := SQRT(VARIANCE(GROUP, depend)),
                      mean:= AVE(GROUP, depend), minSplit:=minNumObj},
                node_id, UNSORTED);
    root_noSplit := node_dep(tot < (2 * minNumObj));
    root_noSplit_node := DEDUP(SORT(JOIN(root_noSplit, node_dep, LEFT.node_id=RIGHT.node_id,
                         TRANSFORM(RIGHT), MANY LOOKUP), node_id, -tot), node_id);
    // Transforming NoSplit Nodes into LEAF Nodes to return
    pass_pure_NoSplit:= PROJECT(root_NoSplit_node, TRANSFORM(TreeNodeC, SELF.id:=0, SELF.number:=0, SELF.value:=0, SELF.level:=p_level, SELF.depend:=LEFT.mean, SELF:=LEFT));
    // Compact Impure Node's data to unique node-attribute values
    root_impure_all := JOIN(root_sorted, root_noSplit, LEFT.node_id = RIGHT.node_id, TRANSFORM(LEFT), LEFT ONLY, LOOKUP);
    root_impure_all_distrib := DISTRIBUTE(root_impure_all, HASH(node_id, number));
    root_acc := TABLE(root_impure_all_distrib, {node_id, number, value, cut_cnt:=COUNT(GROUP), dep_sum:=SUM(GROUP, depend), dep_sum_sq:=SUM(GROUP, depend*depend)},
                      node_id, number, value, LOCAL);
    REC_CUT:= RECORD
      root_acc;
      INTEGER tot_Low:=0;  // number of ocurrences <= treshold
      INTEGER tot_High:=0; // number of ocurrences > treshold
      INTEGER tot;      // number of ocurrences
      REAL all_dep_sum;
      REAL all_dep_sum_sq;
      REAL all_std_dev;
      REAL p_low:=0.0;
      REAL p_high:=0.0;
      REAL dep_sum_low:=0.0;
      REAL dep_sum_high:=0.0;
      REAL dep_sum_sq_low:=0.0;
      REAL dep_sum_sq_high:=0.0;
      REAL info:=0.0;
      REAL minSplit;       // minimum number of occurrences needed to perform a Split
    END;
    cuts := JOIN(root_acc, node_dep, LEFT.node_id = RIGHT.node_id,
                TRANSFORM(REC_CUT, SELF.tot:=RIGHT.tot, SELF.minSplit:=RIGHT.minSplit,
                    SELF.all_dep_sum:=RIGHT.depend_sum, SELF.all_dep_sum_sq:=RIGHT.depend_sum_sq,
                    SELF.all_std_dev:=RIGHT.std_dev, SELF:=LEFT), LOOKUP);
    sort_cuts:= SORT(cuts, node_id, number, value, LOCAL);
    // Set of all posible split points (total counts and split Info initialized with 0)
    REC_CUT rol(sort_cuts le, sort_cuts ri) := TRANSFORM
      t_low:=   ri.cut_Cnt + IF(le.node_id=ri.node_id AND le.number=ri.number , le.tot_Low, 0);
      t_high:=  ri.tot - ri.cut_Cnt - IF(le.node_id=ri.node_id AND le.number=ri.number , le.tot_Low, 0);
      d_sum_low := ri.dep_sum + IF(le.node_id=ri.node_id AND le.number=ri.number , le.dep_sum_low, 0);
      d_sum_high := ri.all_dep_sum - ri.dep_sum - IF(le.node_id=ri.node_id AND le.number=ri.number , le.dep_sum_low, 0);
      d_sum_sq_low := ri.dep_sum_sq + IF(le.node_id=ri.node_id AND le.number=ri.number , le.dep_sum_sq_low, 0);
      d_sum_sq_high := ri.all_dep_sum_sq - ri.dep_sum_sq - IF(le.node_id=ri.node_id AND le.number=ri.number , le.dep_sum_sq_low, 0);
      SELF.p_low:=   t_low/ri.tot;
      SELF.p_high:=  t_high/ri.tot;
      SELF.tot_Low:= t_low;
      SELF.tot_High:= t_high;
      SELF.dep_sum_low:=d_sum_low;
      SELF.dep_sum_high:=d_sum_high;
      SELF.dep_sum_sq_low:=d_sum_sq_low;
      SELF.dep_sum_sq_high:=d_sum_sq_high;
      low_std := StdDev(d_sum_low, d_sum_sq_low, t_low);
      high_std := StdDev(d_sum_high, d_sum_sq_high, t_high);
      SELF.info := le.all_std_dev - (t_low/ri.tot * low_std + t_high/ri.tot * high_std);
      SELF := ri;
    END;
    // Accumulated Counting: t_low # ocurrences <= treshold , t_high # ocurrences > treshold
    x := ITERATE(sort_cuts, rol(LEFT,RIGHT), LOCAL);
    // Filtering cuts with not enough occurrences needed to perform a Split
    cuts_ok:= x((tot_Low >= minSplit) AND (tot_High >= minSplit));
    nodes_ok:= TABLE(cuts_ok, {node_id}, node_id, MERGE);
    cuts_noSplit:= TABLE(x((tot_Low < minSplit ) OR (tot_High < minSplit)), {node_id}, node_id, MERGE);
    // Nodes with none acceptable splits become LEAFS
    node_noSplit:= JOIN(cuts_noSplit, nodes_ok, LEFT.node_id=RIGHT.node_id, LEFT ONLY, LOOKUP);
    noSplit_dep := DEDUP(SORT(JOIN(node_noSplit, node_dep, LEFT.node_id = RIGHT.node_id,
                                TRANSFORM(RIGHT), MANY LOOKUP), node_id, -tot), node_id);
    pass_thru_noSplit:= PROJECT(noSplit_dep, TRANSFORM(TreeNodeC, SELF.level:= p_level, SELF.id:=0,
                                  SELF.number:=0, SELF.value:=0, SELF.depend:=LEFT.mean, SELF:=LEFT));
    // bags_info := DEDUP(SORT(PROJECT(cuts_ok, TRANSFORM(Bag_Info, SELF:=LEFT)), node_id, -info), node_id);
    best_cuts := DEDUP(SORT(cuts_ok, node_id, -info), node_id);
    leaf_nodes := pass_pure_NoSplit + pass_thru_noSplit;
    non_leaf := JOIN(root_impure_all, leaf_nodes, LEFT.node_id = RIGHT.node_id, LEFT ONLY, LOOKUP);
    // Start allocating new node-ids from the highest previous
    new_nodes_low := PROJECT(best_cuts, TRANSFORM(TreeNodeC, SELF.id:=0, SELF.value:=LEFT.value,
                              SELF.depend:=LEFT.dep_sum_low/LEFT.tot_low, SELF.level:=p_level,
                              SELF.high_fork:=FALSE, SELF.child_id:=node_base + 2*COUNTER - 1, SELF:=LEFT));
    new_nodes_high := PROJECT(best_cuts, TRANSFORM(TreeNodeC, SELF.id:=0, SELF.value:=LEFT.value,
                              SELF.depend:=LEFT.dep_sum_high/LEFT.tot_high, SELF.level:=p_level,
                              SELF.high_fork:=TRUE, SELF.child_id:=node_base + 2*COUNTER, SELF:=LEFT));
    new_nodes := new_nodes_low + new_nodes_high;
    R1 := RECORD
      Types.t_Recordid id;
      Types.t_node nodeid;
      BOOLEAN high_fork:=FALSE;
    END;
    record_map := JOIN(non_leaf, new_nodes, LEFT.node_id=RIGHT.node_id AND LEFT.number=RIGHT.number
                                            AND (LEFT.value>RIGHT.value)=RIGHT.high_fork,
                       TRANSFORM(R1, SELF.id:=LEFT.id, SELF.nodeid:=RIGHT.child_id,
                       SELF.high_fork:=RIGHT.high_fork), LOOKUP);
    // Now use the map to actually reset all the points.
    re_mapped_records := JOIN(non_leaf, record_map, LEFT.id=RIGHT.id, TRANSFORM(TreeNodeC, SELF.node_id:=RIGHT.nodeid,
                              SELF.level:=LEFT.level+1, SELF.high_fork:=RIGHT.high_fork, SELF:=LEFT));
    RETURN nodes(level < p_level) + leaf_nodes + new_nodes + re_mapped_records;
  END;

  // Function that is used to create a decision tree based on binary splits.
  SHARED BinarySplitC(DATASET(Types.NumericField) indep, DATASET(Types.NumericField) dep) := FUNCTION
    depth := MIN(1023, maxLevel);
    ind0 := Indep;
    // Initialize nodes
    TreeNodeC init(ind0 le, dep ri) := TRANSFORM
      SELF.node_id := 1;
      SELF.level := 1;
      SELF.depend := ri.value;
      SELF := le;
    END;
    // All instances start at root node (node_id = 1)
    root := JOIN(ind0, dep, LEFT.id = RIGHT.id, init(LEFT, RIGHT));
    // LOOP keep going until split is not possible or the tree reachs maximum level: (~sp or ~mx <=> sp AND mx)
    looped := LOOP(root, MAX(ROWS(LEFT), level) >= COUNTER AND COUNTER < depth, BinaryPartitionC(ROWS(LEFT), COUNTER));
    //node splits
    splits := PROJECT(looped(id=0, number>0), TRANSFORM(SplitC, SELF.new_node_id:=LEFT.child_id,
                      SELF.high_fork:=(INTEGER1)LEFT.high_fork, SELF := LEFT));
    leaves1 := PROJECT(looped(id=0, number=0), TRANSFORM(SplitC, SELF.new_node_id:=0, SELF.value:=LEFT.depend,
                      SELF.high_fork:=(INTEGER1)LEFT.high_fork, SELF := LEFT));
    // non completed leaves
    non_completed := TABLE(looped(id>0, number=1), {node_id, level, ave_dep:=AVE(GROUP, depend)}, node_id, level);
    leaves2 := PROJECT(non_completed, TRANSFORM(SplitC, SELF.new_node_id:=0, SELF.number:=0, SELF.value:=LEFT.ave_dep,
                       SELF.high_fork:=0, SELF:=LEFT));
    RETURN splits + leaves1 + leaves2;
  END;

  SHARED BinaryPartitionD(DATASET(TreeNodeD) nodes, Types.t_level p_level) := FUNCTION
    node_base := MAX(nodes, node_id);
    nodes_level := nodes(level = p_level);
    nodes_level_distrib := DISTRIBUTE(nodes_level, HASH(node_id, number));
    root_sorted := SORT(nodes_level_distrib, node_id, number, value, depend, LOCAL);
    attrib1 := root_sorted(number=1);
    node_dep := TABLE(attrib1, {node_id, tot := COUNT(GROUP), depend_sum := SUM(GROUP, depend),
                      depend_sum_sq:=SUM(GROUP, depend*depend), std_dev := SQRT(VARIANCE(GROUP, depend)),
                      mean:= AVE(GROUP, depend), minSplit:=minNumObj},
                node_id, FEW);
    root_noSplit := node_dep(tot < (2 * minNumObj));
    root_noSplit_node := DEDUP(SORT(JOIN(root_noSplit, node_dep, LEFT.node_id=RIGHT.node_id,
                         TRANSFORM(RIGHT), MANY LOOKUP), node_id, -tot), node_id);
    // Transforming NoSplit Nodes into LEAF Nodes to return
    pass_pure_NoSplit:= PROJECT(root_NoSplit_node, TRANSFORM(TreeNodeD, SELF.id:=0, SELF.number:=0, SELF.value:=0, SELF.level:=p_level, SELF.depend:=LEFT.mean, SELF:=LEFT));
    // Compact Impure Node's data to unique node-attribute values
    root_impure_all := JOIN(root_sorted, root_noSplit, LEFT.node_id = RIGHT.node_id, TRANSFORM(LEFT), LEFT ONLY, LOOKUP);
    root_impure_all_distrib := DISTRIBUTE(root_impure_all, HASH(node_id, number));
    root_acc := TABLE(root_impure_all_distrib, {node_id, number, value, grp_val_cnt:=COUNT(GROUP), dep_sum:=SUM(GROUP, depend), dep_sum_sq:=SUM(GROUP, depend*depend)},
                      node_id, number, value, LOCAL);

    REC_CUT := RECORD
      root_acc;
      REAL std_dev:=0.0;
      REAL prob:=0.0;
      REAL minSplit;       // minimum number of occurrences needed to perform a Split
    END;
    cuts := JOIN(root_acc, node_dep, LEFT.node_id = RIGHT.node_id,
                TRANSFORM(REC_CUT, SELF.minSplit:=RIGHT.minSplit,
                    SELF.prob:=LEFT.grp_val_cnt/RIGHT.tot,
                    SELF.std_dev:=StdDev(LEFT.dep_sum, LEFT.dep_sum_sq, LEFT.grp_val_cnt), SELF:=LEFT), LOOKUP);
    cuts_info := TABLE(cuts, {node_id, number, split_info:=SUM(GROUP, prob * std_dev)}, node_id, number);
    REC_GAIN := RECORD
      Types.t_node node_id;
      Types.t_Discrete number;
      REAL gain;
    END;
    gains := JOIN(cuts_info, node_dep, LEFT.node_id=RIGHT.node_id, TRANSFORM(REC_GAIN, SELF.node_id:=LEFT.node_id,
                  SELF.number:=LEFT.number, SELF.gain:=RIGHT.std_dev-LEFT.split_info));
    split := DEDUP(SORT(DISTRIBUTE(gains, HASH(node_id)), node_id, -gain, LOCAL), node_id, LOCAL);
    // new split nodes found
    new_spl0  := JOIN(cuts, split, LEFT.node_id = RIGHT.node_id AND LEFT.number = RIGHT.number, TRANSFORM(LEFT), LOOKUP);
    new_split := PROJECT(new_spl0, TRANSFORM(TreeNodeD, SELF.child_id:= node_base + COUNTER; SELF.value:= LEFT.value; SELF.depend:=LEFT.dep_sum/LEFT.grp_val_cnt,
                                     SELF.level:= p_level; SELF := LEFT; SELF := [];));
    R1 := RECORD
      Types.t_Recordid id;
      Types.t_node nodeid;
    END;
    record_map := JOIN(root_impure_all_distrib, new_split, LEFT.node_id=RIGHT.node_id AND LEFT.number=RIGHT.number AND LEFT.value=RIGHT.value,
                       TRANSFORM(R1, SELF.id:=LEFT.id, SELF.nodeid:=RIGHT.child_id), LOOKUP, LOCAL);
    node_inst := JOIN(root_impure_all_distrib, record_map, LEFT.id=RIGHT.id,
                      TRANSFORM(TreeNodeD, SELF.node_id:=RIGHT.nodeid, SELF.level:=LEFT.level+1, SELF:=LEFT), LOCAL);
    RETURN nodes(level < p_level) + pass_pure_NoSplit + new_split + node_inst;
  END;

  SHARED BinarySplitD(DATASET(Types.NumericField) indep, DATASET(Types.NumericField) dep) := FUNCTION
    ind0 := indep;
    depth := MIN(maxLevel, COUNT(ind0(id = dep[1].id)));

    // Initialize nodes
    TreeNodeD init(ind0 le, dep ri) := TRANSFORM
      SELF.node_id := 1;
      SELF.level := 1;
      SELF.depend := ri.value;
      SELF := le;
    END;
    // All instances start at root node (node_id = 1)
    root := JOIN(ind0, dep, LEFT.id = RIGHT.id, init(LEFT, RIGHT));
    // LOOP keep going until split is not possible or the tree reachs maximum level: (~sp or ~mx <=> sp AND mx)
    looped := LOOP(root, COUNTER < depth, BinaryPartitionD(ROWS(LEFT), COUNTER));
    //node splits
    splits := PROJECT(looped(id=0), TRANSFORM(SplitD, SELF.new_node_id:=LEFT.child_id, SELF := LEFT));

    // non completed leaves
    non_completed := TABLE(looped(id>0), {node_id, level, ave_dep:=AVE(GROUP, depend)}, node_id, level);
    leaves := PROJECT(non_completed, TRANSFORM(SplitD, SELF.new_node_id:=0, SELF.number:=0, SELF.value:=0, SELF.depend:=LEFT.ave_dep, SELF:=LEFT));
    RETURN splits + leaves;
  END;

  SHARED BinaryPartitionCD(DATASET(TreeNodeCD) nodes, Types.t_level p_level, DATASET(FieldType) f_types) := FUNCTION
    node_base := MAX(nodes, node_id);
    nodes_level := nodes(level = p_level);
    nodes_level_distrib := DISTRIBUTE(nodes_level, HASH(node_id, number));
    root_sorted := SORT(nodes_level_distrib, node_id, number, value, depend, LOCAL);
    attrib1 := root_sorted(number=1);
    node_dep := TABLE(attrib1, {node_id, tot := COUNT(GROUP), depend_sum := SUM(GROUP, depend),
                      depend_sum_sq:=SUM(GROUP, depend*depend), std_dev := SQRT(VARIANCE(GROUP, depend)),
                      mean:= AVE(GROUP, depend), minSplit:=minNumObj},
                node_id, FEW);
    root_noSplit := node_dep(tot < (2 * minNumObj));
    root_noSplit_node := DEDUP(SORT(JOIN(root_noSplit, node_dep, LEFT.node_id=RIGHT.node_id,
                         TRANSFORM(RIGHT), MANY LOOKUP), node_id, -tot), node_id);
    // Transforming NoSplit Nodes into LEAF Nodes to return
    pass_pure_NoSplit:= PROJECT(root_NoSplit_node, TRANSFORM(TreeNodeCD, SELF.id:=0, SELF.number:=0,
                                SELF.value:=0, SELF.level:=p_level, SELF.depend:=LEFT.mean, SELF:=LEFT));
    // Compact Impure Node's data to unique node-attribute values
    root_impure_all := JOIN(root_sorted, root_noSplit, LEFT.node_id = RIGHT.node_id, TRANSFORM(LEFT), LEFT ONLY, LOOKUP);
    root_impure_all_distrib := DISTRIBUTE(root_impure_all, HASH(node_id, number));
    root_acc := TABLE(root_impure_all_distrib, {node_id, number, value, is_continuous, cut_cnt:=COUNT(GROUP),
                      dep_sum:=SUM(GROUP, depend), dep_sum_sq:=SUM(GROUP, depend*depend)},
                      node_id, number, value, is_continuous, LOCAL);
    REC_CONT_CUT:= RECORD
      root_acc;
      INTEGER tot_Low:=0;  // number of ocurrences <= treshold
      INTEGER tot_High:=0; // number of ocurrences > treshold
      INTEGER tot;      // number of ocurrences
      REAL all_dep_sum;
      REAL all_dep_sum_sq;
      REAL all_std_dev;
      REAL p_low:=0.0;
      REAL p_high:=0.0;
      REAL dep_sum_low:=0.0;
      REAL dep_sum_high:=0.0;
      REAL dep_sum_sq_low:=0.0;
      REAL dep_sum_sq_high:=0.0;
      REAL info:=0.0;
      REAL minSplit;       // minimum number of occurrences needed to perform a Split
    END;
    REC_GAIN := RECORD
      Types.t_node node_id;
      Types.t_Discrete number;
      Types.t_FieldReal value := 0.0;
      BOOLEAN is_continuous := TRUE;
      Types.t_FieldReal gain;
      Types.t_FieldReal depend := 0.0;
      // MetaData
      Types.t_FieldReal dep_sum_low := 0.0;
      Types.t_FieldReal dep_sum_high := 0.0;
      Types.t_FieldReal tot_low := 0.0;
      Types.t_FieldReal tot_high := 0.0;
    END;
    // Continuous Attributes
    cont_cuts := JOIN(root_acc(is_continuous=TRUE), node_dep, LEFT.node_id = RIGHT.node_id,
                TRANSFORM(REC_CONT_CUT, SELF.tot:=RIGHT.tot, SELF.minSplit:=RIGHT.minSplit,
                    SELF.all_dep_sum:=RIGHT.depend_sum, SELF.all_dep_sum_sq:=RIGHT.depend_sum_sq,
                    SELF.all_std_dev:=RIGHT.std_dev, SELF:=LEFT), LOOKUP);
    sort_cont_cuts:= SORT(cont_cuts, node_id, number, value, LOCAL);
    // Set of all posible split points (total counts and split Info initialized with 0)
    REC_CONT_CUT rol(sort_cont_cuts le, sort_cont_cuts ri) := TRANSFORM
      t_low:=   ri.cut_Cnt + IF(le.node_id=ri.node_id AND le.number=ri.number , le.tot_Low, 0);
      t_high:=  ri.tot - ri.cut_Cnt - IF(le.node_id=ri.node_id AND le.number=ri.number , le.tot_Low, 0);
      d_sum_low := ri.dep_sum + IF(le.node_id=ri.node_id AND le.number=ri.number , le.dep_sum_low, 0);
      d_sum_high := ri.all_dep_sum - ri.dep_sum - IF(le.node_id=ri.node_id AND le.number=ri.number , le.dep_sum_low, 0);
      d_sum_sq_low := ri.dep_sum_sq + IF(le.node_id=ri.node_id AND le.number=ri.number , le.dep_sum_sq_low, 0);
      d_sum_sq_high := ri.all_dep_sum_sq - ri.dep_sum_sq - IF(le.node_id=ri.node_id AND le.number=ri.number , le.dep_sum_sq_low, 0);
      SELF.p_low:=   t_low/ri.tot;
      SELF.p_high:=  t_high/ri.tot;
      SELF.tot_Low:= t_low;
      SELF.tot_High:= t_high;
      SELF.dep_sum_low:=d_sum_low;
      SELF.dep_sum_high:=d_sum_high;
      SELF.dep_sum_sq_low:=d_sum_sq_low;
      SELF.dep_sum_sq_high:=d_sum_sq_high;
      low_std := StdDev(d_sum_low, d_sum_sq_low, t_low);
      high_std := StdDev(d_sum_high, d_sum_sq_high, t_high);
      SELF.info := le.all_std_dev - (t_low/ri.tot * low_std + t_high/ri.tot * high_std);
      SELF := ri;
    END;
    looped_cont_cuts := ITERATE(sort_cont_cuts, rol(LEFT,RIGHT), LOCAL);
    cont_cuts_ok:= looped_cont_cuts((tot_Low >= minSplit) AND (tot_High >= minSplit));
    cont_gains := PROJECT(DEDUP(SORT(cont_cuts_ok, node_id, number, -info), node_id, number),
                              TRANSFORM(REC_GAIN, SELF.gain:=LEFT.info, SELF:=LEFT));

    // Discrete Attributes
    REC_DISC_CUT := RECORD
      root_acc;
      REAL std_dev:=0.0;
      REAL prob:=0.0;
      REAL minSplit;       // minimum number of occurrences needed to perform a Split
    END;
    disc_cuts := JOIN(root_acc(is_continuous=FALSE), node_dep, LEFT.node_id = RIGHT.node_id,
                TRANSFORM(REC_DISC_CUT, SELF.minSplit:=RIGHT.minSplit,
                    SELF.prob:=LEFT.cut_cnt/RIGHT.tot,
                    SELF.std_dev:=StdDev(LEFT.dep_sum, LEFT.dep_sum_sq, LEFT.cut_cnt), SELF:=LEFT), LOOKUP);
    disc_cuts_info := TABLE(disc_cuts, {node_id, number, split_info:=SUM(GROUP, prob * std_dev)}, node_id, number);
    disc_gains := JOIN(disc_cuts_info, node_dep, LEFT.node_id=RIGHT.node_id, TRANSFORM(REC_GAIN, SELF.node_id:=LEFT.node_id,
                  SELF.number:=LEFT.number, SELF.gain:=RIGHT.std_dev-LEFT.split_info, SELF.is_continuous:=FALSE));
    // Nodes with none acceptable splits become LEAFS
    all_gains := cont_gains + disc_gains;
    nodes_ok := TABLE(all_gains, {node_id}, node_id, MERGE);
    cuts_noSplit := TABLE(looped_cont_cuts((tot_Low < minSplit ) OR (tot_High < minSplit)), {node_id}, node_id, MERGE);
    node_noSplit:= JOIN(cuts_noSplit, nodes_ok, LEFT.node_id=RIGHT.node_id, LEFT ONLY, LOOKUP);
    noSplit_dep := DEDUP(SORT(JOIN(node_noSplit, node_dep, LEFT.node_id = RIGHT.node_id,
                                TRANSFORM(RIGHT), MANY LOOKUP), node_id, -tot), node_id);
    pass_thru_noSplit:= PROJECT(noSplit_dep, TRANSFORM(TreeNodeCD, SELF.level:= p_level, SELF.id:=0,
                                  SELF.number:=0, SELF.value:=0, SELF.depend:=LEFT.mean, SELF:=LEFT));
    best_cuts := DEDUP(SORT(all_gains, node_id, -gain), node_id);
    leaf_nodes := pass_pure_NoSplit + pass_thru_noSplit;
    non_leaf := JOIN(root_impure_all, leaf_nodes, LEFT.node_id = RIGHT.node_id, LEFT ONLY, LOOKUP);
    // New Nodes
    new_cont_nodes_low := PROJECT(best_cuts(is_continuous=TRUE), TRANSFORM(TreeNodeCD, SELF.id:=0, SELF.value:=LEFT.value,
                                  SELF.depend:=LEFT.dep_sum_low/LEFT.tot_low, SELF.level:=p_level,
                                  SELF.high_fork:=FALSE, SELF.child_id:=node_base+COUNTER, SELF:=LEFT));
    node_cont_base := MAX(MAX(new_cont_nodes_low, child_id), node_base);
    new_cont_nodes_high := PROJECT(best_cuts(is_continuous=TRUE), TRANSFORM(TreeNodeCD, SELF.id:=0, SELF.value:=LEFT.value,
                                  SELF.depend:=LEFT.dep_sum_high/LEFT.tot_high, SELF.level:=p_level,
                                  SELF.high_fork:=TRUE, SELF.child_id:=node_cont_base+COUNTER, SELF:=LEFT));
    new_cont_nodes := new_cont_nodes_low + new_cont_nodes_high;
    new_disc_cuts := JOIN(disc_cuts, best_cuts(is_continuous=FALSE),
                          LEFT.node_id=RIGHT.node_id AND LEFT.number=RIGHT.number,
                          TRANSFORM(LEFT), LOOKUP);
    node_disc_base := MAX(MAX(new_cont_nodes, child_id), node_cont_base);
    new_disc_nodes := PROJECT(new_disc_cuts, TRANSFORM(TreeNodeCD, SELF.child_id:= node_disc_base + COUNTER; SELF.value:= LEFT.value; SELF.depend:=LEFT.dep_sum/LEFT.cut_cnt,
                             SELF.level:= p_level; SELF := LEFT; SELF := [];));
    new_nodes := new_cont_nodes + new_disc_nodes;
    // New Record Mappings
    R1 := RECORD
      Types.t_Recordid id;
      Types.t_node nodeid;
      BOOLEAN high_fork:=FALSE;
    END;
    record_map := JOIN(non_leaf, new_nodes, LEFT.node_id=RIGHT.node_id AND LEFT.number=RIGHT.number
                                            AND ((RIGHT.is_continuous=FALSE AND LEFT.value=RIGHT.value) OR
                                                 (RIGHT.is_continuous=TRUE AND ((LEFT.value>RIGHT.value)=RIGHT.high_fork))),
                       TRANSFORM(R1, SELF.id:=LEFT.id, SELF.nodeid:=RIGHT.child_id,
                                 SELF.high_fork:=RIGHT.high_fork), LOOKUP);
    re_mapped_records := JOIN(non_leaf, record_map, LEFT.id=RIGHT.id, TRANSFORM(TreeNodeCD, SELF.node_id:=RIGHT.nodeid,
                              SELF.level:=LEFT.level+1, SELF.high_fork:=RIGHT.high_fork, SELF:=LEFT));
    RETURN nodes(level < p_level) + leaf_nodes + new_nodes + re_mapped_records;
  END;

  SHARED BinarySplitCD(DATASET(Types.NumericField) indep, DATASET(Types.NumericField) dep, DATASET(FieldType) f_types) := FUNCTION
    ind0 := indep;
    depth := MIN(maxLevel, 1023);

    // Initialize nodes
    TreeNodeCD init(ind0 le, dep ri) := TRANSFORM
      SELF.node_id := 1;
      SELF.level := 1;
      SELF.depend := ri.value;
      SELF := le;
    END;
    // All instances start at root node (node_id = 1)
    root := JOIN(JOIN(ind0, dep, LEFT.id = RIGHT.id, init(LEFT, RIGHT)),
            f_types, LEFT.number = RIGHT.number,
            TRANSFORM(TreeNodeCD, SELF.is_continuous:=RIGHT.is_continuous, SELF:=LEFT));
    looped := LOOP(root, MAX(ROWS(LEFT), level) >= COUNTER AND COUNTER < depth, BinaryPartitionCD(ROWS(LEFT), COUNTER, f_types));
    //node splits
    splits := PROJECT(looped(id=0), TRANSFORM(SplitCD, SELF.new_node_id:=LEFT.child_id,
                      SELF.high_fork:=(INTEGER1)LEFT.high_fork, SELF.is_continuous:=(INTEGER1)LEFT.is_continuous, SELF := LEFT));
    // non completed leaves
    non_completed := TABLE(looped(id>0), {node_id, level, ave_dep:=AVE(GROUP, depend)}, node_id, level);
    leaves := PROJECT(non_completed, TRANSFORM(SplitCD, SELF.new_node_id:=0, SELF.number:=0, SELF.value:=0, SELF.depend:=LEFT.ave_dep, SELF:=LEFT));
    RETURN splits + leaves;
  END;

END;
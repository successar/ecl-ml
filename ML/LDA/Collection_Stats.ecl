IMPORT $.Types AS Types;
/**
 * Document statistics for each model.  Used by the initial model
 * generator and by several of the initiol estimate generators.
 * @param models    the model identifiers used for this run.
 * @param docs      the mapped documents for this run
 * @return          the number of documents, unique words, and total word
 *                  occurrences.
 */
EXPORT DATASET(Types.Model_Collection_Stats)
       Collection_Stats(DATASET(Types.Model_Identifier) models,
                        DATASET(Types.Doc_Mapped) docs) := FUNCTION
  //assign docs to models
  Work_Doc := RECORD(Types.Doc_Mapped)
    Types.t_model_id model;
  END;
  Work_Doc amdl(Types.Doc_Mapped doc, Types.Model_Identifier mdl) := TRANSFORM
    SELF.model := mdl.model;
    SELF := doc;
  END;
  assigned := JOIN(docs, models, RIGHT.model IN LEFT.models,
                   amdl(LEFT, RIGHT), ALL); // model is a small record set
  // Extract the terms for counting
  Work_Term := RECORD
    Types.t_model_id model;
    Types.t_record_id rid;
    Types.t_nominal nominal;
    UNSIGNED4 freq;
  END;
  Work_Term ext_term(Work_Doc doc, Types.TermFreq term) := TRANSFORM
    SELF.rid := doc.rid;
    SELF.model := doc.model;
    SELF.nominal := term.nominal;
    SELF.freq := term.v;
  END;
  t_list := NORMALIZE(assigned, LEFT.word_counts, ext_term(LEFT,RIGHT));
  nominal_list := TABLE(t_list,
                        {model, nominal, occurs:=SUM(GROUP,freq)},
                        model, nominal, MERGE);
  t_tab := TABLE(nominal_list,
                 {model, nominals:=COUNT(GROUP), words:=SUM(GROUP,occurs),
                  UNSIGNED8 low_nominal:=MIN(GROUP,nominal),
                  UNSIGNED8 high_nominal:=MAX(GROUP,nominal)},
                 model, MERGE);
  t_tab_sorted := SORT(DISTRIBUTE(t_tab, model), model, LOCAL);
  d_tab := TABLE(assigned, {model, dc:=COUNT(GROUP)}, model, FEW, UNSORTED);
  d_tab_sorted := SORT(DISTRIBUTE(d_tab, model), model, LOCAL);
  Types.Model_Collection_Stats stats(d_tab ds, t_tab ts) := TRANSFORM
    SELF.model := ds.model;
    SELF.docs := ds.dc;
    SELF.unique_words := ts.nominals;
    SELF.words := ts.words;
    SELF.low_nominal := ts.low_nominal;
    SELF.high_nominal := ts.high_nominal;
  END;
  rslt := COMBINE(d_tab_sorted, t_tab_sorted, stats(LEFT,RIGHT), LOCAL);
  RETURN rslt;
END;
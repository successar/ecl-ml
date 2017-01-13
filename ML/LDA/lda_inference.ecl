// lda_inference by variational process
IMPORT ML.LDA;
IMPORT Types FROM $;
// aliases for convenience
Doc_Topics := Types.Doc_Topics;
Topic_Values := Types.Topic_Values;
Topic_Value := Types.Topic_Value;
TermFreq := Types.TermFreq;
OnlyValue := Types.OnlyValue;

EXPORT DATASET(Types.Doc_Topics)
        lda_inference(DATASET(Types.Doc_Topics) dts, UNSIGNED cnt) := FUNCTION
  // Calculate the new phi arrays
  Doc_Topics calcNewPhi(Doc_Topics doc) := TRANSFORM
    num_words := COUNT(doc.word_counts);
    SELF.t_phis := pre_norm_phis(num_words, doc.t_logBetas, doc.t_digammas);
    SELF := doc;
  END;
  docs_pn_phis := PROJECT(dts, calcNewPhi(LEFT));
  Phi_Sums := RECORD
    Types.t_model_id model;
    Types.t_record_id rid;
    UNSIGNED4 num_ranges;
    UNSIGNED4 topic_range;
    EMBEDDED DATASET(OnlyValue) phi_sums;
  END;
  Phi_Sums extSums(Doc_Topics doc) := TRANSFORM
    SELF.phi_sums := topic_values_sum(doc.t_phis);
    SELF := doc;
  END;
  p_sums_4_range := PROJECT(docs_pn_phis, extSums(LEFT));
  Phi_Sums rollSums(Phi_sums sum_phis, Phi_sums incr) := TRANSFORM
    SELF.phi_sums := value_sum(sum_phis.phi_sums, incr.phi_sums);
    SELF := sum_phis;
  END;
  p_sums := ROLLUP(p_sums_4_range, rollSums(LEFT, RIGHT), model, rid, LOCAL);
  p_sums_r := NORMALIZE(p_sums,LEFT.num_ranges,TRANSFORM(Phi_Sums,SELF:=LEFT));
  Topic_Value new_digamma(Topic_Value g) := TRANSFORM
    SELF.topic := g.topic;
    SELF.v := digamma(g.v);
  END;
  Doc_Topics norm_phis(Doc_Topics doc, Phi_Sums ps) := TRANSFORM
    ASSERT(doc.model=ps.model AND doc.rid=ps.rid,
           'Key mismatch, ' + doc.model + '/'+ doc.rid + ':' + doc.topic_range
           + ' v '
           + ps.model + '/' + ps.rid + ':' + ps.topic_range , FAIL);
    t_phis := post_norm_phis(doc.t_phis, ps.phi_sums);
    t_gammas := update_gammas(doc.alpha, t_phis, doc.word_counts);
    SELF.t_phis := t_phis;
    SELF.t_gammas := t_gammas;
    SELF.t_digammas := PROJECT(t_gammas, new_digamma(LEFT));
    SELF := doc;
  END;
  new_dts := COMBINE(docs_pn_phis, p_sums_r, norm_phis(LEFT,RIGHT), LOCAL);
  // Now determine the document log likelihood
  //Formulae uses phi values, gamma values, alpha, and beta values.  Inputs
  //include some document level components, so an initial pass is made
  //to gather up the document level values.
  LogLikelihood := RECORD
    Types.t_model_id model;
    Types.t_record_id rid;
    UNSIGNED4 num_ranges;
    REAL8 log_likelihood;
  END;
  Doc_level := RECORD
    Types.t_model_id model;
    Types.t_record_id rid;
    UNSIGNED4 num_ranges;
    UNSIGNED4 num_topics;
    REAL8 sum_gamma_k;
    REAL8 sum_lgamma_gamma_k;
    REAL8 sum_alpha_k;
    REAL8 sum_lgamma_alpha_k;
  END;
  Doc_level extValues(Doc_Topics dts) := TRANSFORM
    range_topics := (dts.topic_high-dts.topic_low+1);
    SELF.sum_gamma_k := SUM(dts.t_gammas, v);
    SELF.sum_lgamma_gamma_k := SUM(dts.t_gammas, log_gamma(v));
    SELF.sum_alpha_k := range_topics * dts.alpha;
    SELF.sum_lgamma_alpha_k := range_topics * log_gamma(dts.alpha);
    SELF := dts;
  END;
  range_level_values := PROJECT(new_dts, extValues(LEFT));
  grp_rlv := GROUP(range_level_values, model, rid, LOCAL);
  Doc_Level roll_values(Doc_Level dl, DATASET(Doc_Level) g_dls) := TRANSFORM
    SELF.sum_gamma_k := SUM(g_dls, sum_gamma_k);
    SELF.sum_lgamma_alpha_k := SUM(g_dls, sum_lgamma_alpha_k);
    SELF.sum_lgamma_gamma_k := SUM(g_dls, sum_lgamma_gamma_k);
    SELF.sum_alpha_k := SUM(g_dls, sum_alpha_k);
    SELF := dl;
  END;
  grp_dlv := ROLLUP(grp_rlv, GROUP, roll_values(LEFT, ROWS(LEFT)));
  doc_level_values := UNGROUP(grp_dlv);
  dlv_rng_lvl := NORMALIZE(doc_level_values, LEFT.num_ranges,
                           TRANSFORM(Doc_Level, SELF:=LEFT));
  //Determine log likelihood  Done in parts.
  //The term digamma refers to the digamma function and lgamma to the log gamma
  // log likelihood = lgamma(SUM(alpha_k)) - SUM(lgamma(alpha_k))         1&2
  //      + SUM((alpha_k-1)(digamma(gamma_k)-digamma(SUM(gamma_k))))      3
  //      + SUM(SUM(phi_nk (digamma(gamma_k)-digamma(SUM(gamma_k)))))     4
  //      + SUM(SUM(SUM(phi_nk w_n_j log(beta_nk))))                      5
  //      - lgamma(SUM(gamma_k)) + SUM(lgamma(gamma_k))                   6, 7
  //      - SUM((gamma_k-1)(digamma(gamma_k)-digamma(SUM(gamma_k))))      8
  //      - SUM(SUM(phi_nk log(phi_nk)))                                  9
  //N.B., we use a compressed vector with count instead of a 0/1 vector
  LogLikelihood range_ll(Doc_Topics d, Doc_level dl) := TRANSFORM
    ASSERT(d.model=dl.model AND d.rid=dl.rid,
           'Key mismatch at step ll1, ' + d.rid + ' : ' + dl.rid, FAIL);
    eq1 := IF(d.topic_range=1, log_gamma(dl.sum_alpha_k), 0.0);
    eq2 := IF(d.topic_range=1, dl.sum_lgamma_alpha_k, 0.0);
    eq3 := SUM(d.t_digammas, (d.alpha-1)*(v-digamma(dl.sum_gamma_k)));
    eq4 := n_phi_gamma_sum(d.t_phis, d.word_counts, d.t_digammas,
                           digamma(dl.sum_gamma_k));
    eq5 := n_phi_beta_sum(d.t_phis, d.word_counts, d.t_logBetas);
    eq6 := IF(d.topic_range=1, log_gamma(dl.sum_gamma_k), 0.0);
    eq7 := SUM(d.t_gammas, log_gamma(v));
    eq8 := SUM(d.t_gammas, ((v-1.0) * (digamma(v) - digamma(dl.sum_gamma_k))));
    eq9 := n_phi_log_phi_sum(d.t_phis, d.word_counts);
    SELF.log_likelihood := eq1 - eq2 + eq3 + eq4 + eq5 - eq6 + eq7 - eq8 - eq9;
    SELF := d;
  END;
  ll_by_range := COMBINE(new_dts, dlv_rng_lvl, range_ll(LEFT,RIGHT), LOCAL);
  grp_ll := GROUP(ll_by_range, model, rid, LOCAL);
  LogLikelihood sum_ll(LogLikelihood ll, DATASET(LogLikelihood) lls):=TRANSFORM
    SELF.log_likelihood := SUM(lls, log_likelihood);
    SELF := ll;
  END;
  grp_ll_doc := ROLLUP(grp_ll, GROUP, sum_ll(LEFT, ROWS(LEFT)));
  ll_doc := UNGROUP(grp_ll_doc);
  ll := NORMALIZE(ll_doc, LEFT.num_ranges,
                  TRANSFORM(LogLikelihood, SELF := LEFT));
  Doc_Topics upd_ll(Doc_Topics dts, LogLikelihood ll) := TRANSFORM
    ASSERT(dts.model=ll.model AND dts.rid=ll.rid,
           'Key mismatch at step ll2, ' + dts.rid + ' : ' + ll.rid, FAIL);
    ll_change := (dts.likelihood-ll.log_likelihood)/dts.likelihood;
    use_chg := dts.likelihood<>0 AND dts.likelihood<ll.log_likelihood;
    SELF.likelihood := ll.log_likelihood;
    SELF.doc_iterations := cnt;
    SELF.likelihood_change := IF(use_chg, ll_change, 2*dts.doc_epsilon);
    SELF := dts;
  END;
  rslt := COMBINE(new_dts, ll, upd_ll(LEFT, RIGHT), LOCAL);
  RETURN rslt;
END;
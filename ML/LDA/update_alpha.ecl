// From Blei et al.
IMPORT $;
IMPORT Types FROM $;
// Work functions
REAL8 d_alhood(REAL8 a, REAL8 ss, UNSIGNED4 D, UNSIGNED4 K)
    := (D * (K * digamma(K * a) - K * digamma(a)) + ss);

REAL8 d2_alhood(REAL8 a, UNSIGNED4 D, UNSIGNED4 K)
    := (D * (K * K * trigamma(K * a) - K * trigamma(a)));

Types.Alpha_Estimate upd(Types.Alpha_Estimate ae) := TRANSFORM
  REAL8 log_alpha_in := LN(ae.alpha);
  REAL8 df := d_alhood(ae.alpha, ae.suff_stat, ae.docs, ae.num_topics);
  REAL8 d2f := d2_alhood(ae.alpha, ae.docs, ae.num_topics);
  REAL8 log_a := log_alpha_in - df/(d2f * ae.alpha + df);
  REAL8 new_alpha := EXP(log_a);
  SELF.init_alpha := IF(isValid(new_alpha), ae.init_alpha, 10*ae.init_alpha);
  SELF.iter := ae.iter + 1;
  SELF.last_df := IF(isValid(new_alpha), df, 0.1);
  SELF.alpha := IF(isValid(new_alpha), new_alpha, 10*ae.init_alpha);
  SELF := ae;
END;

EXPORT update_alpha(DATASET(Types.Alpha_Estimate) ds) := PROJECT(ds, upd(LEFT));

//SUM((gamma_k-1)(digamma(gamma_k)-digamma(SUM(gamma_k))))
IMPORT ML.LDA;
IMPORT ML.LDA.Types AS Types;
// aliases for convenience
Topic_Values := Types.Topic_Values;
Topic_Value := Types.Topic_Value;
TermFreq := Types.TermFreq;
OnlyValue := Types.OnlyValue;

EXPORT REAL8 gamma_digamma_sum(Types.Topic_Value_DataSet t_gammas,
                               REAL8 gamma_sum) := FUNCTION
  OnlyValue term(Topic_Value gamma) := TRANSFORM
    SELF.v := (gamma.v - 1) * (digamma(gamma.v) - digamma(gamma_sum));
  END;
  rslt := SUM(PROJECT(t_gammas, term(LEFT)), v);
  RETURN rslt;
END;
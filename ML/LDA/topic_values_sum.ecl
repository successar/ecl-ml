//Sum the values by nominal over the topics
IMPORT $.Types;
EXPORT Types.OnlyValue_DataSet
       topic_values_sum(Types.Topic_Values_DataSet tvs) := BEGINC++
  #ifndef ECL_LDA_ONLYVALUE
  #define ECL_LDA_ONLYVALUE
  typedef  struct __attribute__ ((__packed__))  LDAOnlyValue {
    double v;
  };
  #endif
  #ifndef ECL_LDA_TOPIC_VALUES
  #define ECL_LDA_TOPIC_VALUES
  typedef  struct __attribute__ ((__packed__))  LDATopicValues {
    uint32_t topic;
    size32_t sz_vs;   //array of Only Values follows of size sz_vs
  };
  #endif
  #body
  const LDATopicValues* in_tvs = (LDATopicValues*) tvs;
  const LDAOnlyValue* in_vs;
  size32_t consumed = 0;
  size32_t vs_size = in_tvs->sz_vs;
  size32_t fx_size = sizeof(LDATopicValues);
  uint32_t num_words = in_tvs->sz_vs / sizeof(LDAOnlyValue);
  __lenResult = in_tvs->sz_vs;
  __result = rtlMalloc(__lenResult);
  LDAOnlyValue* tv_sums = (LDAOnlyValue*) __result;
  uint32_t word;
  for (word=0; word<num_words; word++) tv_sums[word].v = 0.0;
  while (consumed<lenTvs) {
    in_tvs = (LDATopicValues*)(tvs + consumed);
    in_vs = (LDAOnlyValue*)(tvs + consumed + fx_size);
    if (in_tvs->sz_vs!=vs_size) rtlFail(0,"Variable size source");
    for (word=0; word<num_words; word++) tv_sums[word].v += in_vs[word].v;
    consumed += fx_size + vs_size;
  }
ENDC++;
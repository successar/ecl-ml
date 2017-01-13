// Sum the values.  A vector sum
IMPORT $.Types;
EXPORT Types.OnlyValue_DataSet
        value_sum(Types.OnlyValue_DataSet a1,
                  Types.OnlyValue_Dataset a2) := BEGINC++
  #ifndef ECL_LDA_ONLYVALUE
  #define ECL_LDA_ONLYVALUE
  typedef  struct __attribute__ ((__packed__))  tagLDAOnlyValue {
    double v;
  } LDAOnlyValue;
  #endif
  #body
  if (lenA1!=lenA2) rtlFail(0,"Different length vectors provided");
  __lenResult = lenA1;
  __result = rtlMalloc(lenA1);
  const LDAOnlyValue* in_a1 = (LDAOnlyValue*) a1;
  const LDAOnlyValue* in_a2 = (LDAOnlyValue*) a2;
  LDAOnlyValue* rslt = (LDAOnlyValue*) __result;
  uint32_t num_words = lenA1 / sizeof(LDAOnlyValue);
  for (uint32_t word=0; word<num_words; word++) {
    rslt[word].v = in_a1[word].v + in_a2[word].v;
  }
ENDC++;
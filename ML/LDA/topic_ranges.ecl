//
factor := 100000; // 5000 can be used to force multiple ranges with Blei data
EXPORT topic_ranges(UNSIGNED4 num_topics, UNSIGNED4 terms) := MODULE
  SHARED num_entries := num_topics * terms;
  EXPORT ranges := (num_entries + factor - 1) DIV factor;
  EXPORT per_range := (num_topics + ranges - 1) DIV ranges;
END;
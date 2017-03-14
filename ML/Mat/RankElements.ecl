IMPORT ML;

EXPORT RankElements(DATASET(ML.Mat.Types.Element) A, UNSIGNED groups = 0) := FUNCTION
        sortedA := SORT(A, value);

        N := ML.Mat.Has(sortedA).Stats.NElements;
        fgroups := IF(groups = 0, N, groups);

        RETURN PROJECT(
                sortedA,
                TRANSFORM(
                        ML.Mat.Types.Element,
                        SELF.value := TRUNCATE(COUNTER * fgroups / (N + 1)),
                        SELF := LEFT
                )
        );
END;

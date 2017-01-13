IMPORT TestingSuite.Utils.Types AS UTypes;
T_Test_Distribution_Table := DATASET([
{1,12.706}, {2,4.303}, {3,3.182},{4,2.776},
{5,2.571}, {6,2.447}, {7,2.365}, {8,2.306},
{9,2.262}, {10,2.228}, {11,2.201}, {12,2.179},
{13,2.16}, {14,2.145}, {15,2.131}, {16,2.12},
{17,2.11}, {18,2.101}, {19,2.093}, {20,2.086},
{21,2.08}, {22,2.074}, {23,2.069}, {24,2.064},
{25,2.06}, {26,2.056}, {27,2.052}, {28,2.048},
{29,2.045}, {30,2.042}
], UTypes.t_test_distribution_table_rec);

EXPORT STRING Student_T_Test(DATASET(UTypes.result_rec) A, DATASET(UTypes.result_rec) B) := FUNCTION 
    INTEGER degree_of_freedom := COUNT(A) + COUNT(B) - 2;
    INTEGER number_of_observation_A := COUNT(A);
    INTEGER number_of_observation_B := COUNT(B);
    REAL mean_A := AVE(A, scores);
    REAL mean_B := AVE(B, scores);
    REAL variance_A := VARIANCE(A, scores);
    REAL variance_B := VARIANCE(B, scores);
    REAL T_Measure := ABS(mean_A - mean_B) / SQRT((variance_A/number_of_observation_A) + (variance_B/number_of_observation_B));
    REAL T_alpha := T_Test_Distribution_Table(degree_of_freedom = degree_of_freedom)[1].t_value;
    RETURN IF(T_Measure >  T_alpha, 'FAIL', 'PASS');
END;
/*
// Example
test_A := DATASET(
[{1, 42.1},{1, 41.3},
{1, 42.4},{1, 43.2},
{1, 41.8},{1, 41.0},
{1, 41.8},{1, 42.8},
{1, 42.3},{1, 42.7}]
, UTypes.result_rec);

test_B := DATASET(
[{1, 42.7},{1, 43.8},
{1, 42.5},{1, 43.1},
{1, 44.0},{1, 43.6},
{1, 43.3},{1, 43.5},
{1, 41.7},{1, 44.1}]
, UTypes.result_rec);

OUTPUT(Student_T_Test(test_A, test_B));*/
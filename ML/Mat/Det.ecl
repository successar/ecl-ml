IMPORT ML.Mat AS ML_Mat;
IMPORT ML.Mat.Types AS Types;

EXPORT Det(DATASET(Types.Element) matrix) := 
        AGGREGATE(ML_Mat.Decomp.LU(matrix)(x=y), Types.Element, TRANSFORM(Types.Element, SELF.value := IF(RIGHT.x<>0,LEFT.Value*RIGHT.Value,LEFT.Value), SELF := LEFT),
				TRANSFORM(Types.Element, SELF.value := RIGHT1.Value*RIGHT2.Value, SELF := RIGHT2))[1].value;
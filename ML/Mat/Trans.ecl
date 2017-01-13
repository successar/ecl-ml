IMPORT ML.Mat AS ML_Mat;
IMPORT ML.Mat.Types AS Types;

EXPORT Trans(DATASET(Types.Element) d) := PROJECT(d,TRANSFORM(Types.Element, SELF.x := LEFT.y, SELF.y := LEFT.x, SELF := LEFT));

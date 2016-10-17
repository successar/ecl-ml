IMPORT ML.Mat AS ML_Mat;
IMPORT ML.Mat.Types AS Types;
// the lower triangular portion of the matrix
EXPORT LowerTriangle(DATASET(Types.Element) matrix) := matrix(x>=y);

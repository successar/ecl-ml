IMPORT ML.Mat AS ML_Mat;
IMPORT ML.Mat.Types AS Types;

EXPORT SetDimension(DATASET(Types.Element) A, Types.t_Index I, Types.t_Index J) := IF( ML_Mat.Strict,
	IF(EXISTS(A(x=I,y=J)), A(x<=I,y<=J), A(x<=I,y<=J)+DATASET([{I,J,0}], Types.Element)),A);
IMPORT ML.Mat AS ML_Mat;
IMPORT ML.Mat.Types AS Types;
 
Inverse(DATASET(Types.Element) matrix) := FUNCTION
	dim := ML_Mat.Has(matrix).Dimension;
	mLU := ML_Mat.Decomp.LU(matrix);
	L := ML_Mat.Decomp.LComp(mLU);
	U := ML_Mat.Decomp.UComp(mLU);
	mI := ML_Mat.Identity(dim);
	fsub := ML_Mat.Decomp.f_sub(L, mI);
	matrix_inverse := ML_Mat.Decomp.b_sub(u, fsub);
	RETURN matrix_inverse;
END;
EXPORT Inv(DATASET(Types.Element) matrix) := IF(ML_Mat.Det(matrix)=0, DATASET([],Types.Element), Inverse(matrix) );

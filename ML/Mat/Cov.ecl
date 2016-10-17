IMPORT ML.Mat as ML_Mat;
/*
	http://en.wikipedia.org/wiki/Covariance_matrix

 Covariance matrix (also known as dispersion matrix) is a matrix whose 
 element in the i, j position is the covariance between the ith and jth 
 columns of the original matrix.
*/
EXPORT Cov(DATASET(Types.Element) A) :=  FUNCTION

	ZeroMeanA := ML_Mat.Sub(A, Repmat(ML_Mat.Has(A).MeanCol, ML_Mat.Has(A).Stats.XMax, 1));

	SF := 1/(ML_Mat.Has(A).Stats.XMax-1);
	RETURN ML_Mat.Scale(ML_Mat.Mul(ML_Mat.Trans(ZeroMeanA),ZeroMeanA, 2), SF);
END;
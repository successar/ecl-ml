IMPORT ML.Config;
IMPORT ML.Mat as ML_Mat;
IMPORT ML.Mat.Types AS Types;

/*
	http://en.wikipedia.org/wiki/Eigendecomposition_of_a_matrix
  A (non-zero) vector v of dimension N is an eigenvector of a square (N×N) matrix A if and only 
	if it satisfies the linear equation A*v = lambda*v where lambda is a scalar, termed the eigenvalue 
	corresponding to v. That is, the eigenvectors are the vectors which the linear transformation 
	A merely elongates or shrinks, and the amount that they elongate/shrink by is the eigenvalue. 
	The above equation is called the eigenvalue equation or the eigenvalue problem.

*/
EXPORT eig(DATASET(Types.Element) A, UNSIGNED4 iter=200) := MODULE
SHARED eig_comp := ENUM ( T = 1, Q = 2, conv = 3 );
EXPORT DATASET(Types.Element) QRalgorithm() := FUNCTION

		Q0 := ML_Mat.Decomp.QComp(A);
		R0 := ML_Mat.Decomp.RComp(A);
		T0 := ML_Mat.Mul(R0, Q0);
		Conv0 := DATASET([{1,1,0}],ML_Mat.Types.Element);
		
	loopBody(DATASET( Types.MUElement) ds, UNSIGNED4 k) := FUNCTION

		T := ML_Mat.MU.From(ds, eig_comp.T);	
		Q := ML_Mat.MU.From(ds, eig_comp.Q);
		Conv := ML_Mat.MU.From(ds, eig_comp.conv);

		bConverged:= ML_Mat.Vec.Norm(Vec.FromDiag(T,-1))<Config.RoundingError;
		
		QComp := ML_Mat.Decomp.QComp(T);
		Q1 := ML_Mat.Thin(Mul(Q,QComp));
		RComp := ML_Mat.Decomp.RComp(T);
		T1 := ML_Mat.Thin(Mul(RComp, QComp));
		Conv1 :=  PROJECT(Conv,TRANSFORM(Types.Element,SELF.value:=k, SELF := LEFT));

	RETURN IF(bConverged, ds, ML_Mat.MU.To(T1, eig_comp.T)+MU.To(Q1, eig_comp.Q)+ML_Mat.MU.To(Conv1, eig_comp.conv));
  END;
	
	RETURN LOOP(ML_Mat.Mu.To(T0, eig_comp.T)+Mu.To(Q0, eig_comp.Q)+Mu.To(Conv0, eig_comp.conv), iter, loopBody(ROWS(LEFT),COUNTER));
END;

EXPORT valuesM := ML_Mat.MU.From(QRalgorithm(), eig_comp.T);
EXPORT valuesV := ML_Mat.Vec.FromDiag(valuesM());
EXPORT vectors := ML_Mat.MU.From(QRalgorithm(), eig_comp.Q);
EXPORT convergence := ML_Mat.MU.From(QRalgorithm(), eig_comp.conv)[1].value;

END;
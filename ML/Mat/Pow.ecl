IMPORT ML.Mat AS ML_Mat;
IMPORT ML.Mat.Types AS Types;

// Raise the matrix d to the power of n (which must be positive)
EXPORT Pow(DATASET(Types.Element) d,UNSIGNED2 n) := FUNCTION

// Strategy: create a MU - with matrix 1 the target and matrix 2 the multiplier - perform the multiplication n-1 times
  m := ML_Mat.MU.To(d,1)+ML_Mat.MU.To(d,2);
	mult(DATASET(Types.MUElement) c) := FUNCTION
	  prod := ML_Mat.Mul( ML_Mat.MU.From(c,1), ML_Mat.MU.From(C,2) );
		RETURN c(no=2)+ML_Mat.MU.To(Prod,1);
	END;

	multi := LOOP( m, n-1, mult(ROWS(LEFT)) );

  RETURN IF ( n = 1, d, ML_Mat.MU.From(multi,1) );
	
  END;
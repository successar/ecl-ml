IMPORT ML.Mat AS ML_Mat;
IMPORT ML.Mat.Types AS Types;
IMPORT ML.Config;
// Thin down a sparse matrix by removing any elements which are now 0, while preserving the dimension element
// Encapsulated here in case we eventually want to incorporate some form of error-term in the 0 test
EXPORT Thin(DATASET(Types.Element) d) := FUNCTION
			Dim := ML_Mat.Has(d).Stats;
			thinD := d(ABS(Value) > Config.RoundingError );
      RETURN ML_Mat.SetDimension(thinD, Dim.XMax, Dim.YMax);
END;

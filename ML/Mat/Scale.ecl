IMPORT ML.Mat AS ML_Mat;
IMPORT ML.Mat.Types AS Types;
EXPORT Scale(DATASET(Types.Element) d,Types.t_Value factor) := FUNCTION
  Types.Element mul(d le) := TRANSFORM
	  SELF.value := le.value * factor;
	  SELF := le;
	END;
	RETURN PROJECT(d,mul(LEFT));
  END;
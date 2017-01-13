// Element-wise matrix operations
IMPORT ML.Mat AS ML_Mat;
IMPORT ML.Mat.Types AS Types;
EXPORT Each := MODULE

EXPORT Each_Sqrt(DATASET(Types.Element) d) := FUNCTION
  Types.Element t_sqrt_each_fn(d le) := TRANSFORM
	  SELF.value := SQRT(le.value);
	  SELF := le;
	END;
	RETURN PROJECT(d,t_sqrt_each_fn(LEFT));
END;
	
EXPORT Each_Exp(DATASET(Types.Element) d) := FUNCTION
  Types.Element t_exp_each_fn(d le) := TRANSFORM
	  SELF.value := exp(le.value);
	  SELF := le;
	END;
	RETURN PROJECT(d,t_exp_each_fn(LEFT));
END;

EXPORT Each_Abs(DATASET(Types.Element) d) := FUNCTION
  Types.Element t_abs_each_fn(d le) := TRANSFORM
	  SELF.value := Abs(le.value);
	  SELF := le;
	END;
	RETURN PROJECT(d,t_abs_each_fn(LEFT));
END;

EXPORT Each_Mul(DATASET(Types.Element) l,DATASET(Types.Element) r) := FUNCTION
// Only slight nastiness is that these matrices may be sparse - so either side could be null
Types.Element t_each_Multiply(l le,r ri) := TRANSFORM
    SELF.x := le.x ;
    SELF.y := le.y ;
	  SELF.value := le.value * ri.value; 
  END;
	RETURN JOIN(l,r,LEFT.x=RIGHT.x AND LEFT.y=RIGHT.y,t_each_Multiply(LEFT,RIGHT));
END;


// matrix .+ scalar
EXPORT Each_Add(DATASET(Types.Element) d,Types.t_Value scalar) := FUNCTION
  Types.Element t_each_add(d le) := TRANSFORM
	  SELF.value := le.value + scalar;
	  SELF := le;
	END;
	RETURN PROJECT(d,t_each_add(LEFT));
END;

/*
 factor ./ matrix	; 
*/	
EXPORT Each_Reciprocal(DATASET(Types.Element) d, Types.t_Value factor=1) := FUNCTION
  Types.Element t_each_divide(d le) := TRANSFORM
	  SELF.value := factor / le.value;
	  SELF := le;
	END;
	RETURN PROJECT(d,t_each_divide(LEFT));
 END;

END;
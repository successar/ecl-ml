IMPORT ML.Mat AS ML_Mat;
EXPORT Eq(DATASET(Types.Element) l,DATASET(Types.Element) r) := FUNCTION
        lt := ML_Mat.Thin(l);
	rt := ML_Mat.Thin(r);
	RETURN COUNT(lt)=COUNT(rt) AND ~EXISTS(Thin(Sub(lt,rt)));
  END;
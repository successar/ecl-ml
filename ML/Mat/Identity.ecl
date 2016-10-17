IMPORT ML.Mat AS ML_Mat;

EXPORT Identity(UNSIGNED4 dimension) := ML_Mat.Vec.ToDiag( ML_Mat.Vec.From(dimension,1.0) );
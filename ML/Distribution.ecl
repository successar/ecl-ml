IMPORT ML;
IMPORT ML.Types AS Types;
IMPORT ML.Mat;
IMPORT ML.Mat.Vec AS Vec;
IMPORT ML.Utils AS Utils;
// This module exists to provide routines to support the various statistical distribution functions that exist
EXPORT Distribution := MODULE

// Used for storing a vector of probabilities (usually cumulative)
EXPORT Layout := RECORD
  Types.t_Count RangeNumber;
  Types.t_FieldReal RangeLow; // Values > RangeLow
  Types.t_FieldReal RangeHigh; // Values <= RangeHigh
  Types.t_FieldReal P;
  END;
                                            

// A quick way to get a density vector with everything but the density filled in
// Cannot really move into the base class as the point to pass into the density function can vary from distribution to distribution
SHARED DVec(UNSIGNED Ranges,Types.t_FieldReal low,Types.t_FieldReal width) :=
    PROJECT(Vec.From(Ranges),TRANSFORM(Layout,SELF.RangeNumber:=LEFT.x,
                                              SELF.RangeLow := low+width*(LEFT.x-1),
                                              SELF.RangeHigh := low+width*LEFT.x,
                                              SELF.P := 0));

// NRanges is the number of divisions to split the distribution into  
EXPORT Default := MODULE,VIRTUAL
  EXPORT RangeWidth := 1.0; // Only really works for discrete - the width of each range
  EXPORT Types.t_FieldReal Density(Types.t_FieldReal RH) := 0.0; // Density function at stated point
  // Generating functions are responsible for making these in ascending order
  EXPORT DensityV() := DATASET([],Layout); // Probability of between >PreviosRangigh & <= RangeHigh
  // Default CumulativeV works by simple integration of the DensityVec
  EXPORT CumulativeV() := FUNCTION
    d := DensityV();
    Layout Accum(Layout le,Layout ri) := TRANSFORM
      SELF.p := le.p+ri.p*RangeWidth;
      SELF := ri;
    END;
    RETURN ITERATE(d,Accum(LEFT,RIGHT)); // Global iterates are horrible - but this should be tiny
  END;
  // Default Cumulative works from the Cumulative Vector
  EXPORT Types.t_FieldReal Cumulative(Types.t_FieldReal RH) :=FUNCTION // Cumulative probability at stated point
    cv := CumulativeV();
    // If the range high value is at an intermediate point of a range then interpolate the result\
    // Interpolation done as follows :
    // cumulative(RH) = cumulative(v.RangeHigh) - prob(RH <= x < v.Rangehigh)
    // prob(RH <= x < v.Rangehigh) =  Density((RH + Rangehigh)/2) * (RH - Rangehigh) [Rectangle Rule for integration]
    InterC(Layout v) := IF ( RH=v.RangeHigh, v.P, v.P - Density((v.RangeHigh+RH)/2)*(v.RangeHigh - RH));
    RETURN MAP( RH >= MAX(cv,RangeHigh) => 1.0,
                RH <= MIN(cv,RangeLow) => 0.0,
                InterC(cv(RH>RangeLow,RH<=RangeHigh)[1]) );
  END;
  // Default NTile works from the Cumulative Vector
  EXPORT Types.t_FieldReal NTile(Types.t_FieldReal Pc) :=FUNCTION // Value of the Pc percentile
    cp := Pc / 100.0; // Convert from percentile to cumulative probability
    cv := CumulativeV();
    // If the range high value is at an intermediate point of a range then interpolate the result
    InterP(Layout v) := IF ( cp=v.P, v.RangeHigh, v.RangeHigh+(cp-v.p)/Density((v.RangeHigh+v.RangeLow)/2) );
    RETURN MAP( cp >= MAX(cv,P) => MAX(cv,RangeHigh),
                cp <= 0.0 => MIN(cv,RangeLow),
                InterP(cv(P>=cp)[1]) );
  END;
  EXPORT InvDensity(Types.t_FieldReal delta) := 0.0; //Only sensible for monotonic distributions
  EXPORT Discrete := FALSE;
  END;

// Assume probabilities are uniformly distributed across low/high 
EXPORT Uniform(Types.t_FieldReal low,Types.t_FieldReal high,Types.t_Count NRanges = 10000) := MODULE(Default)
  EXPORT Types.t_FieldReal RangeWidth := (high-low)/NRanges;
  EXPORT Types.t_FieldReal Density(Types.t_FieldReal RH) := IF (RH >= low AND RH <= high,1/NRanges,0);
  EXPORT Types.t_FieldReal Cumulative(Types.t_FieldReal RH) := MAP ( RH <= low => 0,
                                                    RH >= high => 1,
                                                    (RH-low) / (high-low) );

  EXPORT DensityV() := PROJECT(DVec(Nranges,low,RangeWidth),
                         TRANSFORM(Layout,
                           SELF.P := Density( (LEFT.RangeLow+LEFT.RangeHigh)/2 ),
                           SELF := LEFT)); // Take density from mid-point
  EXPORT CumulativeV() := PROJECT(DVec(NRanges,low,RangeWidth),
                            TRANSFORM(Layout,SELF.P := Cumulative(LEFT.RangeHigh),SELF := LEFT)); 
  END;

// A normal distribution with mean 'mean' and standard deviation 'sd'
EXPORT Normal(Types.t_FieldReal mean,Types.t_FieldReal sd,Types.t_Count NRanges = 10000) := MODULE(Default)
  SHARED Types.t_FieldReal Var := sd*sd;
  // Assume the range of a normal function is 4 sd above and below the mean
  SHARED low := mean-4*sd;
  SHARED high := mean+4*sd;
  EXPORT RangeWidth := (high-low)/NRanges;
  // Standard definition of normal probability density function
  SHARED fn_Density(Types.t_FieldReal x,Types.t_FieldReal mu,Types.t_FieldReal sig_sq) := EXP( - POWER(x-mu,2)/(2*sig_sq) )/SQRT(2*Utils.Pi*Sig_Sq);
  EXPORT Types.t_FieldReal Density(Types.t_FieldReal RH) := fn_Density(RH,Mean,Var);

  // using the Abromowitz and Stegun Approximation to the CDF; accurate to 7 decimal places
  EXPORT Types.t_FieldReal Cumulative(Types.t_FieldReal RH) := FUNCTION
    // Need to convert to the equivalent in the N(0,1) realm
    REAL Scaled := ABS(RH-mean)/sd;
    REAL8 t := 1 / (1+0.2316419 * Scaled);
    P := 1 - fn_Density(Scaled,0,1)*(0.319381530*t-0.356563782*t*t+1.781477937*POWER(t,3)-1.821255978*POWER(t,4)+1.330274429*POWER(t,5));
    RETURN IF( RH < Mean,1 - P, P );
  END;
  EXPORT DensityV() := PROJECT(DVec(NRanges,low,RangeWidth),
                         TRANSFORM(Layout,
                           SELF.P := Density((LEFT.RangeLow+LEFT.RangeHigh)/2),
                           SELF := LEFT));
  EXPORT CumulativeV() := PROJECT(DVec(NRanges,low,RangeWidth),
                            TRANSFORM(Layout,SELF.P := Cumulative(LEFT.RangeHigh),SELF := LEFT)); 

  END;

// The normal using the default integration model for comparison purposes to the Abromowitz & Stegun Result
EXPORT Normal2(Types.t_FieldReal mean,Types.t_FieldReal sd,Types.t_Count NRanges = 10000) := MODULE(Default)
  SHARED Types.t_FieldReal Var := sd*sd;
  // Assume the range of a normal function is 4 sd above and below the mean
  SHARED low := mean-4*sd;
  SHARED high := mean+4*sd;
  EXPORT RangeWidth := (high-low)/NRanges;
  // Standard definition of normal probability density function
  SHARED fn_Density(Types.t_FieldReal x,Types.t_FieldReal mu,Types.t_FieldReal sig_sq) := EXP( - POWER(x-mu,2)/(2*sig_sq) )/SQRT(2*Utils.Pi*Sig_Sq);
  EXPORT Types.t_FieldReal Density(Types.t_FieldReal RH) := fn_Density(RH,Mean,Var);
  EXPORT DensityV() := PROJECT(DVec(NRanges,low,RangeWidth),
                         TRANSFORM(Layout,
                           SELF.P := Density((LEFT.RangeLow+LEFT.RangeHigh)/2),
                           SELF := LEFT));
  END;

// Student T distribution
// This distribution is entirely symmetric about the mean - so we will model the >= 0 portion
// Warning - v=1 tops out around the 99.5th percentile, 2+ is fine
EXPORT StudentT(Types.t_Discrete v,Types.t_Count NRanges = 10000) := MODULE(Default)
  SHARED Multiplier := IF ( v & 1 = 0, Utils.DoubleFac(v-1)/(2*SQRT(v)*Utils.DoubleFac(v-2))
                                     , Utils.DoubleFac(v-1)/(Utils.Pi*SQRT(v)*Utils.DoubleFac(v-2)));
  // Compute the value of t for which a given density is obtained                   
  SHARED LowDensity := 0.00001; // Go down as far as a density of 1x10-5
  EXPORT InvDensity(Types.t_FieldReal delta) := SQRT(v*(EXP(LN(delta/Multiplier)*-2.0/(v+1))-1));
  // We are defining a high value as the value at which the density is 'too low to care'
  SHARED high := InvDensity(LowDensity);
  SHARED Low := 0;
  EXPORT RangeWidth := (high-low)/NRanges;
  EXPORT Types.t_FieldReal Density(Types.t_FieldReal RH) := Multiplier * POWER( 1+RH*RH/v,-0.5*(v+1) );
  EXPORT DensityV() := PROJECT(DVec(NRanges,low,RangeWidth),
                         TRANSFORM(Layout,
                           SELF.P := Density((LEFT.RangeLow+LEFT.RangeHigh)/2),
                           SELF := LEFT));  
  EXPORT CumulativeV() := FUNCTION
    d := DensityV();
    // The general integration really doesn't work for v=1 and v=2 - fortunately there are 'nice' closed forms for the CDF for those values of v
    Layout Accum(Layout le,Layout ri) := TRANSFORM
      SELF.p := MAP( v = 1 => 0.5+ATAN(ri.RangeHigh)/Utils.Pi, // Special case CDF for v = 1
                     v = 2 => (1+ri.RangeHigh/SQRT(2+POWER(ri.RangeHigh,2)))/2, // Special case of CDF for v=2
                     IF(le.p=0,0.5,le.p)+ ri.p*RangeWidth );
      SELF := ri;
    END;
    RETURN ITERATE(d,Accum(LEFT,RIGHT)); // Global iterates are horrible - but this should be tiny
  END;
END;

// The exponential (sometimes called negative exponential) distribution
// Nice easy math - all distributions should be made this way ....
EXPORT Exponential(Types.t_FieldReal lamda,Types.t_Count NRanges = 10000) := MODULE(Default)
  EXPORT Types.t_FieldReal Density(Types.t_FieldReal RH) := IF ( RH < 0, 0, lamda * EXP(-lamda*RH));
  EXPORT Types.t_FieldReal Cumulative(Types.t_FieldReal RH) := IF ( RH < 0, 0, 1-EXP(-lamda*RH));
  EXPORT NTile(Types.t_FieldReal pc) := LN(1-(pc/100))/-lamda;
  SHARED High := NTile(99.999);
  EXPORT RangeWidth := high/NRanges;
  EXPORT DensityV() := PROJECT(DVec(NRanges,0,RangeWidth),
                         TRANSFORM(Layout,SELF.P := Density((LEFT.RangeHigh+LEFT.RangeLow)/2), SELF := LEFT));
  EXPORT CumulativeV() := PROJECT(DVec(NRanges,0,RangeWidth),
                         TRANSFORM(Layout,SELF.P := Cumulative(LEFT.RangeHigh), SELF := LEFT));
  END;

// Forms the distribution of getting k successes out of NRanges-1 trials
// where the chances of any one trial being a success is p
EXPORT Binomial(Types.t_FieldReal p,Types.t_Count NRanges = 100) := MODULE(Default)
  // This has to take a real for 'derivation' reasons - but range-high must be an 'integer' from 0->NRanges-1 to be meaningful
  EXPORT Types.t_FieldReal Density(Types.t_FieldReal RH) := 
    IF( RH > NRanges-1 OR RH < 0,0, Utils.NCK(Nranges-1,RH)*POWER(p,RH)*POWER(1-p,(NRanges-1)-RH));
  EXPORT DensityV() := PROJECT(DVec(NRanges,-1,1),
                         TRANSFORM(Layout,SELF.P := Density(LEFT.RangeHigh), SELF := LEFT));
  // Uses the 'default' integration module to construct the cumulative values
  EXPORT Discrete := TRUE;
  END;

// Forms the distribution of the number of successes likely to have occured with success probability p
// before 'failures' number of failures have occured.
// To keep the ranges integral - NRanges is defined as an upper-bound on the number of events that will be attempted
EXPORT NegBinomial(Types.t_FieldReal p,Types.t_Count Failures, Types.t_Count NRanges = 1000) := MODULE(Default)
  EXPORT Types.t_FieldReal Density(Types.t_FieldReal RH) := 
    IF( RH > NRanges-1 OR RH < 0,0, Utils.NCK(RH+Failures-1,RH)*POWER(p,RH)*POWER(1-p,Failures));
  EXPORT DensityV() := PROJECT(DVec(NRanges,-1,1),
                         TRANSFORM(Layout,SELF.P := Density(LEFT.RangeHigh), SELF := LEFT));
  // Uses the 'default' integration module to construct the cumulative values
  EXPORT Discrete := TRUE;
  END;



// A poisson distribution has a mean and variance characterized by lamda
// Lamda need not be an integer although it is used to produce probabilities for a count of discrete events
// It speaks to the question - given I fall over on average 1.6 times a day; what are the chances of me falling over twice today?
// In particular NRanges is a little different for this distribution - it assumes we want probabilities for between 0 and n-1 events
EXPORT Poisson(Types.t_FieldReal lamda,Types.t_Count NRanges = 100) := MODULE(Default)
  // This has to take a real for 'derivation' reasons - but range-high must be an 'integer' from 0->NRanges-1 to be meaningful
  EXPORT Types.t_FieldReal Density(Types.t_FieldReal RH) := EXP(-lamda)*POWER(lamda,RH)/Utils.Fac(RH);
  EXPORT DensityV() := PROJECT(DVec(NRanges,-1,1),
                         TRANSFORM(Layout,SELF.P := Density(LEFT.RangeNumber-1), SELF := LEFT));
  // Uses the 'default' integration module to construct the cumulative values
  EXPORT Discrete := TRUE;
  END;
  
EXPORT FDist(Types.t_Discrete d1, Types.t_Discrete d2, Types.t_Count NRanges = 10000) := MODULE(Default)
  SHARED Multiplier := (1 / Utils.Beta(d1/2, d2/2)) * POWER(d1/d2, d1/2);
  SHARED high := 15;
  SHARED Low := 0;
  EXPORT RangeWidth := (high - low)/NRanges;
  EXPORT Types.t_FieldReal Density(Types.t_FieldReal RH) := Multiplier * POWER(RH, d1/2 - 1) / POWER(1 + d1 * RH/d2, (d1 + d2)/2);
  EXPORT DensityV() := PROJECT(DVec(NRanges,low,RangeWidth),
                         TRANSFORM(Layout,
                           SELF.P := Density((LEFT.RangeLow+LEFT.RangeHigh)/2),
                           SELF := LEFT));  
END;

EXPORT ChiSquare(Types.t_Discrete d1, Types.t_Count NRanges = 10000) := MODULE(Default)
  SHARED Multiplier := 1 / (POWER(2, d1/2) * Utils.gamma(d1/2));
  // Hish value calculated using Normal Approximation to Chi Square Distribution (6-sigma)
  SHARED high := POWER(6 * SQRT(2/(9*d1)) + (1 - (2/9*d1)), 2) * d1;
  SHARED Low := 0;
  EXPORT RangeWidth := (high - low)/NRanges;
  EXPORT Types.t_FieldReal Density(Types.t_FieldReal RH) := Multiplier * POWER(RH, d1/2 - 1) * EXP(-RH/2);
  EXPORT DensityV() := PROJECT(DVec(NRanges,low,RangeWidth),
                         TRANSFORM(Layout,
                           SELF.P := Density((LEFT.RangeLow+LEFT.RangeHigh)/2),
                           SELF := LEFT));  
  EXPORT Types.t_FieldReal Cumulative(Types.t_FieldReal RH) := Utils.lowerGamma(d1/2, RH/2) / Utils.Gamma(d1/2);
END;

// Generate N records (with record ID 1..N)
// Fill in the field 'fld'
// The value will be a random variable from the distribution 'dist'
EXPORT GenData(Types.t_RecordID N,Default dist,Types.t_FieldNumber fld = 1) := FUNCTION
// The presumption here is that N is very high (at least when performance hurts) - therefore significant pre-computation
// can be tolerated to ensure that the actual construction of the random variable is as quick as possible
// The method is as follows - use the Cumulative Probability Vector to construct an extended vector that gives:
// The high & low bounds on the value range for every record
// The high & low bounds on the cumulative probability range for every record
// We then use a random integer % 1M / 1M to give us a cumulative probability point
// We then use the probability ranges (/value ranges) and linear interpolation to provide the datapoint
  CV := dist.CumulativeV();
  Buckets := 10000; 
  t_Bucket := UNSIGNED2;
  R := RECORD
    CV;
    REAL8 LowP := 0;
    t_Bucket PBucket := CV.P*Buckets; // Will be used to turn a ,ALL join into a ,LOOKUP join later
  END;
  CVR := TABLE(CV,R);
  MaxP := MAX(CV,P); // Allows for some of the integration methods not quite reaching a CP of 1
  // Build up the range low and plow values
  R CopyLow(CVR le,CVR ri) := TRANSFORM
    SELF.LowP := le.P;
    SELF := ri;
  END;
  I := ITERATE(CVR,CopyLow(LEFT,RIGHT));
  // Now we potentially bulk up the data a little so that there is a record for every range for every percentile of probability
  R Bulk(R le,t_Bucket C) := TRANSFORM
    SELF.PBucket := le.PBucket-C;
    SELF := le;
  END;
  No := NORMALIZE(I,1+(UNSIGNED)(LEFT.P*Buckets)-(UNSIGNED)(LEFT.LowP*Buckets),Bulk(LEFT,COUNTER-1));
  // Now construct the result vector - first just create the random vector uniformly distributed in the 0-<1 range
  B1 := 1000000;
  V := PROJECT( Vec.From(N),TRANSFORM(Types.NumericField,SELF.Id := LEFT.x, SELF.Value := (RANDOM()%B1) / (REAL8)B1,SELF.number:=fld));
  // Now join to the distribution; interpolating between the specified ranges
  Types.NumericField Trans(Types.NumericField le,No ri) := TRANSFORM
    SELF.value := IF ( Dist.Discrete OR ri.P=ri.LowP, ri.RangeHigh, ri.RangeLow+(ri.RangeHigh-ri.RangeLow)*(le.value-ri.LowP)/(ri.P-ri.LowP) );
    SELF := le;
  END;
  J := JOIN(V,No,(UNSIGNED)(LEFT.Value * Buckets)=RIGHT.PBucket AND ( LEFT.Value > RIGHT.LowP OR LEFT.Value = 0 AND RIGHT.LowP = 0 ) AND ( LEFT.Value <= RIGHT.P OR LEFT.Value >= MaxP ),Trans(LEFT,RIGHT),LOOKUP);
  RETURN J;
  END;

  END;
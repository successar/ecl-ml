IMPORT ML;

REAL8 gamma(REAL8 z) := BEGINC++
  #option pure
  #include <math.h>
  #body
  return tgamma(x);
ENDC++;

REAL8 lowerGamma(REAL8 x, REAL8 y)	:= BEGINC++
        #include <math.h>
        double n,r,s,ga,t,gin;
        int k;

        if ((x < 0.0) || (y < 0)) return 0;
        n = -y+x*log(y);

        if (y == 0.0) {
                gin = 0.0;
                return gin;
        }

        if (y <= 1.0+x) {
                s = 1.0/x;
                r = s;
                for (k=1;k<=100;k++) {
                        r *= y/(x+k);
                        s += r;
                        if (fabs(r/s) < 1e-15) break;
             }

        gin = exp(n)*s;
        }
        else {
                t = 0.0;
                for (k=100;k>=1;k--) {
                  t = (k-x)/(1.0+(k/(y+t)));
                }
                ga = exp(tgamma(x));
                gin = ga-(exp(n)/(y+t));
        }
        return gin;
ENDC++;

ChiSquareCDF(INTEGER df, REAL cv) := FUNCTION
        RETURN lowerGamma(df / 2, cv / 2) / gamma(df / 2);
END;

EXPORT Mahalanobis(DATASET(ML.Mat.Types.Element) A, REAL sensitivity = 0.05) := MODULE
        ZComp := ML.Mat.Pca(A).ZComp;

        // Calculate standard deviation for each column
        stdev := TABLE(ZComp, {
                y;
                stdev := SQRT(VARIANCE(GROUP, value));
        }, y);

        // Creates ZComp with normalized columns
        ZCompNorm := PROJECT(JOIN(ZComp, stdev, LEFT.y = RIGHT.y), TRANSFORM(
                ML.Mat.Types.Element,
                SELF.x := LEFT.x;
                SELF.y := LEFT.y;
                SELF.value := LEFT.value / LEFT.stdev;
        ));

        // Calculates distance squared
        EXPORT dsq := PROJECT(TABLE(ZCompNorm, {
                x;
                dsq := SUM(GROUP, value * value);
        }, x), TRANSFORM(
                ML.Mat.Types.Element,
                SELF.x := LEFT.x;
                SELF.y := 1;
                SELF.value := LEFT.dsq;
        ));

        // Calculates how many degrees of freedom we are dealing with
        df := ML.Mat.Has(A).Stats.YMax;

        // Verifies if p-value of each DSQ exceeds defined sensitivity
        EXPORT is_outlier := PROJECT(dsq, TRANSFORM(
                ML.Mat.Types.Element,
                SELF.x := LEFT.x;
                SELF.y := 1;
                SELF.value := IF(ChiSquareCDF(df, LEFT.value) > (1 - sensitivity), 1, 0);
        ));
END;

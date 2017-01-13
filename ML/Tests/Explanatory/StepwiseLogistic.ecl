   IMPORT ML;

   value_record := RECORD
         UNSIGNED   rid;
         REAL     age;
         REAL     height;
         integer1   sex; // 0 = female, 1 = male
   END;
   d := DATASET([{1,35,149,0},{2,11,138,0},{3,12,148,1},{4,16,156,0},
                 {5,32,152,0},{6,16,157,0},{7,14,165,0},{8,8,152,1},
                 {9,35,177,0},{10,33,158,1},{11,40,166,0},{12,28,165,0},  
                 {13,23,160,0},{14,52,178,1},{15,46,169,0},{16,29,173,1},
                 {17,30,172,0},{18,21,163,0},{19,21,164,0},{20,20,189,1},
                 {21,34,182,1},{22,43,184,1},{23,35,174,1},{24,39,177,1},
                 {25,43,183,1},{26,37,175,1},{27,32,173,1},{28,24,173,1},
                 {29,20,162,0},{30,25,180,1},{31,22,173,1},{32,25,171,1}]
                 ,value_record);
   ML.ToField(d,flds0);
   f4 := 
      PROJECT(flds0(Number=3)
              ,TRANSFORM(ML.Types.NumericField
                         ,SELF.Number := 4
                         ,SELF.Value := 1-LEFT.Value
                         ,SELF := LEFT
               )
      );
   flds1 := flds0+f4;
   flds := ML.Discretize.ByRounding(flds1);
	 Vars := DATASET([{1}], {UNSIGNED4 number});
	 X := flds1(number<=2);
	 Y := flds(number=3);
	 
	 Logf := ML.StepwiseLogistic.Forward(0.0);
	 regf:= Logf.Regression(X, Y);
	 OUTPUT(regf.Steps, NAMED('StepsF'));
	 modelf := regf.BestModel;
	 OUTPUT(Logf.Model(modelf), NAMED('BetasF'));
	 OUTPUT(Logf.ZStat(modelf), NAMED('ZStatF'));
	 Xf := Logf.ExtractX(X, regf.MapX);
	 OUTPUT(Logf.ClassifyC(Xf, modelf), NAMED('ClassifyF'));
	 OUTPUT(Logf.confint(95, modelf), NAMED('ConfIntF'));
	 Devf := Logf.DevianceC(Xf,Y, modelf);
	 OUTPUT(Devf.ResidDev, NAMED('ResidualDevF'));
	 OUTPUT(Devf.NullDev, NAMED('NullDevF'));
	 OUTPUT(Devf.AIC, NAMED('AICF'));
	 OUTPUT(Logf.AOD(Devf.NullDev, Devf.ResidDev), NAMED('AODF'));
	 
	 Logb := ML.StepwiseLogistic.Backward(0.0);
	 regb:= Logb.Regression(X, Y);
	 OUTPUT(regb.Steps, NAMED('StepsB'));
	 modelb := regb.BestModel;
	 OUTPUT(Logb.Model(modelb), NAMED('BetasB'));
	 OUTPUT(Logb.ZStat(modelb), NAMED('ZStatB'));
	 Xb := Logb.ExtractX(X, regb.MapX);
	 OUTPUT(Logb.ClassifyC(Xb, modelb), NAMED('ClassifyB'));
	 OUTPUT(Logb.confint(95, modelb), NAMED('ConfIntB'));
	 Devb := Logb.DevianceC(Xb,Y, modelb);
	 OUTPUT(Devb.ResidDev, NAMED('ResidualDevB'));
	 OUTPUT(Devb.NullDev, NAMED('NullDevB'));
	 OUTPUT(Devb.AIC, NAMED('AICB'));
	 OUTPUT(Logb.AOD(Devb.NullDev, Devb.ResidDev), NAMED('AODB'));
	 
	 Logbi := ML.StepwiseLogistic.Bidirectional(0.0);
	 regbi:= Logbi.Regression(X, Y, Vars);
	 OUTPUT(regbi.Steps, NAMED('StepsBi'));
	 modelbi := regbi.BestModel;
	 OUTPUT(Logbi.Model(modelbi), NAMED('BetasBi'));
	 OUTPUT(Logbi.ZStat(modelbi), NAMED('ZStatBi'));
	 Xbi := Logbi.ExtractX(X, regbi.MapX);
	 OUTPUT(Logbi.ClassifyC(Xbi, modelbi), NAMED('ClassifyBi'));
	 OUTPUT(Logbi.confint(95, modelbi), NAMED('ConfIntBi'));
	 Devbi := Logbi.DevianceC(Xbi,Y, modelbi);
	 OUTPUT(Devbi.ResidDev, NAMED('ResidualDevBi'));
	 OUTPUT(Devbi.NullDev, NAMED('NullDevBi'));
	 OUTPUT(Devbi.AIC, NAMED('AICBi'));
	 OUTPUT(Logbi.AOD(Devbi.NullDev, Devbi.ResidDev), NAMED('AODBi'));

/*R-code
B <- matrix(c(1,35,149,0,2,11,138,0,3,12,148,1,4,16,156,0,
                 5,32,152,0,6,16,157,0,7,14,165,0,8,8,152,1,
                 9,35,177,0,10,33,158,1,11,40,166,0,12,28,165,0,  
                 13,23,160,0,14,52,178,1,15,46,169,0,16,29,173,1,
                 17,30,172,0,18,21,163,0,19,21,164,0,20,20,189,1,
                 21,34,182,1,22,43,184,1,23,35,174,1,24,39,177,1,
                 25,43,183,1,26,37,175,1,27,32,173,1,28,24,173,1,
                 29,20,162,0,30,25,180,1,31,22,173,1,32,25,171,1
),nrow = 32, ncol = 4, byrow=TRUE);

Y <- B[, 4];
X1 <- B[, 2];
X2 <- B[, 3];

step(glm(Y~1, family="binomial"),direction="forward",scope=(~X1+X2));
OUTPUT :
Start:  AIC=46.24
Y ~ 1

       Df Deviance    AIC
+ X2    1   34.647 38.647
<none>      44.236 46.236
+ X1    1   42.931 46.931

Step:  AIC=38.65
Y ~ X2

       Df Deviance    AIC
<none>      34.647 38.647
+ X1    1   34.071 40.071

Call:  glm(formula = Y ~ X2, family = "binomial")

Coefficients:
(Intercept)           X2  
   -19.0634       0.1146  

Degrees of Freedom: 31 Total (i.e. Null);  30 Residual
Null Deviance:      44.24 
Residual Deviance: 34.65        AIC: 38.65

CODE :
step(glm(Y~X1+X2, family="binomial"),direction="backward",scope=(~X1+X2));
OUTPUT:
Start:  AIC=40.07
Y ~ X1 + X2

       Df Deviance    AIC
- X1    1   34.647 38.647
<none>      34.071 40.071
- X2    1   42.931 46.931

Step:  AIC=38.65
Y ~ X2

       Df Deviance    AIC
<none>      34.647 38.647
- X2    1   44.236 46.236

Call:  glm(formula = Y ~ X2, family = "binomial")

Coefficients:
(Intercept)           X2  
   -19.0634       0.1146  

Degrees of Freedom: 31 Total (i.e. Null);  30 Residual
Null Deviance:      44.24 
Residual Deviance: 34.65        AIC: 38.65

CODE:
step(glm(Y~X1, family="binomial"),direction="both",scope=(~X1+X2)); 
OUTPUT:
Start:  AIC=46.93
Y ~ X1

       Df Deviance    AIC
+ X2    1   34.071 40.071
- X1    1   44.236 46.236
<none>      42.931 46.931

Step:  AIC=40.07
Y ~ X1 + X2

       Df Deviance    AIC
- X1    1   34.647 38.647
<none>      34.071 40.071
- X2    1   42.931 46.931

Step:  AIC=38.65
Y ~ X2

       Df Deviance    AIC
<none>      34.647 38.647
+ X1    1   34.071 40.071
- X2    1   44.236 46.236

Call:  glm(formula = Y ~ X2, family = "binomial")

Coefficients:
(Intercept)           X2  
   -19.0634       0.1146  

Degrees of Freedom: 31 Total (i.e. Null);  30 Residual
Null Deviance:      44.24 
Residual Deviance: 34.65        AIC: 38.65
*/
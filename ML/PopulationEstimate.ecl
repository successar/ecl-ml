// Population Estmates of samples
IMPORT ML;
IMPORT ML.Mat AS Mat;
IMPORT PBblas;
IMPORT STD;
EXPORT PopulationEstimate := MODULE
/*
ConcordV1 is verison 1 of CONCORD in ECL. It takes a n x p data matrix Y (n observations, p variables), a real number lambda as the penalization paramater, the maximum number of iterates and the tolerance level as inputs and returns a p x p inverse covariance matrix.
*/
EXPORT ConcordV1(DATASET(Mat.Types.Element) Y, REAL8 lambda, INTEGER8 maxIter=100, REAL8 tol=0.00001) := FUNCTION
  p:= MAX(Y,y); //number of variables
  n:= MAX(Y,x); //number of observations
  S:= Mat.Scale(Mat.Mul(Mat.Trans(Y),Y),(1/n));
  O:= Mat.Identity(p);  
  OmegaType := ENUM(UNSIGNED1, Unknown=0, Om, OldOm, IterCnt);    
  OmegaElement := RECORD(ML.Mat.Types.Element)
     OmegaType typ;
  END;  
  OmegaElement cvt2OE(ML.Mat.Types.Element elm, OmegaType typ) := TRANSFORM
    SELF.typ := typ;
    SELF := elm;
  END;  
  old_oe1 := PROJECT(Mat.Sub(O,O), cvt2OE(LEFT, OmegaType.OldOm));
  newoe1 := PROJECT(O, cvt2OE(LEFT, OmegaType.Om));  
  newO:= old_oe1 + newoe1 + PROJECT(DATASET([{0,0,0}],Mat.Types.Element), cvt2OE(LEFT, OmegaType.IterCnt));
  INTEGER8 sign(REAL8 u) := FUNCTION
    out:= IF(u>0,1,IF(u<0,-1,0));
    RETURN out;
  END;
  REAL8 softThreshold(REAL8 x, REAL8 lambda) := FUNCTION
    st := sign(x)*MAX((ABS(x)-lambda),0);
    return st;
  END;  
  Mat.Types.Element MatAdd(Mat.Types.Element x1, Mat.Types.Element y1) := TRANSFORM
    SELF.x:=x1.x;
    SELF.y:=x1.y;
    SELF.value:=x1.value+y1.value;
  END;  
  Mat.Types.Element MatSub(Mat.Types.Element x1, Mat.Types.Element y1) := TRANSFORM
    SELF.x:=x1.x;
    SELF.y:=x1.y;
    SELF.value:=x1.value-y1.value;
  END;
  Mat.Types.Element sum2cell(Mat.Types.Element b, DATASET(Mat.Types.Element) cells) := TRANSFORM
    SELF.x := b.x;
    SELF.y := b.y;
    SELF.value := SUM(cells, value);
  END;  
  OmegaElement updateOmega(DATASET(OmegaElement) new1) := FUNCTION
    unchanged_cells_new:=new1(typ<>OmegaType.OldOm);
    new_mat := PROJECT(new1(typ=OmegaType.Om), ML.Mat.Types.Element);
    new_OE := PROJECT(new_mat, cvt2OE(LEFT, OmegaType.OldOm));
    RETURN unchanged_cells_new+new_OE;
  END;  
  MaxAbsDff(DATASET(OmegaElement) work_zz) := FUNCTION
    zz1:= PROJECT(work_zz(typ=OmegaType.Om), ML.Mat.Types.Element);
    zz2:= PROJECT(work_zz(typ=OmegaType.OldOm), ML.Mat.Types.Element);
    Mat.Types.Element MatAbsDifference(Mat.Types.Element x1, Mat.Types.Element y1) := TRANSFORM
      SELF.x:=x1.x;
      SELF.y:=x1.y;
      SELF.value:=ABS(x1.value-y1.value);
    END;
    result := JOIN(zz1,zz2,(LEFT.x=RIGHT.x) AND (LEFT.y=RIGHT.y),MatAbsDifference(LEFT,RIGHT),FULL OUTER);
    result2 := MAX(result,value);
    RETURN result2;
  END;  
  innerBody(DATASET(OmegaElement) work_OE, UNSIGNED i, UNSIGNED j) := FUNCTION
    work_O := PROJECT(work_OE(typ=OmegaType.Om), ML.Mat.Types.Element);
    work_OldO := PROJECT(work_OE(typ=OmegaType.OldOm), ML.Mat.Types.Element);
    unchanged_cells := work_O((x<>i OR y<>j)AND(x<>j OR y<>i));
    t1 := JOIN(work_O(x=i AND y<>j), S(x=j AND y<>j),
      LEFT.y=RIGHT.y,
      Transform(Mat.Types.Element,SELF.x:=LEFT.x,SELF.value:=LEFT.value*RIGHT.value,Self:=LEFT));
    t2 := JOIN(work_O(x=j AND y<>i), S(x=i AND y<>i),
      LEFT.y=RIGHT.y,
      Transform(Mat.Types.Element,SELF.x:=LEFT.x,SELF.value:=LEFT.value*RIGHT.value,Self:=LEFT));
    g0 := GROUP(t1+t2, i, j, ALL);
    r0 := ROLLUP(g0, GROUP, sum2cell(LEFT, ROWS(LEFT)));
    numer := softThreshold(-r0[1].value,(lambda/n));
    denom := S(x=i AND y=i)[1].value+S(x=j AND y=j)[1].value;
    new_value := numer/denom;
    new_c1 := DATASET([{i, j, new_value},{j, i, new_value}], Mat.Types.Element);
    newOm := unchanged_cells + new_c1;
    old_oe := PROJECT(work_OldO, cvt2OE(LEFT, OmegaType.OldOm));
    newoe := PROJECT(newOm, cvt2OE(LEFT, OmegaType.Om));
    rslt := old_oe + newoe + work_OE(typ=OmegaType.IterCnt);    
    RETURN rslt;
  END;  
  outerBody(DATASET(OmegaElement) work_s, UNSIGNED i) :=
    LOOP(work_s, (p-i), innerBody(ROWS(LEFT), i, COUNTER+i));
  Body(DATASET(OmegaElement) work_OE, UNSIGNED i) := FUNCTION
    work_O := PROJECT(work_OE(typ=OmegaType.Om), ML.Mat.Types.Element);
    work_OldO := PROJECT(work_OE(typ=OmegaType.OldOm), ML.Mat.Types.Element);
    unchanged_cells := work_O(x<>i OR y<>i);
    t1 := JOIN(work_O(x=i AND y<>i),S(x=i AND y<>i),
      LEFT.y=RIGHT.y,
      Transform(Mat.Types.Element,SELF.x:=LEFT.x,SELF.value:=LEFT.value*RIGHT.value,Self:=LEFT));
    r0 := SUM(t1, value);
    numer := -r0 + SQRT(POWER(r0,2.0)+4*S(x=i AND y = i)[1].value);
    denom := 2*S(x=i AND y=i)[1].value;
    new_value := numer/denom;
    new_c1 := DATASET([{i, i, new_value}], Mat.Types.Element);
    newOm := unchanged_cells + new_c1;    
    old_oe := PROJECT(work_OldO, cvt2OE(LEFT, OmegaType.OldOm));
    newoe := PROJECT(newOm, cvt2OE(LEFT, OmegaType.Om));
    rslt := old_oe + newoe + work_OE(typ=OmegaType.IterCnt);    
    RETURN rslt;    
  END;
  OuterOuterBody2(DATASET(OmegaElement) work_O, UNSIGNED4 cnt):= FUNCTION
    work_Omega:= updateOmega(work_O);    
    step1 := LOOP(work_Omega, (p-1), outerBody(ROWS(LEFT), COUNTER));
    step2 := LOOP(step1, p, Body(ROWS(LEFT), COUNTER));
    work_OmegaElement := step2(typ<>OmegaType.IterCnt)+PROJECT(DATASET([{0,0,cnt}],Mat.Types.Element), cvt2OE(LEFT, OmegaType.IterCnt));
    RETURN work_OmegaElement;
  END;
  omegahat:= LOOP(newO,(COUNTER<=maxIter AND
                        MaxAbsDff(ROWS(LEFT))>tol),
                  OuterOuterBody2(ROWS(LEFT),COUNTER));  
  RETURN SORT(PROJECT(omegahat(typ=OmegaType.Om), ML.Mat.Types.Element),x,y);
END; //Concordv1
//Concordv2
EXPORT ConcordV2(DATASET(Mat.Types.Element) Y, REAL8 lambda, INTEGER8 maxIter=100, REAL8 tol=0.00001) := FUNCTION
  p:= MAX(Y,y); //number of variables
  n:= MAX(Y,x); //number of observations
  S:= Mat.Scale(Mat.Mul(Mat.Trans(Y),Y),(1/n));
  O:= Mat.Identity(p);  
  OmegaType := ENUM(UNSIGNED1, Unknown=0, Om, OldOm, Residual, IterCnt);    
  OmegaElement := RECORD(ML.Mat.Types.Element)
     OmegaType typ;
  END;
  OmegaElement cvt2OE(ML.Mat.Types.Element elm, OmegaType typ) := TRANSFORM
    SELF.typ := typ;
    SELF := elm;
  END;  
  old_oe1 := PROJECT(Mat.Sub(O,O), cvt2OE(LEFT, OmegaType.OldOm));
  new_res1 := PROJECT(Y, cvt2OE(LEFT, OmegaType.Residual));
  newoe1 := PROJECT(O, cvt2OE(LEFT, OmegaType.Om));
  newO:= old_oe1 + new_res1 + newoe1 + PROJECT(DATASET([{0,0,0}],Mat.Types.Element), cvt2OE(LEFT, OmegaType.IterCnt));
  INTEGER8 sign(REAL8 u) := FUNCTION
    out:= IF(u>0,1,IF(u<0,-1,0));
    RETURN out;
  END;
  REAL8 softThreshold(REAL8 x, REAL8 lambda) := FUNCTION
    st := sign(x)*MAX((ABS(x)-lambda),0);
    return st;
  END;  
  Mat.Types.Element MatAdd(Mat.Types.Element x1, Mat.Types.Element y1) := TRANSFORM
    SELF.x:=x1.x;
    SELF.y:=x1.y;
    SELF.value:=x1.value+y1.value;
  END;
  Mat.Types.Element MatSub(Mat.Types.Element x1, Mat.Types.Element y1) := TRANSFORM
  SELF.x:=x1.x;
  SELF.y:=x1.y;
  SELF.value:=x1.value-y1.value;
END;
  Mat.Types.Element sum2cell(Mat.Types.Element b, DATASET(Mat.Types.Element) cells) := TRANSFORM
    SELF.x := b.x;
    SELF.y := b.y;
    SELF.value := SUM(cells, value);
  END;  
  Mat.Types.Element MatDiagInv(Mat.Types.Element x1) := TRANSFORM
    SELF.x:=x1.x;
    SELF.y:=x1.y;
    SELF.value:=1/(x1.value);
  END;
  OmegaElement updateOmega(DATASET(OmegaElement) new1) := FUNCTION
    unchanged_cells_new:=new1(typ<>OmegaType.OldOm);
    new_mat := PROJECT(new1(typ=OmegaType.Om), ML.Mat.Types.Element);
    new_OE := PROJECT(new_mat, cvt2OE(LEFT, OmegaType.OldOm));
    RETURN unchanged_cells_new+new_OE;
  END;  
  MaxAbsDff(DATASET(OmegaElement) work_zz) := FUNCTION
    zz1:= PROJECT(work_zz(typ=OmegaType.Om), ML.Mat.Types.Element);
    zz2:= PROJECT(work_zz(typ=OmegaType.OldOm), ML.Mat.Types.Element);
    Mat.Types.Element MatAbsDifference(Mat.Types.Element x1, Mat.Types.Element y1) := TRANSFORM
      SELF.x:=x1.x;
      SELF.y:=x1.y;
      SELF.value:=ABS(x1.value-y1.value);
    END;
    result := JOIN(zz1,zz2,(LEFT.x=RIGHT.x) AND (LEFT.y=RIGHT.y),MatAbsDifference(LEFT,RIGHT),FULL OUTER);
    result2 := MAX(result,value);
    RETURN result2;
  END;  
  offdiagupdateresiduals(DATASET(Mat.Types.Element) residuals, DATASET(Mat.Types.Element)omegacurrent,
    DATASET(Mat.Types.Element) omegaold, i, j) := FUNCTION
    unchanged_cells:= residuals(y<>j);
    c:= (omegacurrent(x=j AND y=i)[1].value-omegaold(x=j AND y=i)[1].value)/omegacurrent(x=j AND y=j)[1].value;
    newc:=Mat.Scale(Y(y=i),c);
    newvalue := JOIN(residuals(y=j),newc,LEFT.x=RIGHT.x,MatAdd(LEFT,RIGHT));
    RETURN newvalue+unchanged_cells;
  END;  
  diagupdateresiduals(DATASET(Mat.Types.Element) residuals, DATASET(Mat.Types.Element) currentomega,
    DATASET(Mat.Types.Element) oldomega, i) := FUNCTION
    unchanged_cells:= residuals(y<>i);
    c:= oldomega(x=i AND y=i)[1].value/currentomega(x=i AND y=i)[1].value;
    newvalue1:= JOIN(residuals(y=i),Y(y=i),LEFT.x=RIGHT.x,MatSub(LEFT,RIGHT));
    newc:=Mat.Scale(newvalue1,c);
    newvalue2 := JOIN(Y(y=i),newc,LEFT.x=RIGHT.x,MatAdd(LEFT,RIGHT));
    RETURN newvalue2+unchanged_cells;
  END;
  innerBody(DATASET(OmegaElement) work_OE, UNSIGNED i, UNSIGNED j) := FUNCTION
    work_O := PROJECT(work_OE(typ=OmegaType.Om), ML.Mat.Types.Element);
    work_OldO := PROJECT(work_OE(typ=OmegaType.OldOm), ML.Mat.Types.Element);
    res := PROJECT(work_OE(typ=OmegaType.Residual), ML.Mat.Types.Element);
    unchanged_cells := work_O((x<>i OR y<>j)AND(x<>j OR y<>i));
    ri := res(y=i);
    Yjritemp:= JOIN(Y(y=j), ri, LEFT.x=RIGHT.x,
                    Transform(Mat.Types.Element,SELF.y:=LEFT.y,SELF.value:=LEFT.value*RIGHT.value,Self:=LEFT));
    Yjri := SUM(Yjritemp, value);
    t1 := -work_O(x=i,y=j)[1].value*S(x=j,y=j)[1].value+(1/n)*work_O(x=i,y=i)[1].value*Yjri;
    rj := res(y=j);
    Yirjtemp:= JOIN(Y(y=i),rj, LEFT.x=RIGHT.x,
                    Transform(Mat.Types.Element,SELF.y:=LEFT.y,SELF.value:=LEFT.value*RIGHT.value,Self:=LEFT));
    Yirj := SUM(Yirjtemp, value);
    t2:= -work_O(x=j,y=i)[1].value*S(x=i,y=i)[1].value+(1/n)*work_O(x=j,y=j)[1].value*Yirj;    
    numer := softThreshold(-(t1+t2),(lambda/n));
    denom := S(x=i AND y=i)[1].value+S(x=j AND y=j)[1].value;
    new_value := numer/denom;
    new_c1 := DATASET([{i, j, new_value},{j, i, new_value}], Mat.Types.Element);
    newOm:= unchanged_cells + new_c1;
    res1:= offdiagupdateresiduals(res, newOm, work_O, i, j);
    res2:= offdiagupdateresiduals(res1, newOm, work_O, j, i);
    old_oe := PROJECT(work_OldO, cvt2OE(LEFT, OmegaType.OldOm));
    new_res := PROJECT(res2, cvt2OE(LEFT, OmegaType.Residual));
    newoe := PROJECT(newOm, cvt2OE(LEFT, OmegaType.Om));
    rslt := old_oe + new_res + newoe + work_OE(typ=OmegaType.IterCnt);    
    RETURN rslt;
  END;  
  outerBody(DATASET(OmegaElement) work_s, UNSIGNED i) :=
    LOOP(work_s, (p-i), innerBody(ROWS(LEFT), i, COUNTER+i));
  Body(DATASET(OmegaElement) work_OE, UNSIGNED i) := FUNCTION
    work_O := PROJECT(work_OE(typ=OmegaType.Om), ML.Mat.Types.Element);
    work_OldO := PROJECT(work_OE(typ=OmegaType.OldOm), ML.Mat.Types.Element);
    res := PROJECT(work_OE(typ=OmegaType.Residual), ML.Mat.Types.Element);
    unchanged_cells := work_O(x<>i OR y<>i);
    ri := res(y=i);
    Yiritemp:= JOIN(Y(y=i),ri, LEFT.x=RIGHT.x,
                    Transform(Mat.Types.Element,SELF.y:=LEFT.y,SELF.value:=LEFT.value*RIGHT.value,Self:=LEFT));
    Yiri := SUM(Yiritemp, value);
    r0 := -work_O(x=i,y=i)[1].value*S(x=i,y=i)[1].value+(1/n)*work_O(x=i,y=i)[1].value*Yiri;    
    numer := -r0 + SQRT(POWER(r0,2.0)+4*S(x=i AND y = i)[1].value);
    denom := 2*S(x=i AND y=i)[1].value;
    new_value := numer/denom;
    new_c1 := ROW({i, i, new_value}, Mat.Types.Element);
    newOm:= unchanged_cells + new_c1;
    res1:= diagupdateresiduals(res, newOm, work_OldO, i);    
    old_oe := PROJECT(work_OldO, cvt2OE(LEFT, OmegaType.OldOm));
    new_res := PROJECT(res1, cvt2OE(LEFT, OmegaType.Residual));
    newoe := PROJECT(newOm, cvt2OE(LEFT, OmegaType.Om));
    rslt := old_oe + new_res + newoe + work_OE(typ=OmegaType.IterCnt);    
    RETURN rslt;
  END;
  OuterOuterBody2(DATASET(OmegaElement) work_O, UNSIGNED4 cnt):= FUNCTION
    //residual calculation
    work_Omega:= updateOmega(work_O);
    work_Om:= PROJECT(work_O(typ=OmegaType.Om), ML.Mat.Types.Element);
    omega_tilde := PROJECT(work_Om(x=y), MatDiagInv(LEFT));
    newomega:= Mat.Mul(work_Om,omega_tilde);
    residuals:= Mat.Mul(Y,newomega);  
    new_res := PROJECT(residuals, cvt2OE(LEFT, OmegaType.Residual));
    othercells:= work_Omega(typ<>OmegaType.Residual);
    newrslt := new_res + othercells;
    step1 := LOOP(newrslt, (p-1), outerBody(ROWS(LEFT), COUNTER));
    step2 := LOOP(step1, p, Body(ROWS(LEFT), COUNTER));
    work_OmegaElement := step2(typ<>OmegaType.IterCnt)+
                          PROJECT(DATASET([{0,0,cnt}],Mat.Types.Element), cvt2OE(LEFT, OmegaType.IterCnt));
    RETURN work_OmegaElement;
  END;
  omegahat:= LOOP(newO,(COUNTER<=maxIter AND
                        MaxAbsDff(ROWS(LEFT))>tol),
                  OuterOuterBody2(ROWS(LEFT),COUNTER));  
  RETURN SORT(PROJECT(omegahat(typ=1), ML.Mat.Types.Element),x,y);
END; //Concord v2
//concord v2 with blas operations
EXPORT ConcordV2Blas(DATASET(Mat.Types.Element) Y, REAL8 lambda, INTEGER8 maxIter=100, REAL8 tol=0.00001) := FUNCTION
  p:= MAX(Y,y); //number of variables
  n:= MAX(Y,x); //number of observations
  partts:= STD.System.Thorlib.nodes(); //number of nodes  
  PBblas.Types.value_t ScalebyN(PBblas.Types.value_t v,
                                      PBblas.Types.dimension_t r,
                                      PBblas.Types.dimension_t c) := v/n;
  Ydivisor:= ((p-1) DIV partts)+1;
  Ymap:= PBblas.Matrix_Map(n,p,Ydivisor,Ydivisor);
  Omap:= PBblas.Matrix_Map(p,p,Ydivisor,Ydivisor);  
  Y_cc:= ML.Dmat.Converted.FromElement(Y,Ymap);
  YTY:= PBblas.PB_dgemm(TRUE, FALSE, 1.0, Ymap, Y_cc, Ymap, Y_cc, Omap);
  Sblas:= PBblas.Apply2Elements(Omap, YTY, ScalebyN);
  S:= ML.Dmat.Converted.FromPart2Elm(Sblas);
  O:= Mat.Identity(p);  
  OmegaType := ENUM(UNSIGNED1, Unknown=0, Om, OldOm, Residual, IterCnt);    
  OmegaElement := RECORD(ML.Mat.Types.Element)
     OmegaType typ;
  END;  
  OmegaElement cvt2OE(ML.Mat.Types.Element elm, OmegaType typ) := TRANSFORM
    SELF.typ := typ;
    SELF := elm;
  END;  
  old_oe1 := PROJECT(Mat.Sub(O,O), cvt2OE(LEFT, OmegaType.OldOm));
  new_res1 := PROJECT(Y, cvt2OE(LEFT, OmegaType.Residual));
  newoe1 := PROJECT(O, cvt2OE(LEFT, OmegaType.Om));  
  newO:= old_oe1 + new_res1 + newoe1 + PROJECT(DATASET([{0,0,0}],Mat.Types.Element), cvt2OE(LEFT, OmegaType.IterCnt));
  INTEGER8 sign(REAL8 u) := FUNCTION
    out:= IF(u>0,1,IF(u<0,-1,0));
    RETURN out;
  END;
  REAL8 softThreshold(REAL8 x, REAL8 lambda) := FUNCTION
    st := sign(x)*MAX((ABS(x)-lambda),0);
    return st;
  END;  
  Mat.Types.Element MatAdd(Mat.Types.Element x1, Mat.Types.Element y1) := TRANSFORM
    SELF.x:=x1.x;
    SELF.y:=x1.y;
    SELF.value:=x1.value+y1.value;
  END;  
  Mat.Types.Element MatSub(Mat.Types.Element x1, Mat.Types.Element y1) := TRANSFORM
    SELF.x:=x1.x;
    SELF.y:=x1.y;
    SELF.value:=x1.value-y1.value;
  END;
  Mat.Types.Element sum2cell(Mat.Types.Element b, DATASET(Mat.Types.Element) cells) := TRANSFORM
    SELF.x := b.x;
    SELF.y := b.y;
    SELF.value := SUM(cells, value);
  END;  
  PBblas.Types.value_t MatDiagInv(PBblas.Types.value_t v,
                                      PBblas.Types.dimension_t r,
                                      PBblas.Types.dimension_t c) := IF(r=c,1/v,0);
  OmegaElement updateOmega(DATASET(OmegaElement) new1) := FUNCTION
    unchanged_cells_new:=new1(typ<>OmegaType.OldOm);
    new_mat := PROJECT(new1(typ=OmegaType.Om), ML.Mat.Types.Element);
    new_OE := PROJECT(new_mat, cvt2OE(LEFT, OmegaType.OldOm));
    RETURN unchanged_cells_new+new_OE;
  END;  
  MaxAbsDff(DATASET(OmegaElement) work_zz) := FUNCTION
    zz1:= PROJECT(work_zz(typ=OmegaType.Om), ML.Mat.Types.Element);
    zz2:= PROJECT(work_zz(typ=OmegaType.OldOm), ML.Mat.Types.Element);
    Mat.Types.Element MatAbsDifference(Mat.Types.Element x1, Mat.Types.Element y1) := TRANSFORM
      SELF.x:=x1.x;
      SELF.y:=x1.y;
      SELF.value:=ABS(x1.value-y1.value);
    END;
    result := JOIN(zz1,zz2,(LEFT.x=RIGHT.x) AND (LEFT.y=RIGHT.y),MatAbsDifference(LEFT,RIGHT),FULL OUTER);
    result2 := MAX(result,value);
    RETURN result2;
  END;  
  offdiagupdateresiduals(DATASET(Mat.Types.Element) residuals, DATASET(Mat.Types.Element)omegacurrent,
    DATASET(Mat.Types.Element) omegaold, i, j) := FUNCTION
    unchanged_cells:= residuals(y<>j);
    c:= (omegacurrent(x=j AND y=i)[1].value-omegaold(x=j AND y=i)[1].value)/omegacurrent(x=j AND y=j)[1].value;
    newc:=Mat.Scale(Y(y=i),c);
    newvalue := JOIN(residuals(y=j),newc,LEFT.x=RIGHT.x,MatAdd(LEFT,RIGHT));
    RETURN newvalue+unchanged_cells;
  END;  
  diagupdateresiduals(DATASET(Mat.Types.Element) residuals, DATASET(Mat.Types.Element) currentomega,
    DATASET(Mat.Types.Element) oldomega, i) := FUNCTION
    unchanged_cells:= residuals(y<>i);
    c:= oldomega(x=i AND y=i)[1].value/currentomega(x=i AND y=i)[1].value;
    newvalue1:= JOIN(residuals(y=i),Y(y=i),LEFT.x=RIGHT.x,MatSub(LEFT,RIGHT));
    newc:=Mat.Scale(newvalue1,c);
    newvalue2 := JOIN(Y(y=i),newc,LEFT.x=RIGHT.x,MatAdd(LEFT,RIGHT));
    RETURN newvalue2+unchanged_cells;
  END;
  innerBody(DATASET(OmegaElement) work_OE, UNSIGNED i, UNSIGNED j) := FUNCTION
    work_O := PROJECT(work_OE(typ=OmegaType.Om), ML.Mat.Types.Element);
    work_OldO := PROJECT(work_OE(typ=OmegaType.OldOm), ML.Mat.Types.Element);
    res := PROJECT(work_OE(typ=OmegaType.Residual), ML.Mat.Types.Element);
    unchanged_cells := work_O((x<>i OR y<>j)AND(x<>j OR y<>i));
    ri := res(y=i);
    Yjritemp:= JOIN(Y(y=j), ri, LEFT.x=RIGHT.x,
                    Transform(Mat.Types.Element,SELF.y:=LEFT.y,SELF.value:=LEFT.value*RIGHT.value,Self:=LEFT));
    Yjri := SUM(Yjritemp, value);
    t1 := -work_O(x=i,y=j)[1].value*S(x=j,y=j)[1].value+(1/n)*work_O(x=i,y=i)[1].value*Yjri;
    rj := res(y=j);
    Yirjtemp:= JOIN(Y(y=i),rj, LEFT.x=RIGHT.x,
                    Transform(Mat.Types.Element,SELF.y:=LEFT.y,SELF.value:=LEFT.value*RIGHT.value,Self:=LEFT));
    Yirj := SUM(Yirjtemp, value);
    t2:= -work_O(x=j,y=i)[1].value*S(x=i,y=i)[1].value+(1/n)*work_O(x=j,y=j)[1].value*Yirj;
    numer := softThreshold(-(t1+t2),(lambda/n));
    denom := S(x=i AND y=i)[1].value+S(x=j AND y=j)[1].value;
    new_value := numer/denom;
    new_c1 := DATASET([{i, j, new_value},{j, i, new_value}], Mat.Types.Element);
    newOm:= unchanged_cells + new_c1;
    res1:= offdiagupdateresiduals(res, newOm, work_O, i, j);
    res2:= offdiagupdateresiduals(res1, newOm, work_O, j, i);
    old_oe := PROJECT(work_OldO, cvt2OE(LEFT, OmegaType.OldOm));
    new_res := PROJECT(res2, cvt2OE(LEFT, OmegaType.Residual));
    newoe := PROJECT(newOm, cvt2OE(LEFT, OmegaType.Om));
    rslt := old_oe + new_res + newoe + work_OE(typ=OmegaType.IterCnt);    
    RETURN rslt;
  END;  
  outerBody(DATASET(OmegaElement) work_s, UNSIGNED i) :=
    LOOP(work_s, (p-i), innerBody(ROWS(LEFT), i, COUNTER+i));
  Body(DATASET(OmegaElement) work_OE, UNSIGNED i) := FUNCTION
    work_O := PROJECT(work_OE(typ=OmegaType.Om), ML.Mat.Types.Element);
    work_OldO := PROJECT(work_OE(typ=OmegaType.OldOm), ML.Mat.Types.Element);
    res := PROJECT(work_OE(typ=OmegaType.Residual), ML.Mat.Types.Element);
    unchanged_cells := work_O(x<>i OR y<>i);
    ri := res(y=i);
    Yiritemp:= JOIN(Y(y=i),ri, LEFT.x=RIGHT.x,
                    Transform(Mat.Types.Element,SELF.y:=LEFT.y,SELF.value:=LEFT.value*RIGHT.value,Self:=LEFT));
    Yiri := SUM(Yiritemp, value);
    r0 := -work_O(x=i,y=i)[1].value*S(x=i,y=i)[1].value+(1/n)*work_O(x=i,y=i)[1].value*Yiri;    
    numer := -r0 + SQRT(POWER(r0,2.0)+4*S(x=i AND y = i)[1].value);
    denom := 2*S(x=i AND y=i)[1].value;
    new_value := numer/denom;
    new_c1 := ROW({i, i, new_value}, Mat.Types.Element);
    newOm:= unchanged_cells + new_c1;
    res1:= diagupdateresiduals(res, newOm, work_OldO, i);    
    old_oe := PROJECT(work_OldO, cvt2OE(LEFT, OmegaType.OldOm));
    new_res := PROJECT(res1, cvt2OE(LEFT, OmegaType.Residual));
    newoe := PROJECT(newOm, cvt2OE(LEFT, OmegaType.Om));
    rslt := old_oe + new_res + newoe + work_OE(typ=OmegaType.IterCnt);    
    RETURN rslt;    
  END;
  OuterOuterBody2(DATASET(OmegaElement) work_O, UNSIGNED4 cnt):= FUNCTION
      //residual calculation
    divisor:= ((p-1) DIV partts)+1;
    map1:= PBblas.Matrix_Map(n,p,divisor,divisor);
    map2:= PBblas.Matrix_Map(p,p,divisor,divisor);        
    work_Omega:= updateOmega(work_O);
    work_Om:= PROJECT(work_O(typ=OmegaType.Om), ML.Mat.Types.Element);
    Y_c:= ML.Dmat.Converted.FromElement(Y,map1);
    work_Om_c:= ML.Dmat.Converted.FromElement(work_Om,map2);
    omega_tilde:= PBblas.Apply2Elements(map1, work_Om_c, MatDiagInv);
    newomega:= PBblas.PB_dgemm(FALSE, FALSE,
                             1.0, map2, work_Om_c, map2, omega_tilde,
                             map2);
    residuals_c:=  PBblas.PB_dgemm(FALSE, FALSE,
                             1.0, map1, Y_c, map2, newomega,
                             map1);
    residuals:= ML.Dmat.Converted.FromPart2Elm(residuals_c);  
    new_res := PROJECT(residuals, cvt2OE(LEFT, OmegaType.Residual));
    othercells:= work_Omega(typ<>OmegaType.Residual);
    newrslt := new_res + othercells;
    step1 := LOOP(newrslt, (p-1), outerBody(ROWS(LEFT), COUNTER));
    step2 := LOOP(step1, p, Body(ROWS(LEFT), COUNTER));
    work_OmegaElement := step2(typ<>OmegaType.IterCnt)+
                          PROJECT(DATASET([{0,0,cnt}],Mat.Types.Element), cvt2OE(LEFT, OmegaType.IterCnt));
    RETURN work_OmegaElement;
  END;
  omegahat:= LOOP(newO,(COUNTER<=maxIter AND
                        MaxAbsDff(ROWS(LEFT))>tol),
                  OuterOuterBody2(ROWS(LEFT),COUNTER));
  RETURN SORT(PROJECT(omegahat(typ=1), ML.Mat.Types.Element),x,y);
END;//concordv2blas
/*
InverseCovariance uses ConcordV1 to get an estimate of the inverse covariance matrix.
*/
EXPORT InverseCovariance(DATASET(Mat.Types.Element) Ydata, REAL8 lambda1, INTEGER8 maxIter=100, REAL8 tol=0.0001) :=
  ConcordV2(Y:=Ydata, lambda:=lambda1);
/*
Covariance uses ConcordV1 to get an estimate of the inverse covariance matrix and then takes it's inverse.
*/
EXPORT Covariance(DATASET(Mat.Types.Element) Ydata, REAL8 lambda1, INTEGER8 maxIter=100, REAL8 tol=0.0001) := FUNCTION
  out:= Mat.Inv(ConcordV2(Y:=Ydata, lambda:=lambda1));
  RETURN out;
END; // Covariance
END; // PopulationEstimate
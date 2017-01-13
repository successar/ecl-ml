//Take a dataset of cells for a partition and pack into a dense matrix.  Specify Row or Column major
//First row and first column are one based.
//Insert is used insert columns with a spacific value.  Typical use is building a matrix for a solver
//where the first column is an inserted column of 1 values for the intercept.
IMPORT PBblas.Types;
dimension_t := Types.dimension_t;
value_t := Types.value_t;
Layout_Cell := Types.Layout_Cell;

EXPORT SET OF REAL8 makeR8Set(dimension_t r, dimension_t s,
                              dimension_t first_row, dimension_t first_col,
                              DATASET(Layout_Cell) D,
                              dimension_t insert_columns,
                              value_t insert_value) := BEGINC++
    typedef struct work1 {      // copy of Layout_Cell translated to C
      uint32_t x;
      uint32_t y;
      double v;
    };
    #body
    __lenResult = r * s * sizeof(double);
    __isAllResult = false;
    double * result = new double[r*s];
    __result = (void*) result;
    work1 *cell = (work1*) d;
    uint32_t cells = lenD / sizeof(work1);
    uint32_t i;
    uint32_t pos;
    for (i=0; i<r*s; i++) {
      result[i] =  i/r < insert_columns  ? insert_value   : 0.0;
    }
    int x, y;
    for (i=0; i<cells; i++) {
      x = cell[i].x - first_row;                   // input co-ordinates are one based,
      y = cell[i].y + insert_columns - first_col;  //x and y are zero based.
      if(x < 0 || (uint32_t) x >= r) continue;   // cell does not belong
      if(y < 0 || (uint32_t) y >= s) continue;
      pos = (y*r) + x;
      result[pos] = cell[i].v;
    }
  ENDC++;

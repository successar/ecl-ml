// Return natural log of the gamma function
EXPORT REAL8 log_gamma(REAL8 x) := BEGINC++
  #option pure
  #include <math.h>
  #body
  return lgamma(x);
ENDC++;
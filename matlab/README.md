This folder contains helpful test scripts for modules in RunSuiteSparse.

Dependencies:
  ba_utils/
  SPQR.  (See SuiteSparse).  Use 'spqr_make.m' to build for your machine.

  NOTE:  Some add_path(.); in the original script may not contain proper paths.

    compare_qr.m:

        Compares a saved R (sparse matrix) and Q'*b (dense vector) with the
          solution provided by matlabs qr(...)  and spqr(...).


    compare_qr_matlab.m

        This is an attempt to build the compare_split_qr.cc program within
        matlab.  The R matrix becomes singular.  Something is not right.

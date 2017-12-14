compare_qr_double.cc:

  This program does a solve for the system Jx=b where J is a provided argument
  of sparse matrix type.

compare_split_qr.cc

  This program is an attempt to split the provided matrix into sub-systems in
  order to speed up parallelization.

    Jx=b  ===>   [J1, J2, ..., Jn]^T * x = [b1, b2, ... bn]^T.

  WARNING:
    The R being returned by the QR factorization is not matching the R which is
    returned by SuiteSparse's matlab implementation.  spqr(...).

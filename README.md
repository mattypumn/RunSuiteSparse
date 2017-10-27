This repo has multiple programs for testing SuiteSparse with floats and with
doubles.


REQUIREMENTS:
      - Up-to-date customized SuiteSparse to work in floats
            https://github.com/mattypumn/SuiteSparse.git
      - SuiteSparse 4.5.5
            http://faculty.cse.tamu.edu/davis/suitesparse.html
      - Patients.

      - glogging / gflags.

      Be sure the above have been compiled.
      NOTE: CUDA is not necessary for this library.

EXECUTABLES:

    spqr_user_guide_example

        This was the original program used to build and test a working
        version of SuiteSparse that runs with only floats.  It Reades in a
        matrix from file. And solves the linear system where the
        right-hand-side is all ones:  Ax = b.  This is solved using
        SparseSuiteQR();

      COMPILE:
          - Edit .../RunSuiteSparse/suite_sparse_float/CMakeLists.txt:

              Set the SUITE_SPARSE_FLOAT to the absolute path of the above
              version of SuiteSparse.

          - From .../RunSuiteSparse/ directory run the following command:

              $> mkdir -p build && cd build && make spqr_user_guide_example


    compare_qr_float

        This program loads binary sparse matrices written by
        MARS::bls::io::EigenIO. Then solves the system Ax = b. Where A is the
        loaded matrix and b is a vector of all ones. Every scalar is of type
        float.


        HARD CODED INFORMATION: (Change to run locally).

          times_file: output file for the timing data.

          residuals_file: Output file for the residuals for each solve.

          kNumSolves: Number of solves when running TimeSolvesN(...).

          kLoadTranspose: Set to true if the matrices being loaded were saved
                          as the transpose. See MARS::bls::io::EigenIO and
                          source code for more information.

        COMPILE:
          - Edit .../RunSuiteSparse/suite_sparse_float/CMakeLists.txt:

              Set the SUITE_SPARSE_FLOAT to the absolute path of the above
              version of SuiteSparse.

          - From .../RunSuiteSparse/ directory run the following command:

              $> mkdir -p build && cd build && make compare_qr_float


    compare_qr_double

        This program loads binary sparse matrices written by
        MARS::bls::io::EigenIO. Then solves the system Ax = b. Where A is the
        loaded matrix and b is a vector of all ones.  Every scalar is of type
        double.


        HARD CODED INFORMATION: (Change to run locally).

          times_file: output file for the timing data.

          residuals_file: Output file for the residuals for each solve.

          kNumSolves: Number of solves when running TimeSolvesN(...).

          kLoadTranspose: Set to true if the matrices being loaded were saved
                          as the transpose. See MARS::bls::io::EigenIO and
                          source code for more information.

        COMPILE:
          - Edit .../RunSuiteSparse/suite_sparse_double/CMakeLists.txt:

              Set the SUITE_SPARSE_DOUBLE to the absolute path of the original
              version of SuiteSparse.  Source code can be found http://faculty.cse.tamu.edu/davis/suitesparse.html

          - From .../RunSuiteSparse/ directory run the following command:

              $> mkdir -p build && cd build && make compare_qr_double

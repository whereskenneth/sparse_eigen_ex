#include <iostream>
#define EIGEN_USE_MKL_ALL 1
#include <Eigen/SparseCore>
#include <Eigen/PardisoSupport>
#include <unsupported/Eigen/SparseExtra>
#include <mkl_pardiso.h>
#include <fstream>

int main() {

  int matxSize = 25000;
  int bandwidth = 50;
  Eigen::SparseMatrix<float> sparseA(matxSize, matxSize);
  Eigen::SparseMatrix<float> identity(matxSize, matxSize);
  identity.setIdentity();
  sparseA.resize(matxSize, matxSize);
  sparseA.reserve(Eigen::VectorXi::Constant(matxSize, bandwidth));
  for (int i = 0; i < matxSize; i++) {
    if (i % 100 == 0)
      std::cout << i << std::endl;
    for (int j = 0; j < bandwidth && (i + j) < matxSize; j++) {
      sparseA.insert(i, i + j) = static_cast<float>(i + j) / (j + 1);
    }
  }
  Eigen::SparseMatrix<float> ata = sparseA * sparseA.transpose();
  Eigen::PardisoLU<Eigen::SparseMatrix<float>> solver;

  // Out of core solver
  // https://software.intel.com/en-us/mkl-developer-reference-c-pardiso-iparm-parameter
  solver.pardisoParameterArray()[59] = 2;

  std::cout << "Computing..." << std::endl;
  solver.compute(ata);
  std::cout << "Solving..." << std::endl;
  Eigen::SparseMatrix <float> inverse = solver.solve(identity);
  std::cout << "Computed inverse!" << std::endl;
  std::cout << inverse.rows() << "X" << inverse.cols() << std::endl;
  std::cout << "Writing ouput matrix." << std::endl;
  std::cout << "Number of cols in inverse" << inverse.cols() << std::endl;
  for (int k = 0; k < inverse.outerSize(); k++) {
    for (Eigen::SparseMatrix<float>::InnerIterator it(inverse, k); it; ++it) {
      std::cout << it.value() << " " << it.row() << " " << it.col() << " " << it.index() << std::endl;
    }
  }
  //Eigen::saveMarket(inverse., "Inverse.mtx");
//  if (of.is_open()) {
//    std::cout << inverse.(0, 0);
//    of.close();
//  }

  std::cout << "Hello, World!" << std::endl;
  return 0;
}
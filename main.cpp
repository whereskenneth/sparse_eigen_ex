#include <iostream>
#define EIGEN_USE_MKL_ALL 1
#include <Eigen/SparseCore>
#include <Eigen/PardisoSupport>
#include <unsupported/Eigen/SparseExtra>
#include <mkl_pardiso.h>
#include <fstream>

using Eigen::SparseMatrix;
using Eigen::SparseVector;
using Eigen::PardisoLU;

void solve_full_identity(
    const SparseMatrix<float> &ata,
    PardisoLU<SparseMatrix<float>> &solver) {

  SparseMatrix<float> identity(ata.rows(), ata.cols());
  identity.setIdentity();
  std::cout << "Solving..." << std::endl;
  SparseMatrix <float> inverse = solver.solve(identity);
  if (solver.info() != Eigen::Success) {
    std::cout << "Failed to solve identity column" << std::endl;
    return;
  }
  std::cout << "Computed inverse!" << std::endl;
  std::cout << inverse.rows() << "X" << inverse.cols() << std::endl;
  std::cout << "Writing ouput matrix." << std::endl;
  std::cout << "Number of cols in inverse" << inverse.cols() << std::endl;
}

void solve_iterative_identity(
    const SparseMatrix<float> &ata,
    PardisoLU<SparseMatrix<float>> &solver) {
  SparseMatrix<float> identity(ata.rows(), ata.cols());
  identity.setIdentity();
  for (int i = 0; i < identity.cols(); i++) {
    if (i % 100 == 0) {
      std::cout << "Solving rhs # " << i << std::endl;
    }
    SparseMatrix<float> col = identity.col(i);
    SparseMatrix<float> sparse_lhs = solver.solve(col);
    if (solver.info() != Eigen::Success) {
      std::cout << "Failed to solve identity column" << std::endl;
      return;
    }
  }
}

int main() {

  int matxSize = 25000;
  int bandwidth = 50;
  Eigen::SparseMatrix<float> sparseA(matxSize, matxSize);
  sparseA.resize(matxSize, matxSize);
  sparseA.reserve(Eigen::VectorXi::Constant(matxSize, bandwidth));
  for (int i = 0; i < matxSize; i++) {
    for (int j = 0; j < bandwidth && (i + j) < matxSize; j++) {
      sparseA.insert(i, i + j) = static_cast<float>(i + j) / (j + 1);
    }
  }

  // Make ATA so we have pos semi definite matrix to solve
  Eigen::SparseMatrix<float> ata = sparseA * sparseA.transpose();
  Eigen::PardisoLU<Eigen::SparseMatrix<float>> solver;
  std::cout << "Computing..." << std::endl;
  // Out of core solver
  // https://software.intel.com/en-us/mkl-developer-reference-c-pardiso-iparm-parameter
  solver.pardisoParameterArray()[59] = 1;
  solver.compute(ata);
  if (solver.info() != Eigen::Success) {
    std::cout << "Failed to copmute ATA." << std::endl;
    return -1;
  }


  //solve_full_identity(ata, solver);
  solve_iterative_identity(ata, solver);
  return 0;
}

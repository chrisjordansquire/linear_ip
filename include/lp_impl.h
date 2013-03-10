#ifndef LINEAR_IP_IMPL_H
#define LINEAR_IP_IMPL_H

/**
 * @file lp_impl.h
 * @brief The implementation class of an interior point linear programming solver.
 * 
 * @author Chris Jordan-Squire
 * @date 03/02/2013
 */

#include "lp.h"
#include <iostream>
#include <sstream>
#include <cmath>
#include <cassert>

namespace linear_ip{

typedef Eigen::ColPivHouseholderQR<Matrix> cph_qr;
typedef cph_qr::PermutationType cph_qr_perm;
typedef Eigen::TriangularView<const Matrix, Eigen::Upper> uc_triang;
typedef Eigen::TriangularView<Matrix, Eigen::Upper> u_triang;

/** @brief POD Container for residuals used to determine the
 * update directions
 */
struct residuals{
    /**@brief Ax-b */
    Matrix b;
    /**@brief A^T*lam + s-c*/
    Matrix c;
    /**@brief XSe */
    Matrix xs;};

/** @brief POD Container for the update directions for the Newton
 * step.
 */
struct directions{
    Matrix lam;
    Matrix s;
    Matrix x;}; 

/** @relates lp_impl
 * Compute the stepsize for the corrector direction
 * in the predictor-corrector method.
 * @param v The vector to be updated.
 * @param dv The direction to take a step in.
 * @param eta Damping parameter so v + stepsize*dv stays off the
 * boundary. Assumed between 0 and 1.
 * @return The stepsize.*/
double corrector_stepsize(const Matrix &v, const Matrix &dv, double eta);

/** @relates lp_impl
 * Compute the maximum stepsize in the direction dx which maintains* x>=0.
 * @param x The current solution.
 * @param dx Direction to update x.
 * @return Largest stepsize which mainstains x>=0.*/
double min_ratio(const Vector &x, const Vector &dx); 

class lp_impl{
    public: 
        /** @brief The cost vector */
        const Vector c_;
        /** @brief The constraint matrix*/
        const Matrix A_;
        /** @brief The constraint vector*/
        const Vector b_;
        /** @brief The current estimated primal solution.*/
        Vector x_;
        /** @brief The current estimated dual solution for 
         * the constraints Ax=b.*/
        Vector lam_;
        /** @brief The current estimated dual solution for the 
         * constraints x>=0.*/
        Vector s_;
        /** @brief Has the problem been solved.*/
        bool is_solved_;
        /** @brief The number of rows of A; the dual dimension.*/
        int rows_;
        /** @brief The number of columns of A; the primal dimension.*/
        int cols_;
        /** @brief The tolerance used to test convergence. 
         * By default the tolerance is set to 1e-6. */
        double tol_;
        /** @brief The maximum number of iterations in the interior 
         * point method. 
         * By default this is set to 10. */
        int max_itr_;

        /** @brief The (only) constructor.
         * @param c The cost vector
         * @param A The constraint matrix
         * @param b The constraint vector
         */
        lp_impl(Vector c, Matrix A, Vector b);

        /** @brief Initializes the primal and dual solutions. 
         * This initializes the primal and dual solutions using
         * the method described in Section 14.2 of Nocedal/Wright
         * 2nd edition. */
        void init();
        
        /** Use the Mehrota predictor-corrector interior point
         * solver to solver the LP instance.*/
        void solve();

        /** @brief Prints the problem information
         * Mainly used for debugging. 
         * @return A string containing the vectors and matrix
         * used to construct the instance. */
        std::string print_prob();

        /** In the interior point solver, test at the end of
         * each iteration of the predictor-corrector method if
         * the solution is optimal. */
        bool test_convergence();
        /** Compute the residuals, i.e. the vector solved for
         * in the Newton method each iteration.
         * @param r The residuals for the current primal and 
         * dual variables*/
        void compute_residuals(residuals &r);
        /** Compute the lambda direction for a given set of residuals,
         * i.e. the update direction for the constraints Ax=b.
         * @param r The residuals for the current primal and dual 
         * variables.
         * @param M_LDL The Cholesky factorization of a AXS^(-1)A^T. 
         * @return The update direction for lambda.
         */
        Matrix compute_dlam(const residuals &r, 
                const Eigen::LDLT<Matrix> &M_LDL);
        /** Compute the Newton direction for a given
         * residual vector. 
         * @param d The new direction vectors are placed in this.
         * @param r The residual vectors.
         * @param M_LDL The Cholesky factorization of AXS^(-1)A^T.*/
        void compute_dir(directions &d,
                const residuals &r,
                const Eigen::LDLT<Matrix> &M_LDL);

};
}
#endif

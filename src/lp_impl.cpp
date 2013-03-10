#include "lp_impl.h"

namespace linear_ip{

lp_impl::lp_impl(Vector c, 
        Matrix A, Vector b):
    A_(A), b_(b), c_(c){

    rows_ = A_.rows();
    cols_ = A_.cols();

    if( rows_ > cols_ ){
        std::cout << "Methods for initializing will fail: too many "
            "constraints.";
        std::cout << std::endl;
    }

    is_solved_ = false;
    max_itr_ = 10;
    tol_ = 1e-6;

}

std::string lp_impl::print_prob(){

    std::stringstream strm;
    strm << c_ << "\n" << A_ << "\n" << b_ << "\n";
    return strm.str();
}


void lp_impl::solve(){
    
    init();

    bool converged = false;

    Matrix M(rows_, rows_);
    Eigen::LDLT<Matrix> M_LDL;

    residuals r;
    directions d;

    double eta = 0.9;
    int itr=0;


    while( !converged && itr <= max_itr_){
        itr++;
        double mu = x_.dot(s_) / cols_;
        double sigma;
        double mu_aff;

        //cholesky factor the normal bit
        M = A_ * x_.asDiagonal() * s_.asDiagonal().inverse() * A_.transpose();
        M_LDL = M.ldlt();

        //predictor step
        compute_residuals(r);
        compute_dir(d, r, M_LDL);

        double alpha_aff_pri = min_ratio(x_, d.x);
        double alpha_aff_dual = min_ratio(s_, d.s);

        mu_aff = (x_+alpha_aff_pri*d.x).dot(s_+alpha_aff_dual*d.s) / cols_;
        sigma = std::pow(mu_aff/mu, 3);

        //corrector step
        r.xs.noalias() += d.x.cwiseProduct(d.s) - Eigen::MatrixXd::Constant(cols_,1,sigma*mu);

        compute_dir(d, r, M_LDL);

        double alpha_pri = corrector_stepsize(x_, d.x, eta);
        double alpha_dual = corrector_stepsize(s_, d.s, eta);

        eta = 1 - 0.5 * (1-eta);
        x_ += alpha_pri * d.x;
        lam_ += alpha_dual * d.lam;
        s_ += alpha_dual * d.s;

        if( test_convergence() ){
            is_solved_ = true;
            converged = true;
        }

    }

}

void lp_impl::compute_residuals(residuals &r){
        r.c = A_.transpose() * lam_ + s_ - c_;
        r.b = A_ * x_ - b_;
        r.xs = x_.asDiagonal() * s_;
}

void lp_impl::compute_dir(directions &d,
        const residuals &r,
        const Eigen::LDLT<Matrix> &M_LDL){

        d.lam = compute_dlam(r, M_LDL);        

        d.s = -r.c - A_.transpose() * d.lam;

        d.x = -s_.cwiseInverse().cwiseProduct(r.xs);
        d.x.noalias() -= x_.cwiseProduct(s_.cwiseInverse().cwiseProduct(d.s));
}

Matrix lp_impl::compute_dlam(const residuals &r,
        const Eigen::LDLT<Matrix> &M_LDL){

    Matrix dlam; 
    dlam = -r.b - A_* x_.cwiseProduct(s_.cwiseInverse().cwiseProduct(r.c)) + A_ * s_.cwiseInverse().cwiseProduct(r.xs);

    return M_LDL.solve(dlam);
}


bool lp_impl::test_convergence(){
        
    //TODO:Check individual complimentarity
    
    bool duality_gap = (x_.dot(c_) - lam_.dot(b_)) < tol_;
    bool x_feasibility = (A_ * x_ - b_).norm() < tol_; 
    bool s_feasibility = (A_.transpose() * lam_ + s_ - c_).norm() < tol_;

    return duality_gap && x_feasibility && s_feasibility;
}


void lp_impl::init(){
    const cph_qr qr_fact(A_.transpose());

    //Do the following declarations make copies? 
    //I don't think the R or P do, but I think the Q might
    
    const Matrix Q(qr_fact.householderQ());

    Matrix tmp(qr_fact.matrixQR());
    Matrix tmp2 = tmp.block(0, 0, rows_, rows_);
    const u_triang R1 = tmp2.triangularView<Eigen::Upper>(); 
    const cph_qr_perm &P = qr_fact.colsPermutation();


    //first pass
    x_ = P.inverse() * b_;
    R1.solveInPlace(x_);
    x_ = Q.leftCols(rows_)*x_;

    lam_ = qr_fact.solve(c_); //according to ggael on SO, this is least-squares
    s_ = c_ - A_.transpose()*lam_;

    //second pass
    //make x and s non-negative
    double min_x = x_.minCoeff();
    if(min_x < 0){
        x_ += Matrix::Constant(cols_, 1, (-1.5) * min_x );
    }
    else if(min_x < tol_){
        x_ += Matrix::Constant(cols_, 1, 100*tol_);
    }

    double min_s = s_.minCoeff();
    if(min_s < 0){
        s_ += Matrix::Constant(cols_, 1, (-1.5) * min_s);
    }
    else if(min_s == 0){
        s_ += Matrix::Constant(cols_, 1, 100*tol_);
    }

    //third pass
   double xt_s = x_.dot(s_);
   double x_norm = x_.lpNorm<1>();
   double s_norm = s_.lpNorm<1>();

   double delta_x = 0.5 * xt_s / s_norm;
   double delta_s = 0.5 * xt_s / x_norm;

   x_ += Matrix::Constant(cols_, 1, delta_x);
   s_ += Matrix::Constant(cols_, 1, delta_s);

}


double corrector_stepsize(const Matrix &v, 
        const Matrix &dv, double eta){

    double stepsize = min_ratio(v, dv);
    stepsize *= eta;
    stepsize = std::min<double>(stepsize, 1);
    return stepsize;
}

double min_ratio(const linear_ip::Vector &x, 
        const linear_ip::Vector &dx){

    double mr = 1;
    assert(x.size() == dx.size());

    for(int i=0; i<x.size(); i++){
        if( dx(i)<0 && -x(i)/dx(i)<mr ){
            mr = -x(i)/dx(i);
        }
    }

    return mr;
}

}

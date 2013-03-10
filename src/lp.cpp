#include "lp.h"
#include "lp_impl.h"


namespace linear_ip{

 lp::lp(Vector c, Matrix A, Vector b){
    ip = new lp_impl(c, A, b);
}

 bool lp::is_solved() const{
    return ip->is_solved_;
}

 void lp::solve(){
    ip->solve();
}

 Vector lp::get_x() const{
    assert(is_solved());
    return ip->x_;
}

 Vector lp::get_lam() const{
    assert(is_solved());
    return ip->lam_;
}

 Vector lp::get_s() const{
    assert(is_solved());
    return ip->s_;
}

 Matrix lp::get_A() const{
    return ip->A_;
}

 Vector lp::get_c() const{
    return ip->c_;
}

 Vector lp::get_b() const{
    return ip->b_;
}

 void lp::set_max_itr(int max_itr){
    assert(max_itr>0);
    ip->max_itr_ = max_itr; 
}

void lp::set_tol(double tol){
    assert(tol>0);
    ip->tol_ = tol;
}

}

#include "gtest/gtest.h"
#include "lp_impl.h"

TEST(Basic, Constructor){
    linear_ip::Matrix A(2,3);
    linear_ip::Vector c(3), b(2);

    for(int i=0; i<c.rows(); ++i){
        c(i)=3*(i+1);
    }

    for(int i=0; i<b.rows(); ++i){
        b(i)=-i-1;
    }

    for(int i=0; i<A.rows(); ++i){
        for(int j=0; j<A.cols(); ++j){
            A(i,j) = 2*(i+1)-3*(j+2);
        }
    }


    linear_ip::lp_impl P(c, A, b);


    EXPECT_EQ(A, P.A_);
    EXPECT_EQ(b, P.b_);
    EXPECT_EQ(c, P.c_);

}


TEST(Basic, PrintProb){
    linear_ip::Vector c(1), b(1);
    linear_ip::Matrix A(1,1);

    c(0)=1;
    b(0)=3;
    A(0,0)=2;
    
    linear_ip::lp_impl P(c,A,b);

    std::string expected("1\n2\n3\n");
    std::string actual(P.print_prob());

    EXPECT_EQ(expected, actual);
}
 
TEST(Basic, min_ratio){
    linear_ip::Vector x(3), dx(3);

    for(int i=0; i<3; i++){
        x(i) = i-1;
        dx(i) = 2*(1-i);
    }

    double actual;
    
    actual = linear_ip::min_ratio(x, dx);
    EXPECT_EQ(0.5, actual);

    x(1) = 3;
    dx(1) = -12;
    actual = linear_ip::min_ratio(x, dx);
    EXPECT_EQ(0.25, actual);

}

TEST(Basic, corrector_stepsize){
    linear_ip::Vector x(3), dx(3);

    for(int i=0; i<3; i++){
        x(i) = i-1;
        dx(i) = 2*(1-i);
    }
    double eta = 0.8;

    double actual = linear_ip::corrector_stepsize(x, dx, eta);
    EXPECT_EQ(0.4, actual);
}

class TestProblemVerySimple : public testing::Test{
    public:
        linear_ip::Matrix A;
        linear_ip::Vector b, c;
        linear_ip::lp_impl* ptr_P;

        TestProblemVerySimple(): A(2,2), b(2), c(2) {}

        virtual void SetUp(){
            A(0,0)=0;
            A(0,1)=1;
            A(1,0)=1;
            A(1,1)=0;

            b(0)=2;
            b(1)=3;

            c(0)=-1;
            c(1)=-5;

            ptr_P = new linear_ip::lp_impl(c,A,b);
        }

        virtual void TearDown(){
            delete ptr_P;
        }
};

TEST_F(TestProblemVerySimple, init){
    linear_ip::Vector x(2), s(2), lam(2);

    x << 4.25, 3.25;
    lam << -5, -1;
    s << 0.00015, 0.00015;

    ptr_P->init();

    EXPECT_TRUE(x.isApprox(ptr_P->x_));
    EXPECT_TRUE(lam.isApprox(ptr_P->lam_));
    EXPECT_TRUE(s.isApprox(ptr_P->s_));
}
class TestMemberHelpers : public testing::Test{
    public:
        linear_ip::Matrix A;
        linear_ip::Vector b, c, s, x;
        linear_ip::lp_impl* ptr_P;
        Eigen::LDLT<linear_ip::Matrix> M_LDL;

        TestMemberHelpers(): A(2,4), b(2), c(4), s(4), x(4) {}

        virtual void SetUp(){

            A << 1,1,1,0,-1,1,0,1;
            b << 2,1;
            c << 1,2,0,0;
            x << 2, 1, 0.5, 1;
            s << 1, 2, 1, 0.5;

            ptr_P = new linear_ip::lp_impl(c,A,b);
            
            ptr_P->x_.resize(4);
            ptr_P->s_.resize(4);
            ptr_P->lam_.resize(2);

            ptr_P->x_ = x;
            ptr_P->s_ = s;
            ptr_P->lam_ << 2,1;

            M_LDL = (A * x.asDiagonal() * s.asDiagonal().inverse() * A.transpose()).ldlt();

       }

        virtual void TearDown(){
            delete ptr_P;
        }
};


TEST_F(TestMemberHelpers, compute_residuals){

    linear_ip::residuals r;
    ptr_P->compute_residuals(r);


    linear_ip::Matrix rc(4,1), rb(2,1), rxs(4,1);
    
    rc << 1, 3, 3, 1.5;
    rb << 1.5, -1;
    rxs << 2, 2, 0.5, 0.5;

    EXPECT_TRUE(rc.isApprox(r.c, 1e-5));
    EXPECT_TRUE(rb.isApprox(r.b, 1e-5));
    EXPECT_TRUE(rxs.isApprox(r.xs, 1e-5));
}

TEST_F(TestMemberHelpers, compute_dlam){

    linear_ip::residuals r;
    ptr_P->compute_residuals(r);
    
    linear_ip::Matrix expected(2,1);
    expected << -1.4, -0.8;

    linear_ip::Matrix actual = ptr_P->compute_dlam(r, M_LDL);

    EXPECT_TRUE(expected.isApprox(actual, 1e-5));
}

TEST_F(TestMemberHelpers, compute_dir){
    linear_ip::residuals r;
    ptr_P->compute_residuals(r);

    linear_ip::directions actual;
    ptr_P->compute_dir(actual, r, M_LDL);

    linear_ip::Matrix dlam(2,1), ds(4,1), dx(4,1);
    dlam << -1.4, -0.8;
    ds << -0.4, -0.8, -1.6, -0.7;
    dx << -1.2, -0.6, 0.3, 0.4;

    EXPECT_TRUE(dlam.isApprox(actual.lam, 1e-5));
    EXPECT_TRUE(ds.isApprox(actual.s, 1e-5));
    EXPECT_TRUE(dx.isApprox(actual.x, 1e-5));
}
 

class TestProblemBertsimasTsitsiklis : public testing::Test{
    public:
        linear_ip::Matrix A;
        linear_ip::Vector b, c;
        linear_ip::lp_impl* ptr_P;

        TestProblemBertsimasTsitsiklis(): A(2,4), b(2), c(4) {}

        virtual void SetUp(){

            A << 1,1,1,0,-1,1,0,1;
            b << 2,1;
            c << 1,2,0,0;

            ptr_P = new linear_ip::lp_impl(c,A,b);
        }

        virtual void TearDown(){
            delete ptr_P;
        }
};

TEST_F(TestProblemBertsimasTsitsiklis, init){
    ptr_P->init();

    linear_ip::Vector x(4), lam(2), s(4);

    x << 0.642157 ,1.30882, 0.97549, 0.642157;
    lam << 1, 0.333333;
    s << 2.58333, 2.91667, 1.25, 1.91667;

    EXPECT_TRUE(x.isApprox(ptr_P->x_, 1e-5));
    EXPECT_TRUE(lam.isApprox(ptr_P->lam_, 1e-5));
    EXPECT_TRUE(s.isApprox(ptr_P->s_, 1e-5));

}

TEST_F(TestProblemBertsimasTsitsiklis, solve){
    ptr_P->solve();

    linear_ip::Vector x(4), s(4), lam(2);

    x << 0, 0, 2, 1;
    s << 1, 2, 0, 0;

    EXPECT_TRUE(x.isApprox(ptr_P->x_, 1e-5));
    EXPECT_TRUE(ptr_P->lam_.norm() < 1e-5);
    EXPECT_TRUE(s.isApprox(ptr_P->s_, 1e-5));
}


TEST(Solve, Simple){
    linear_ip::Vector c(5), b(3);
    linear_ip::Matrix A(3, 5);

    //init b
    b << 6, 12, 10;

    //init c
    c << 2, 1, 0, 0, 0;

    //init A
    A << 1, 1, -1, 0, 0,
        1, 4, 0 , -1, 0,
        1, 2, 0, 0, 1;

    linear_ip::lp_impl P(c, A, b);

    P.solve();

    linear_ip::Vector x(5), s(5), lam(3);

    x << 2, 4, 0, 6, 0;
    lam << 3, 0, -1;
    s << 0, 0, 3, 0, 1;

    EXPECT_TRUE(x.isApprox(P.x_, 1e-5));
    EXPECT_TRUE(lam.isApprox(P.lam_, 1e-5));
    EXPECT_TRUE(s.isApprox(P.s_, 1e-5));

}



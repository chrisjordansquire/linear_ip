#include "gtest/gtest.h"
#include "lp.h"

TEST(Interface, Simple){

    linear_ip::Vector c(5) ,b(3);
    linear_ip::Matrix A(3, 5);

    b << 6, 12, 10;
    c << 2, 1, 0, 0, 0;
    A << 1, 1, -1, 0, 0,
        1, 4, 0, -1, 0,
        1, 2, 0, 0, 1;

    linear_ip::lp P(c, A, b);

    P.solve();

    linear_ip::Vector x(5), s(5), lam(3);
    
    x << 2, 4, 0, 6, 0;
    s << 0, 0, 3, 0, 1;
    lam << 3, 0, -1;

    EXPECT_TRUE(x.isApprox(P.get_x(), 1e-5));
    EXPECT_TRUE(s.isApprox(P.get_s(), 1e-5));
    EXPECT_TRUE(lam.isApprox(P.get_lam(), 1e-5));

}

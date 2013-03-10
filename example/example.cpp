#include <iostream>
#include "lp.h"

int main(){


    linear_ip::Vector c(5) ,b(3);
    linear_ip::Matrix A(3, 5);

    b << 6, 12, 10;
    c << 2, 1, 0, 0, 0;
    A << 1, 1, -1, 0, 0,
        1, 4, 0, -1, 0,
        1, 2, 0, 0, 1;

    linear_ip::lp P(c, A, b);

    P.solve();

    std::cout << "x=\n" << P.get_x()<< "\n" << std::endl;
    std::cout << "lambda=\n" << P.get_lam() << "\n" << std::endl;
    std::cout << "s=\n" << P.get_s() << "\n" << std::endl;

}

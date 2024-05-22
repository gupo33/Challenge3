#include <vector>
#include <iostream>
#include <cmath>
#include <numbers>

int main(int argc, char* argv[]){
    int k = 0;
    int maxit = 1000;
    int npoints = 20;
    double a = -1;
    double b = 1;
    double h = (b-a)/((double)(npoints-1));

    double tol = 0.000001;

    std::cout << h << std::endl;

    std::vector<std::vector<double>>U;
    std::vector<std::vector<double>>Unew;
    std::vector<std::vector<double>>mesh;

    //generate matrix

    for(int i = 0; i<npoints;++i){
        U.emplace_back(npoints);
        Unew.emplace_back(npoints);
    }

    //impose boundary conditions

    for(int i = 0;i<npoints;++i){
        U[i][0] = 0;
        U[0][i] = 0;
        U[npoints-1][i] = 0;
        U[i][npoints-1] = 0;

        Unew[i][0] = 0;
        Unew[0][i] = 0;
        Unew[npoints-1][i] = 0;
        Unew[i][npoints-1] = 0;
    }

    //copy into another matrix

    Unew = U;

    //create mesh

    //create f

    auto f = [](double x,double y){return 8*M_PI*M_PI*std::sin(2*M_PI*x)*std::sin(2*M_PI*y);};

    #ifdef DEBUG
        for(auto vec : U){
            for(auto num : vec){
                std::cout << num << " ";
            }
            std::cout << std::endl;
        }
    #endif

    //commence

    double error = tol+1;

    while(k < maxit && error > tol){

        //update matrix
        for(int i = 1;i<npoints-1;++i){
            for(int j=1;j<npoints-1;++j){
                Unew[i][j] = 0.25 * (U[i-1][j] + U[i+1][j] + U[i][j-1] + U[i][j+1] + h*h*f(a+i*h,b+j*h));
            }
        }

        //compute error
        error = 0;
        for(int i = 1;i<npoints-1;++i){
            for(int j=1;j<npoints-1;++j){
                error += (U[i][j] - Unew[i][j]) * (U[i][j] - Unew[i][j]);
            }
        }
        error *= h;
        error = std::sqrt(error);

        U.swap(Unew);
        k++;
    }

    std::cout << k << std::endl;

    #ifdef DEBUG
        for(auto vec : U){
            for(auto num : vec){
                std::cout << num << " ";
            }
            std::cout << std::endl;
        }
    #endif

    return 0;

}
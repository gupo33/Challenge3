#include <vector>
#include <iostream>
#include <cmath>
#include <numbers>
#include <chrono>
#include <mpi.h>

int main(int argc, char* argv[]){

    MPI_Init(&argc, &argv);

    //parameters for the method

    int k = 0;
    int maxit = 1000;
    int npoints = 20;
    double a = -1;
    double b = 1;
    double h = (b-a)/((double)(npoints-1));
    double tol = 0.000001;

    auto f = [](double x,double y){return 8*M_PI*M_PI*std::sin(2*M_PI*x)*std::sin(2*M_PI*y);};
    double bc = 5;

    //parameters for parallel execution

    int rank,size;

    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size (MPI_COMM_WORLD,&size);

    static std::chrono::_V2::system_clock::time_point t_start, t_end;

    using DataStructure = std::vector<std::vector<double>>;

    DataStructure U;
    DataStructure Unew;

    t_start = std::chrono::high_resolution_clock::now();

    //generate each local matrix (assuming that the size is compatible with the number of processors)

    int nrows = npoints / size;
    int &ncols = npoints;

    for(int i = 0; i<nrows;++i){
        U.emplace_back(ncols);
        Unew.emplace_back(ncols);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    //impose boundary conditions

    for(int i = 0;i<nrows;++i){
        U[i][0] = bc;
        U[i][ncols-1] = bc;

        Unew[i][0] = bc;
        Unew[i][ncols-1] = bc;
    }

    if(rank == 0){
        for(int i = 1; i<ncols-1;++i){
            U[0][i] = bc;
            Unew[0][i] = bc;
        }
    }

    if(rank == size-1){
        for(int i = 1; i<ncols-1;++i){
            U[nrows-1][i] = bc;
            Unew[nrows-1][i] = bc;
        }
    }

    #ifdef DEBUG
        MPI_Barrier(MPI_COMM_WORLD);
        for(int i = 0; i<size;++i){
            if(rank == i){
                std::cout<<"U in rank "<<rank<<std::endl;
                for(auto vec : U){
                    for(auto num : vec){
                        std::cout << num << " ";
                    }
                    std::cout << std::endl;
                }
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        if(rank == 0){
            std::cout << "Assembled initial matrix of points" << std::endl;
        }
    #endif

    //commence

    /*

    double error = tol+1;

    while(k < maxit && error > tol){

        //update matrix
        for(int i = 0;i<npoints-1;++i){
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

    */

    //to properly update the elements at the edges of each local U, we must store the top row from the next process
    //and the bottom row from the previous process, when necessary

    std::vector<double> upper(ncols);
    std::vector<double> lower(ncols);

    double error = tol+1;

    while(k < maxit && error > tol){

        MPI_Barrier(MPI_COMM_WORLD);

        //reset error

        error = 0;

    
        //ok, so:
        //everyone but the last rank should send the last to the next
        //everyone but the first rank should send the first to the previous

        //then:
        //everyone but the first rank should receive upper from previous
        //everyone but the last rank should receive lower from next
        if(rank!=0){
            MPI_Send(U[0].data(),ncols,MPI_DOUBLE,rank-1,rank,MPI_COMM_WORLD);
            #ifdef DEBUG
                std::cout << "sent first row from " << rank << "to "<<rank-1<<std::endl;
            #endif
        }
        if(rank!=size-1){
            MPI_Send(U[nrows-1].data(),ncols,MPI_DOUBLE,rank+1,rank,MPI_COMM_WORLD);
            #ifdef DEBUG
                std::cout << "sent last row from " << rank << "to "<<rank+1<<std::endl;
            #endif
        }

        if(rank!=0){
            MPI_Recv(upper.data(),ncols,MPI_DOUBLE,rank-1,rank-1,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
            #ifdef DEBUG
                std::cout << rank <<" received upper from " << rank-1 << std::endl;
            #endif
        }
        if(rank!=size-1){
            MPI_Recv(lower.data(),ncols,MPI_DOUBLE,rank+1,rank+1,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
            #ifdef DEBUG
                std::cout <<rank<< " received lower from " << rank+1 <<std::endl;
            #endif
        }

        //update matrix

        if(rank > 0 && rank < size-1){
            for(int i = 0;i<nrows;++i){
                for(int j=1;j<ncols-1;++j){
                    if(i-1 < 0){
                        Unew[i][j] = 0.25 * (upper[j] + U[i+1][j] + U[i][j-1] + U[i][j+1] + h*h*f(a+i*h,b+j*h));
                    }
                    else{
                        if(i+1>=nrows){
                            Unew[i][j] = 0.25 * (U[i-1][j] + lower[j] + U[i][j-1] + U[i][j+1] + h*h*f(a+i*h,b+j*h));
                        }
                        else{
                        Unew[i][j] = 0.25 * (U[i-1][j] + U[i+1][j] + U[i][j-1] + U[i][j+1] + h*h*f(a+i*h,b+j*h));
                        }
                    }
                    error += (U[i][j] - Unew[i][j]) * (U[i][j] - Unew[i][j]);
                }
            }
        }
        else{
            if(rank == 0){ //don't update first row
                for(int i = 1;i<nrows;++i){
                    for(int j=1;j<ncols-1;++j){
                        if(i+1>=nrows){
                            Unew[i][j] = 0.25 * (U[i-1][j] + lower[j] + U[i][j-1] + U[i][j+1] + h*h*f(a+i*h,b+j*h));
                        }
                        else{
                        Unew[i][j] = 0.25 * (U[i-1][j] + U[i+1][j] + U[i][j-1] + U[i][j+1] + h*h*f(a+i*h,b+j*h));
                        }
                        error += (U[i][j] - Unew[i][j]) * (U[i][j] - Unew[i][j]);
                    }
                }
            }
            else{ //don't update last row
                for(int i = 0;i<nrows-1;++i){
                    for(int j=1;j<ncols-1;++j){
                        if(i-1 < 0){
                            Unew[i][j] = 0.25 * (upper[j] + U[i+1][j] + U[i][j-1] + U[i][j+1] + h*h*f(a+i*h,b+j*h));
                        }
                        else{
                            Unew[i][j] = 0.25 * (U[i-1][j] + U[i+1][j] + U[i][j-1] + U[i][j+1] + h*h*f(a+i*h,b+j*h));
                        }
                        error += (U[i][j] - Unew[i][j]) * (U[i][j] - Unew[i][j]);
                    }
                }
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);

        //compute errors

        MPI_Reduce(MPI_IN_PLACE,&error,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
        if(rank == 0){
            error *= h;
            error = std::sqrt(error);
        }
        MPI_Bcast(&error,1,MPI_DOUBLE,0,MPI_COMM_WORLD);

        U.swap(Unew);
        k++;
    }

    t_end = std::chrono::high_resolution_clock::now();

    #ifdef DEBUG
        for(int i = 0; i<size;++i){
            if(rank == i){
                std::cout<<"U in rank "<<rank<<std::endl;
                for(auto vec : U){
                    for(auto num : vec){
                        std::cout << num << " ";
                    }
                    std::cout << std::endl;
                }
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        if(rank == 0){
            std::cout << "Finished computing solution in points" << std::endl;
        }
    #endif

    std::cout << "Elapsed time is " << (t_end-t_start).count()*1E-9 << " seconds\n";

    MPI_Finalize();

    return 0;

}
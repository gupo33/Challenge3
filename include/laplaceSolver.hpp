#include <vector>
#include <iostream>
#include <cmath>
#include <numbers>
#include <mpi.h>
#include <functional>

using fun_type = std::function<double(std::vector<double>)>;

bool laplaceSolver(const unsigned max_it,const int a, const int b, const double tol, const unsigned npoints, const double bc, fun_type f){

    double h = (b-a)/((double)(npoints-1)); //distance between each point of the grid

    unsigned k = 0; //number of iterations

    int rank,size;

    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size (MPI_COMM_WORLD,&size);

    using DataStructure = std::vector<std::vector<double>>;

    DataStructure U;
    DataStructure Unew;

    //generate each local matrix (assuming that the size is compatible with the number of processors)

    const unsigned nrows = npoints / size;
    auto& ncols = npoints;

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

    /*

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

    */

    //commence

    //to properly update the elements at the edges of each local U, we must store the top row from the next process
    //and the bottom row from the previous process, when necessary

    std::vector<double> upper(ncols);
    std::vector<double> lower(ncols);

    double error = 0;

    bool global_conv = false; //convergence for all processes
    bool local_conv = false; //convergence for each process

    while(k < max_it && !global_conv){

        MPI_Barrier(MPI_COMM_WORLD);

        //reset error

        error = 0;

        //everyone but the last rank should send the last to the next
        //everyone but the first rank should send the first to the previous

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

        //update each local matrix 

        if(rank > 0 && rank < size-1){
            for(int i = 0;i<nrows;++i){
                for(int j=1;j<ncols-1;++j){
                    if(i-1 < 0){
                        Unew[i][j] = 0.25 * (upper[j] + U[i+1][j] + U[i][j-1] + U[i][j+1] + h*h*f({a+i*h,b+j*h}));
                    }
                    else{
                        if(i+1>=nrows){
                            Unew[i][j] = 0.25 * (U[i-1][j] + lower[j] + U[i][j-1] + U[i][j+1] + h*h*f({a+i*h,b+j*h}));
                        }
                        else{
                        Unew[i][j] = 0.25 * (U[i-1][j] + U[i+1][j] + U[i][j-1] + U[i][j+1] + h*h*f({a+i*h,b+j*h}));
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
                            Unew[i][j] = 0.25 * (U[i-1][j] + lower[j] + U[i][j-1] + U[i][j+1] + h*h*f({a+i*h,b+j*h}));
                        }
                        else{
                        Unew[i][j] = 0.25 * (U[i-1][j] + U[i+1][j] + U[i][j-1] + U[i][j+1] + h*h*f({a+i*h,b+j*h}));
                        }
                        error += (U[i][j] - Unew[i][j]) * (U[i][j] - Unew[i][j]);
                    }
                }
            }
            else{ //don't update last row
                for(int i = 0;i<nrows-1;++i){
                    for(int j=1;j<ncols-1;++j){
                        if(i-1 < 0){
                            Unew[i][j] = 0.25 * (upper[j] + U[i+1][j] + U[i][j-1] + U[i][j+1] + h*h*f({a+i*h,b+j*h}));
                        }
                        else{
                            Unew[i][j] = 0.25 * (U[i-1][j] + U[i+1][j] + U[i][j-1] + U[i][j+1] + h*h*f({a+i*h,b+j*h}));
                        }
                        error += (U[i][j] - Unew[i][j]) * (U[i][j] - Unew[i][j]);
                    }
                }
            }
        }
        

        U.swap(Unew);

        //compute errors and check for convergence in each rank

        error *= h;
        error = std::sqrt(error);

        local_conv = error <= tol;

        MPI_Allreduce(&local_conv,&global_conv,1,MPI_CXX_BOOL,MPI_LAND,MPI_COMM_WORLD); //check for global convergence

        #ifdef DEBUG
            std::cout << "error for rank " << rank << " is " << error << "," << "local convergence is " << local_conv << std::endl;
            if(rank == 0){
                std::cout << "global convergence is " << global_conv << std::endl;
            }
        #endif
        k++;
    }

    /*

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

    */

    return global_conv;

}
#include <vector>
#include <map>
#include <iostream>
#include <cmath>
#include <numbers>
#include <mpi.h>
#include <functional>
#include "writeVTK.hpp"

using fun_type = std::function<double(std::vector<double>)>;
using DataStructure = std::map<unsigned,std::vector<double>>;

bool laplaceSolver(const unsigned max_it,const double a, const double b, const double tol, const unsigned npoints, const double bc, fun_type f){

    double h = (b-a)/((double)(npoints-1)); //distance between each point of the grid

    unsigned k = 0; //number of iterations

    int rank,size;

    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size (MPI_COMM_WORLD,&size);

    DataStructure U;
    DataStructure Unew;

    //generate each local matrix (assuming that the size is compatible with the number of processors)

    const unsigned nrows = npoints / size;
    auto& ncols = npoints;

    for(unsigned i = 0; i<nrows;++i){
        U.emplace(i,std::vector<double>(ncols));
        Unew.emplace(i,std::vector<double>(ncols));
    }

    //impose boundary conditions

    for(unsigned i = 0;i<nrows;++i){
        U[i][0] = bc;
        U[i][ncols-1] = bc;

        Unew[i][0] = bc;
        Unew[i][ncols-1] = bc;
    }

    if(rank == 0){
        for(unsigned i = 1; i<ncols-1;++i){
            U[0][i] = bc;
            Unew[0][i] = bc;
        }
    }

    if(rank == size-1){
        for(unsigned i = 1; i<ncols-1;++i){
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
                    for(int j=0;j<ncols;++j){
                        std::cout << vec.second[j] << " ";
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

    //to properly update the elements at the edges of each local U, we must store the top row from the next process
    //and the bottom row from the previous process, when necessary

    std::vector<double> upper(ncols);
    std::vector<double> lower(ncols);

    double error = 0; //error for each rank

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

        //update interior of local matrix

        for(unsigned i = 1;i<nrows-1;++i){
            for(unsigned j = 1;j<ncols-1;++j){
                Unew[i][j] = 0.25 * (U[i-1][j] + U[i+1][j] + U[i][j-1] + U[i][j+1] + h*h*f({a+i*h,b+j*h}));
                error += (U[i][j] - Unew[i][j]) * (U[i][j] - Unew[i][j]);
            }
        }

        if(rank > 0 && rank < size-1){
            for(unsigned j =1;j<ncols-1;++j){
                Unew[0][j] = 0.25*(upper[j]+ U[1][j] + U[0][j-1] + U[0][j+1] + h*h*f({a,b+j*h}));
                Unew[nrows-1][j] = 0.25 * (U[nrows-2][j] + lower[j] + U[nrows-1][j-1] + U[nrows-1][j+1] + h*h*f({a+(nrows-1)*h,b+j*h}));
                error += (U[0][j] - Unew[0][j]) * (U[0][j] - Unew[0][j]) + (U[nrows-1][j] - Unew[nrows-1][j]) * (U[nrows-1][j] - Unew[nrows-1][j]);
            }
        }
        else{
            if(rank == 0){
                for(unsigned j =1;j<ncols-1;++j){
                    Unew[nrows-1][j] = 0.25 * (U[nrows-2][j] + lower[j] + U[nrows-1][j-1] + U[nrows-1][j+1] + h*h*f({a+(nrows-1)*h,b+j*h}));
                    error += (U[nrows-1][j] - Unew[nrows-1][j]) * (U[nrows-1][j] - Unew[nrows-1][j]);
                }
            }
            else{
                for(unsigned j =1;j<ncols-1;++j){
                    Unew[0][j] = 0.25*(upper[j]+ U[1][j] + U[0][j-1] + U[0][j+1] + h*h*f({(double)a,b+j*h}));
                    error += (U[0][j] - Unew[0][j]) * (U[0][j] - Unew[0][j]);
                }
            }
        }

        //TODO: implement case where nrows = 1!!!

        U.swap(Unew);

        //compute errors and check for convergence in each rank

        error *= h;
        error = std::sqrt(error);

        local_conv = error <= tol;

        MPI_Barrier(MPI_COMM_WORLD);

        MPI_Allreduce(&local_conv,&global_conv,1,MPI_CXX_BOOL,MPI_LAND,MPI_COMM_WORLD); //check for global convergence

        #ifdef DEBUG
            std::cout << "error for rank " << rank << " is " << error << "," << "local convergence is " << local_conv << std::endl;
            if(rank == 0){
                std::cout << "global convergence is " << global_conv << std::endl;
                if(global_conv) std::cout << "iterations: "<< k << std::endl;
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

    //gather the results in vector of vectors rank zero to write the VTK file

    /*

    //first, turn the map into a vector of vectors

    std::vector<std::vector<double>> U_vect;

    int i = 0;

    if(rank == 0){ 
        U_vect.resize(npoints);
        for(auto vec : U_vect){
            vec.resize(npoints);
            i+=vec.size();
            #ifdef DEBUG
                std::cout << "element "<<i<<"is here"<<std::endl;
            #endif
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    #ifdef DEBUG
        if(rank == 0){
            std::cout << "started gathering" << std::endl;
        }   
    #endif

    for(int i = 0; i<size;++i){
        for(int j = 0; j<nrows;++j){
            if(rank == i){
                MPI_Send(U[j].data(),ncols,MPI_DOUBLE,0,i*nrows+j,MPI_COMM_WORLD);
                #ifdef DEBUG
                    if(rank == 0) std::cout << "sending row " << i*nrows + j << std::endl;
                #endif
            }
            MPI_Barrier(MPI_COMM_WORLD);
            if(rank == 0){
                MPI_Recv(U[i*nrows+j].data(),ncols,MPI_DOUBLE,i,i*nrows+j,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
                #ifdef DEBUG
                    if(rank == 0) std::cout << "receiving row " << i*nrows + j << std::endl;
                #endif
            }
            MPI_Barrier(MPI_COMM_WORLD);

        }
    }

    #ifdef DEBUG
        if(rank == 0){
            std::cout << "finished gathering" << std::endl;
            for(auto vec: U_vect){
                for(auto )
            }
        }
    #endif

    */

   //populate VTK file in order of rank

   for(unsigned i = 0; i<size;++i){
        if(rank == i){
            #ifdef DEBUG
                std::cout << "rank "<<i<<"is populating the vtk fike"<<std::endl;
            #endif
            generateVTKFile("results.vtk",U,ncols,nrows,h);
        }
        MPI_Barrier(MPI_COMM_WORLD);
   }



    return global_conv;

}
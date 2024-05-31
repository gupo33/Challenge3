#include <array>
#include <iostream>
#include <fstream>
#include <mpi.h>
#include "../include/nlohmann/json.hpp"
#include "../include/parser/mpParser.h"
#include "../include/laplaceSolver.hpp"

using json = nlohmann::json;


int main(int argc, char* argv[]){

    MPI_Init(&argc, &argv);

    int rank,size;

    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size (MPI_COMM_WORLD,&size);

    //initialize variables for parsed items for all ranks

    unsigned max_it;
    double a,b;
    double tol;
    unsigned npoints;
    double bc;

    std::string fun;
    std::array<std::string,2> vars{"x","y"};

    unsigned fun_size, vars_size;

    if(rank == 0){ //rank zero parsers everything

        std::ifstream file("data.json");
        json data = json::parse(file);

        //parse method parameters

        max_it = data.value("max_it",1000);
        a = data.value("a",-1.);
        b = data.value("b",1.);
        tol = data.value("tol",0.001);
        npoints = data.value("npoints",10);
        bc = data.value("bc",0);

        //parse function and variables

        fun = data.value("fun","0.");
        fun_size = fun.size();
    }

    //broadcast everything for all ranks

    MPI_Bcast(&max_it,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(&a,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
    MPI_Bcast(&b,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
    MPI_Bcast(&tol,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
    MPI_Bcast(&npoints,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(&bc,1,MPI_DOUBLE,0,MPI_COMM_WORLD);

    //first, I need to resize fun and vars

    MPI_Bcast(&fun_size,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(&vars_size,1,MPI_INT,0,MPI_COMM_WORLD);

    if(rank!=0){
        fun.resize(fun_size);
    }


    MPI_Bcast(fun.data(),fun.size(),MPI_CHAR,0,MPI_COMM_WORLD);

    #ifdef DEBUG
        std::cout << rank <<": "<<a<<" "<<b<<" "<<tol<<" "<<npoints<<" "<<bc<<" "<<fun<<" "<<vars[0]<<vars[1]<<std::endl;
    #endif

    //every rank now creaters a muParser function and calls the method with the right parameters

    mup::ParserX p_fun; //initialize parser for function

    //initialize values and variables

    std::vector<mup::Value> val_vec{0.,0.};
    std::vector<mup::Variable> var_vec{&val_vec[0],&val_vec[1]};

    //define function variables

    p_fun.DefineVar(vars[0],&var_vec[0]);
    p_fun.DefineVar(vars[1],&var_vec[1]);

    //set function expression

    p_fun.SetExpr(fun);

    //define function wrapper to input into the method

    auto muFun = [&val_vec,&p_fun](std::vector<double> x){
        val_vec[0] = x[0];
        val_vec[1] = x[1];
        return static_cast<double>(p_fun.Eval().GetFloat());
    };

    //start timer

    static std::chrono::_V2::system_clock::time_point t_start, t_end;

    if(rank == 0) t_start = std::chrono::high_resolution_clock::now();

    //call method

    laplaceSolver(max_it,a,b,tol,npoints,bc,muFun);

    MPI_Barrier(MPI_COMM_WORLD);

    if(rank == 0) t_end = std::chrono::high_resolution_clock::now();

    //stop timer

    if(rank == 0) std::cout << "Elapsed time is " << (t_end-t_start).count()*1E-9 << " seconds\n";

    MPI_Finalize();
}
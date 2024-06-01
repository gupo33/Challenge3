#ifndef WRITEVTK_HPP
#define WRITEVTK_HPP

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <map>
#include <mpi.h>

static bool firstAccess = true;

// generates a STRUCTURES VTK file with a scalar field
void generateVTKFile(const std::string & filename, 
                     const std::map<unsigned, std::vector<double>> & scalarField, 
                     int num_cols, int num_rows, double h) {

    int rank,size;

    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size (MPI_COMM_WORLD,&size);

    std::ofstream vtkFile;

    if(rank == 0 && firstAccess){ //when first accessing the file, overwrite it
        vtkFile.open(filename);
    }
    else{ //when accessing it later, append
        vtkFile.open(filename,std::ios::app);
    }

    // check if the file was opened
    if (!vtkFile.is_open()) {
        std::cerr << "Error: could not open file " << filename << std::endl;
        return;
    }

    //first rank should write the header when it first writes to the file

    if(rank == 0 && firstAccess){

        firstAccess = false;

        // Write VTK header
        vtkFile <<  "# vtk DataFile Version 3.0\n";
        vtkFile << "Scalar Field Data\n";
        vtkFile << "ASCII\n";                                // file format
        

        // Write grid data
        vtkFile << "DATASET STRUCTURED_POINTS\n";                             // format of the dataset
        vtkFile << "DIMENSIONS " << num_cols << " " << num_cols << " " << 1 << "\n";  // number of points in each direction
        vtkFile << "ORIGIN 0 0 0\n";                                          // lower-left corner of the structured grid
        vtkFile << "SPACING" << " " << h << " " << h << " " << 1 << "\n";   // spacing between points in each direction
        vtkFile << "POINT_DATA " << (num_cols) * (num_cols) << "\n";                  // number of points
                                                                    
        
        // Write scalar field data
        vtkFile << "SCALARS scalars double\n";               // description of the scalar field
        vtkFile << "LOOKUP_TABLE default\n";                 // color table
    
    }

    // Write vector field data
    for (int i = 0; i < num_rows; ++i) {
        for(int j = 0; j<num_cols; ++j){
            vtkFile << scalarField.at(i)[j] << " ";
        }
        vtkFile << "\n";
    }
}

#endif // WRITEVTK_HPP

# DD2360HT22_PROJECT

### SputniPIC - Modified ###
Welcome to SputniPIC, a tool for simulating the motion of charged particles in a plasma. This project is a modified variant from
the original repository at: https://github.com/KTH-HPC/sputniPIC-DD2360. This modified repository experiments with have smaller
data types to run the simulation in the mover function of the code.

To build and run the program, follow these instructions:

### Building the program ###
- From the root directory of the project, run make to compile the source code.
- The compiled executable will be placed in the bin directory, named sputniPIC.out.

### Running the program ###
Run the program with the following command: ./bin/sputniPIC.out inputfiles/GEM_2D.inp
To run the program with single-precision floats, execute: ./bin/sputniPIC.out.float inputfiles/GEM_2D.inp

### Input files ###
The program requires an input file to specify the parameters of the simulation. The input file should be placed in the inputfiles directory. The provided GEM_2D.inp file is an example of the format of the input file.

### Output files ###
The program will generate output files in the outputfiles directory, containing the results of the simulation. The format and content of these files will depend on the parameters specified in the input file.
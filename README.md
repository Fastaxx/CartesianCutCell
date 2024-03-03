# Cartesian Cut Cell Method 

Cartesian Cut Cell Method repository. This repository contains code and resources related to the Cartesian cut cell method

## Overview

The Cartesian cut cell method is a powerful numerical technique for solving partial differential equations (PDEs) on irregular domains or complex geometries. It combines the simplicity and efficiency of structured Cartesian grids with the flexibility to handle arbitrary geometries using cut cells.

This repository aims to provide:

- Implementation of the Cartesian cut cell method in [Julia].
- Examples and tutorials demonstrating the application of the method to various PDEs and geometries.
- Documentation and resources for understanding the theory and implementation details of the Cartesian cut cell method.

## Contents

1. **Implementation**: `./src` : Contains the code implementing the Cartesian cut cell method. This includes the core algorithm as well as any utilities or helper functions.

2. **Examples**: `./test` : Provides examples demonstrating how to use the Cartesian cut cell method

3. **Documentation**: `.\docs` : Contains documentation and implementation details

4. **References**: 

## Requirements

- [Julia] [1.10]

## DONE

- Implement Operators, Matrix : Cut Cell machinery - OK
- Test Operators - WIP :
	- Gradient : OK
	- Divergence : WIP
- Poisson Disk Inside : 
	- Implement OK 
	- Test Ok 
	- Comp WIP

## TO DO

1. Poisson Disk Outside : WIP : Deal Interface Boundary condition and Border Condition
2. Test Poisson 3D - Fix CartesianGeometry `axs`
3. Stefan Problem : Cut Cell operators formulation + Implementation



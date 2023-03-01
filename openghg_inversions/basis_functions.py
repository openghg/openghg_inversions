# *****************************************************************************
# Created: 7 Nov. 2022
# Author: Eric Saboya, School of Geographical Sciences, University of Bristol
# Contact: eric.saboya@bristol.ac.uk
# *****************************************************************************
# About
#   Basis functions for used for HBMCMC inversions. Originally created by
#   Anita Ganesan and updated, here, by Eric Saboya.
#   HBMCMC uses the QuadTree algorithm for creating basis functions for
#   the inversion runs.
# *****************************************************************************

import os
import glob
import uuid
import getpass
import scipy.optimize
import numpy as np
import xarray as xr
import pandas as pd

 
class quadTreeNode:    
    
    def __init__(self, xStart, xEnd, yStart, yEnd):
        self.xStart = xStart
        self.xEnd = xEnd
        self.yStart = yStart
        self.yEnd = yEnd
        
        self.child1 = None #top left
        self.child2 = None #top right
        self.child3 = None #bottom left
        self.child4 = None #bottom right
    
    def isLeaf(self):
        if self.child1 or self.child2 or self.child3 or self.child4:
            return False
        else:
            return True
        
    def createChildren(self, grid, limit):
        value = np.sum(grid[self.xStart:self.xEnd, self.yStart:self.yEnd])#.values

        #stop subdividing if finest resolution or bucket level reached
        if (value < limit or
            (self.xEnd-self.xStart < 2) or (self.yEnd-self.yStart <2)):
            return

        dx = (self.xEnd-self.xStart)
        dy = (self.yEnd-self.yStart)

        #create 4 children for subdivison
        self.child1 = quadTreeNode(self.xStart, self.xStart + dx//2, self.yStart, self.yStart + dy//2)
        self.child2 = quadTreeNode(self.xStart + dx//2, self.xStart + dx, self.yStart, self.yStart + dy//2)
        self.child3 = quadTreeNode(self.xStart, self.xStart + dx//2, self.yStart + dy//2, self.yStart + dy)
        self.child4 = quadTreeNode(self.xStart + dx//2, self.xStart + dx, self.yStart + dy//2, self.yStart + dy)
        
        #apply recursion on all child nodes
        self.child1.createChildren(grid, limit)
        self.child2.createChildren(grid, limit)
        self.child3.createChildren(grid, limit)
        self.child4.createChildren(grid, limit)
        
    def appendLeaves(self, leafList):
        #recursively append all leaves/end nodes to leafList
        if (self.isLeaf()):
            leafList.append(self)
        else:
            self.child1.appendLeaves(leafList)
            self.child2.appendLeaves(leafList)
            self.child3.appendLeaves(leafList)
            self.child4.appendLeaves(leafList)
           
def quadTreeGrid(grid, limit):
    '''
    Apply quadtree division algorithm
    -----------------------------------
    Args:
      grid (array): 
        2d numpy array to apply quadtree division to
      limit (float): 
        Use value as bucket level for defining maximum subdivision

    Returns:
      outputGrid (array): 
        2d numpy grid, same shape as grid, with values correpsonding to 
        each  box from boxList
      boxList: (list of lists) 
        Each sublist describes the corners of a quadtree leaf
    -----------------------------------
    '''
    #start with a single node the size of the entire input grid:
    parentNode = quadTreeNode(0, grid.shape[0], 0, grid.shape[1])
    parentNode.createChildren(grid, limit)

    leafList = []
    boxList = []
    parentNode.appendLeaves(leafList)
    
    outputGrid = np.zeros_like(grid)

    for i, leaf in enumerate(leafList):
        outputGrid[leaf.xStart:leaf.xEnd, leaf.yStart:leaf.yEnd] = i
        boxList.append([leaf.xStart, leaf.xEnd, leaf.yStart, leaf.yEnd])
    
    return outputGrid, boxList

def quadtreebasisfunction(emissions_name, fp_all, sites, 
                          start_date, domain, species, outputname, outputdir=None,
                          nbasis=100, abs_flux=False):
    '''
    Creates a basis function with nbasis grid cells using a quadtree algorithm.
    The domain is split with smaller grid cells for regions which contribute
    more to the a priori (above basline) mole fraction. This is based on the
    average footprint over the inversion period and the a priori emissions field.
    Output is a netcdf file saved to /Temp/<domain> in the current directory 
    if no outputdir is specified or to outputdir if specified.
    The number of basis functions is optimised using dual annealing. Probably
    not the best or fastest method as there should only be one minima, but doesn't
    require the Jacobian or Hessian for optimisation.
    -----------------------------------
    Args:
      emissions_name (list): 
        List of "source" key words as used for retrieving specific emissions
        from the object store.      
      fp_all (dict): 
        Output from footprints_data_merge() function. Dictionary of datasets.
      sites (list): 
        List of site names (This could probably be found elsewhere)
      start_date (str): 
        String of start date of inversion
      domain (str): 
        The inversion domain
      species (str): 
        Atmospheric trace gas species of interest (e.g. 'co2')
      outputname (str): 
        Identifier or run name
      outputdir (str, optional): 
        Path to output directory where the basis function file will be saved.
        Basis function will automatically be saved in outputdir/DOMAIN
        Default of None makes a temp directory.
      nbasis (int): 
        Number of basis functions that you want. This will optimise to 
        closest value that fits with quadtree splitting algorithm, 
        i.e. nbasis % 4 = 1.
      abs_flux (bool):
        If True this will take the absolute value of the flux
    
    Returns:
        If outputdir is None, then returns a Temp directory. The new basis function is saved in this Temp directory.
        If outputdir is not None, then does not return anything but saves the basis function in outputdir.
    -----------------------------------
    '''
    if abs_flux:
        print('Using absolute values of flux array')
    if emissions_name == None:
        flux = np.absolute(fp_all['.flux']['all'].data.flux.values) if abs_flux else fp_all['.flux']['all'].data.flux.values 
        meanflux = np.squeeze(flux)
    else:
        if isinstance(fp_all[".flux"][emissions_name[0]], dict):
            arr = fp_all[".flux"][emissions_name[0]]
            flux = np.absolute(arr.data.flux.values) if abs_flux else arr.data.flux.values
            meanflux = np.squeeze(flux)
        else:
            flux = np.absolute(fp_all[".flux"][emissions_name[0]].data.flux.values) if abs_flux else \
                   fp_all[".flux"][emissions_name[0]].data.flux.values 
            meanflux = np.squeeze(flux)
    meanfp = np.zeros((fp_all[sites[0]].fp.shape[0],fp_all[sites[0]].fp.shape[1]))
    div=0
    for site in sites:
        meanfp += np.sum(fp_all[site].fp.values,axis=2)
        div += fp_all[site].fp.shape[2]
    meanfp /= div
    
    if meanflux.shape != meanfp.shape:
        meanflux = np.mean(meanflux, axis=2)
    fps = meanfp*meanflux

    def qtoptim(x):
        basisQuad, boxes = quadTreeGrid(fps, x)
        return (nbasis - np.max(basisQuad)-1)**2

    cost = 1e6
    pwr = 0
    while cost > 3.:
        optim = scipy.optimize.dual_annealing(qtoptim, np.expand_dims([0,100/10**pwr], axis=0))
        cost = np.sqrt(optim.fun)
        pwr += 1
        if pwr > 10:
            raise Exception("Quadtree did not converge after max iterations.")
    basisQuad, boxes = quadTreeGrid(fps, optim.x[0])
    
    lon = fp_all[sites[0]].lon.values
    lat = fp_all[sites[0]].lat.values    
    
    base = np.expand_dims(basisQuad+1,axis=2)
    
    time = [pd.to_datetime(start_date)]
    newds = xr.Dataset({'basis' : ([ 'lat','lon', 'time'], base)}, 
                        coords={'time':(['time'], time), 
                    'lat' : (['lat'],  lat), 'lon' : (['lon'],  lon)})    
    newds.lat.attrs['long_name'] = 'latitude' 
    newds.lon.attrs['long_name'] = 'longitude' 
    newds.lat.attrs['units'] = 'degrees_north'
    newds.lon.attrs['units'] = 'degrees_east'     
    newds.attrs['creator'] = getpass.getuser()
    newds.attrs['date created'] = str(pd.Timestamp.today())
    
    if outputdir is None:
        cwd = os.getcwd()
        tempdir = os.path.join(cwd,f"Temp_{str(uuid.uuid4())}")
        os.mkdir(tempdir)    
        os.mkdir(os.path.join(tempdir,f"{domain}/"))
        newds.to_netcdf(os.path.join(tempdir,domain,f"quadtree_{species}-{outputname}_{domain}_{start_date.split('-')[0]}.nc"), mode='w')
        return tempdir
    else:
        basisoutpath = os.path.join(outputdir,domain)
        if not os.path.exists(basisoutpath):
            os.makedirs(basisoutpath)
        newds.to_netcdf(os.path.join(basisoutpath,f"quadtree_{species}-{outputname}_{domain}_{start_date.split('-')[0]}.nc"), mode='w')

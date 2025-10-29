"""Module to create basis regions for the inversion using a quadtree algorithm."""

import numpy as np
import scipy.optimize


class quadTreeNode:
    """Node of the quadtree algorithm.

    Class attributes 'xStart', 'xEnd', 'yStart', 'yEnd' store the
    coordinates (in terms of indexes of the grid) of the node;
    'child1', 'child2', 'child3', 'child4' store its children
    nodes (e.g. its subdivisions).
    """

    def __init__(self, xStart: int, xEnd: int, yStart: int, yEnd: int):
        """Init quadTreeNode.

        Args :
            xStart:
                index of the grid on the first axis
                on which the node starts.
            xEnd:
                index of the grid on the first axis
                on which the node ends.
            yStart:
                index of the grid on the second axis
                on which the node starts.
            yEnd:
                index of the grid on the second axis
                on which the node ends.
        """
        self.xStart = xStart
        self.xEnd = xEnd
        self.yStart = yStart
        self.yEnd = yEnd

        self.child1 = None  # top left
        self.child2 = None  # top right
        self.child3 = None  # bottom left
        self.child4 = None  # bottom right

    def isLeaf(self):
        """Return True if node is a leaf (i.e. don't have children), False if not."""
        return not (self.child1 or self.child2 or self.child3 or self.child4)

    def createChildren(self, grid: np.ndarray, limit: float):
        """Create children nodes. If finest resolution or bucket level reached,
        no children nodes are created and the node is thus a leaf.

        Args :
            grid:
                2d numpy array to wich the quadtree division is applied.
            limit:
                Bucket level (i.e. targeted resolution which is compared to the
                sum of the grid points in the node).
        """
        value = np.sum(grid[self.xStart : self.xEnd, self.yStart : self.yEnd])  # .values

        # stop subdividing if finest resolution or bucket level reached
        if value < limit or (self.xEnd - self.xStart < 2) or (self.yEnd - self.yStart < 2):
            return

        dx = self.xEnd - self.xStart
        dy = self.yEnd - self.yStart

        # create 4 children for subdivison
        self.child1 = quadTreeNode(self.xStart, self.xStart + dx // 2, self.yStart, self.yStart + dy // 2)
        self.child2 = quadTreeNode(
            self.xStart + dx // 2, self.xStart + dx, self.yStart, self.yStart + dy // 2
        )
        self.child3 = quadTreeNode(
            self.xStart, self.xStart + dx // 2, self.yStart + dy // 2, self.yStart + dy
        )
        self.child4 = quadTreeNode(
            self.xStart + dx // 2, self.xStart + dx, self.yStart + dy // 2, self.yStart + dy
        )

        # apply recursion on all child nodes
        self.child1.createChildren(grid, limit)
        self.child2.createChildren(grid, limit)
        self.child3.createChildren(grid, limit)
        self.child4.createChildren(grid, limit)

    def appendLeaves(self, leafList: list):
        """Recursively look for leaves in the node offsprings and append them to the leafList.

        Args :
          leafList:
            list containing all the leaves, i.e. basis regions that will be used
            in the hbmcmc inversion.
        """
        # recursively append all leaves/end nodes to leafList
        if self.isLeaf():
            leafList.append(self)
        else:
            self.child1.appendLeaves(leafList)
            self.child2.appendLeaves(leafList)
            self.child3.appendLeaves(leafList)
            self.child4.appendLeaves(leafList)


def quadTreeGrid(grid: np.ndarray, limit: float) -> np.ndarray:
    """Apply quadtree division algorithm.

    Args:
      grid (array):
        2d numpy array to apply quadtree division to
      limit (float):
        Use value as bucket level for defining maximum subdivision

    Returns:
      outputGrid (array):
        2d numpy grid, same shape as grid, with values correpsonding to
        each box from boxList
    """
    # start with a single node the size of the entire input grid:
    parentNode = quadTreeNode(0, grid.shape[0], 0, grid.shape[1])
    parentNode.createChildren(grid, limit)

    leafList = []
    parentNode.appendLeaves(leafList)

    outputGrid = np.zeros_like(grid)

    for i, leaf in enumerate(leafList):
        outputGrid[leaf.xStart : leaf.xEnd, leaf.yStart : leaf.yEnd] = i

    return outputGrid


def get_quadtree_basis(fps: np.ndarray, nbasis: int, seed: int | None = None) -> np.ndarray:
    """Given an array and a specified number of basis functions, return basis regions specified by
    the quadtree algorithm.

    Args:
        fps: array (mean flux times mean footprints) to use to calculate basis regions
        nbasis: target number of basis regions
        seed: optional random seed to use (for testing or reproducing results)

    Returns:
        2D numpy array with positive integer values representing basis regions.
    """

    def qtoptim(x):
        basisQuad = quadTreeGrid(fps, x)
        return (nbasis - np.max(basisQuad) - 1) ** 2

    # Calculate search range based on data characteristics
    non_zero_vals = fps[fps > 0]
    if len(non_zero_vals) == 0:
        raise ValueError("Input array contains no non-zero values")
    
    # For quadtree, the effective limit values are typically larger than individual data values
    # They represent aggregate sums over regions. Use percentiles to estimate good ranges.
    percentiles = np.percentile(non_zero_vals, [95, 99.9])
    max_val = np.max(non_zero_vals)
    
    # Set search range to cover the spectrum from high percentiles to well above max value
    # This captures the range where meaningful subdivision control occurs
    search_min = percentiles[0]  # 95th percentile
    search_max = max_val * 1000  # Well above maximum value
    
    best_result = None
    best_cost = float('inf')
    best_x = None
    
    max_iterations = 25
    tolerance = max(3.0, np.sqrt(nbasis) * 0.5)  # Scale tolerance with target size
    
    for pwr in range(max_iterations):
        # Use both exponential and logarithmic spacing to cover the range effectively
        if pwr < 15:
            # Exponential scaling - divide search range
            current_max = search_max / (1.5**pwr)
            current_min = search_min / (1.5**pwr)
        else:
            # Switch to logarithmic spacing for finer control
            log_min = np.log10(search_min)
            log_max = np.log10(search_max)
            log_current = log_min + (log_max - log_min) * (pwr - 15) / 10
            current_min = 10**log_current
            current_max = current_min * 10
        
        # Ensure minimum bound
        current_min = max(current_min, search_min / 1000)
        
        if current_min >= current_max or current_max <= 0:
            continue
        
        try:
            optim = scipy.optimize.dual_annealing(
                qtoptim, np.expand_dims([current_min, current_max], axis=0), 
                seed=seed, maxiter=1000
            )
            cost = np.sqrt(optim.fun)
            
            # Keep track of the best solution found so far
            if cost < best_cost:
                best_cost = cost
                best_result = quadTreeGrid(fps, optim.x[0])
                best_x = optim.x[0]
                
            # Early termination if we found a good solution
            if cost <= tolerance:
                break
                
        except Exception:
            # If optimization fails, continue with next iteration
            continue
    
    # Accept solution if it's reasonably close to target
    if best_result is not None:
        actual_nbasis = np.max(best_result) + 1
        relative_error = abs(actual_nbasis - nbasis) / nbasis
        
        # Accept solution if it's within 20% of target or within 5 basis functions
        if relative_error < 0.2 or abs(actual_nbasis - nbasis) <= 5:
            return best_result + 1
    
    # If no acceptable solution found, raise an informative error
    if best_result is not None:
        actual_nbasis = np.max(best_result) + 1
        raise RuntimeError(
            f"Quadtree could not find a solution close enough to target. "
            f"Target: {nbasis}, closest found: {actual_nbasis}, "
            f"relative error: {abs(actual_nbasis - nbasis) / nbasis:.1%}. "
            f"Try a different target number of basis functions."
        )
    else:
        raise RuntimeError(
            f"Quadtree optimization failed completely. "
            f"The data may be too sparse or the target number of basis functions ({nbasis}) "
            f"may be incompatible with the data structure."
        )

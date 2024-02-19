from typing import Optional

import numpy as np
import scipy.optimize


class quadTreeNode:
    def __init__(self, xStart, xEnd, yStart, yEnd):
        self.xStart = xStart
        self.xEnd = xEnd
        self.yStart = yStart
        self.yEnd = yEnd

        self.child1 = None  # top left
        self.child2 = None  # top right
        self.child3 = None  # bottom left
        self.child4 = None  # bottom right

    def isLeaf(self):
        if self.child1 or self.child2 or self.child3 or self.child4:
            return False
        else:
            return True

    def createChildren(self, grid, limit):
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

    def appendLeaves(self, leafList):
        # recursively append all leaves/end nodes to leafList
        if self.isLeaf():
            leafList.append(self)
        else:
            self.child1.appendLeaves(leafList)
            self.child2.appendLeaves(leafList)
            self.child3.appendLeaves(leafList)
            self.child4.appendLeaves(leafList)


def quadTreeGrid(grid, limit):
    """
    Apply quadtree division algorithm.

    Args:
      grid (array):
        2d numpy array to apply quadtree division to
      limit (float):
        Use value as bucket level for defining maximum subdivision

    Returns:
      outputGrid (array):
        2d numpy grid, same shape as grid, with values correpsonding to
        each  box from boxList
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


def get_quadtree_basis(fps: np.ndarray, nbasis: int, seed: Optional[int] = None) -> np.ndarray:
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

    cost = 1e6
    pwr = 0
    while cost > 3.0:
        optim = scipy.optimize.dual_annealing(
            qtoptim, np.expand_dims([0, 100 / 10**pwr], axis=0), seed=seed
        )
        cost = np.sqrt(optim.fun)
        pwr += 1
        if pwr > 10:
            raise RuntimeError("Quadtree did not converge after max iterations.")

    return quadTreeGrid(fps, optim.x[0]) + 1

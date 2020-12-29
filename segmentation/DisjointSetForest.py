"""
Copyright December 2020 - Filippo Guerranti <filippo.guerranti@student.unisi.it>

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np
from PIL import Image, ImageFilter


class DisjointSetForest:
    """ Disjoint-set forest data structure (https://en.wikipedia.org/wiki/Disjoint-set_data_structure).
    Stores a collection of disjoint non-overlapping sets (or a partition of a set into disjoint subsets).
    Provides operations for adding new sets, merging sets (replacing them by their union) and find a 
    representative member of a set. 
    """

    # Class members
    parent = None       # list providing the parent of the indexed node (list)
    rank = None         # list providing the rank of the indexed node (list)
    size = None         # list providing the size of the child tree of the indexed node (list)


    def __init__(self, num_nodes: int):
        """ Disjoint-set forest class constructor.

        Args:
            num_nodes (int): total number of elements to be partitioned.
        """
        self.parent = [i for i in range(num_nodes)]
        self.rank = [0 for i in range(num_nodes)]
        self.size = [1 for i in range(num_nodes)]


    
    def size_of(self, u: int):
        """ Return the size of the child tree of the given component 
        
        Args:
            u (int): node of which we want to know the size.
        
        Returns:
            size (int): number of nodes in the subtree of u.    
        """
        return self.size[u]



    def find(self, u: int):
        """ Return the representative of a subset.

        Args:
            u (int): node of which we want to know the representative.

        Returns:
            representative of the subset/node.
        """
        if self.parent[u] == u:
            return u

        self.parent[u] = self.find(self.parent[u])
        return self.parent[u]



    def merge(self, u: int, v: int):
        """ Merge two nodes/subsets into one based on their rank.

        Args:
            u (int): first node/subset
            v (int): second node/subset
        """
        u = self.find(u)
        v = self.find(v)

        if u != v:
            if self.rank[u] > self.rank[v]:
                u, v = v, u

            self.parent[u] = v
            self.size[v] += self.size[u]
            if self.rank[u] == self.rank[v]:
                self.rank[v] += 1



    def num_components(self):
        """ Return the number of current components.
        
        Returns:
            num_components (int): number of current components
        """
        return len(self.parents())



    def parents(self):
        """ Return the parent nodes.

        Returns:
            parents (list): list of parent nodes
        """
        return list(set(self.parent))


    def sorted_parents(self):
        """ Return the parents nodes sorted by size in descreasing order.

        Returns:
            sorted_parents (list): list of sorted parent nodes
        """
        return sorted(self.parents, key=lambda parent: self.size_of(parent), reverse=True)
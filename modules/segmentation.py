""" modules.segmentation.py
Summary
-------
This module contains the classes which are necessary for the graph-based image segmentation algorithm  
proposed by Felzenszwalb et. al. ([paper](http://cs.brown.edu/people/pfelzens/papers/seg-ijcv.pdf)).  

Classes
-------
DisjointSetForest
    implements the base data structure for the graph-based segmentation algorithm

GraphBasedSegmentation
    implements the graph-based segmentation algorithm


Copyright 2021 - Filippo Guerranti <filippo.guerranti@student.unisi.it>

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
import torch.tensor
from math import ceil
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw


class DisjointSetForest:
    """ Stores a collection of disjoint non-overlapping subset of an input set.  

    Provides operations for:
    * add new sets
    * merge sets (replacing them by their union set)
    * find a set representative

    References
    ----------
    Disjoint-set forest data structure ([link](https://en.wikipedia.org/wiki/Disjoint-set_data_structure)).
    """

    def __init__(
        self, 
        num_nodes: int
        ) -> None:
        """ Disjoint-set forest class constructor.  

        Initializes the set (parent, rank and size lists).  
        Each node is initially parent of itself.  
        Each node has initially rank equal to zero.  
        Each node is initially alone (subset size equal to one)

        Parameters
        ----------
        num_nodes: int 
            total number of elements to be partitioned
        """
        
        self.parent = [i for i in range(num_nodes)]
        self.rank = [0 for i in range(num_nodes)]
        self.size = [1 for i in range(num_nodes)]


    
    def size_of(
        self, 
        u: int
        ) -> int:
        """ Returns the number of nodes which have `u` as parent.
        
        Parameters
        ----------
        u: int
            node of which we want to know the size
        
        Returns
        -------
        size: int
            number of nodes in the subtree of u (nodes which have u as parent)
        """
        return self.size[u]



    def find(
        self, 
        u: int
        ) -> int:
        """ Returns the representative of a node.

        Given a subset of nodes belonging they all have the same representative.

        Parameters
        ----------
        u: int
            node of which we want to know the representative

        Returns
        -------
        self.parent[u]: int
            index of the representative node of node `u`.

        Notes
        -----
        If a node is parent of itself, it means that it is the representative.  
        Otherwise the `find()` method is applied recursively until the parent of the other nodes is found. 
        """
        if self.parent[u] == u:
            return u

        self.parent[u] = self.find(self.parent[u])
        return self.parent[u]



    def merge(
        self, 
        u: int, 
        v: int
        ) -> None:
        """ Given two nodes, merges the subset they belongs to into one subset.

        Parameters
        ----------
        u: int
            first node
        v: int
            second node

        Notes
        -----
        The representatives of the input nodes are found and, if different, the subsets of the two nodes are merged.  
        The subset whose representative has the higher rank becomes the parent of the other representative.
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



    def num_components(self) -> int:
        """ Returns the number of current subsets.
        
        Returns
        -------
        num_components: int
            number of current subsets
        """
        return len(self.parents())



    def parents(self) -> list:
        """ Returns the parent nodes (a.k.a. representatives)

        If a node is not parent of itself, it is not in this list.

           e       f     
         /  \    /  \   
        a   b   c   d 

        parents = [e, f]

        Returns
        -------
        parents: list)
            list of parent nodes
        """
        return list(set(self.parent))



    def sorted_parents(self) -> list:
        """ Returns the parents nodes in decreasing order of child size.

           f       g     
         /  \    / |  \   
        a   b   c  d  e 

        sorted_parents = [g, f]

        Returns
        -------
        sorted_parents: list
            list of sorted parent nodes
        """
        return sorted(self.parents(), key=lambda parent: self.size_of(parent), reverse=True)



class GraphBasedSegmentation:
    """ Implementats the graph-based segmentation algorithm.

    A graph is built starting from an input image.  
    Each pixel is a vertex of G = (V, E) where:
    * G is the graph
    * V is the set of vertices
    * E is the set of edges
    
    Each pixel is connected to its neighbors in an 8-grid sense.
    The algorithm is applied exploiting the disjoint-set forest structure.

    References
    ----------
    Graph-based segmentation algorithm ([paper](http://cs.brown.edu/people/pfelzens/papers/seg-ijcv.pdf))
    """

    def __init__(
        self, 
        img: str or np.ndarray
        ) -> None:
        """ GraphBasedSegmentation class constructor.

        Parameters
        ----------
        img: str or np.ndarray
            path to the input image (to be preprocessed)
            np.ndarray of the input image (already preprocessed)

        Raises
        ------
        
        """
        if isinstance(img, str):
            self.img = Image.open(img)
        elif isinstance(img, np.ndarray):
            self.img = Image.fromarray(img)
        else:
            raise TypeError("input image must be either str or np.ndarray.")

        self.img_width, self.img_height = self.img.size
    


    @staticmethod
    def __preprocessing(
        img: Image, 
        contrast: float=1.5, 
        gaussian_blur: float=2.3, 
        width: int=300, 
        height: int=None
        ) -> np.ndarray:
        """ Applies preprocessing operations to the input image.
        
        Conversion to grayscale.  
        Gaussian blur filter.  
        Contrast enhancement.  
        Resize.

        Parameters
        ----------
        img: PIL.Image
            input image to be preprocessed

        constrast: float (default=1.5) 
            contrast factor (if constrast==1, the output is the original image)

        gaussian_blur: float (default=2.3) 
            Gaussian Blur filter radius

        width: int (default=300) 
            new image width

        height: int (default=None) 
            new image height (if None it is resize in order to maintain the original ratio)
        
        Returns
        -------
        img: np.ndarray
            Numpy array representing the preprocessed image
        """
        
        img = img.convert("L")

        img = img.filter(ImageFilter.GaussianBlur(gaussian_blur))

        img = ImageEnhance.Contrast(img).enhance(contrast)

        if height is None:
            percentage = float(width / img.size[0])
            height = int((float(img.size[1]) * float(percentage)))
        img = img.resize((width, height), Image.ANTIALIAS)
        
        return np.array(img)



    @staticmethod
    def __get_diff(
        img: np.ndarray, 
        u_coords: tuple, 
        v_coords: tuple
        ) -> np.uint8:
        """ Returns the difference in terms of intensity between two pixels of an image.
        
        Parameters
        ----------
        img: numpy.ndarray
            input grayscale image in array format (pixels values from 0 to 255)

        u_coords: tuple
            coordinates of first pixel (x1, y1)

        v_coords: tuple)
            coordinates of second pixel (x2, y2)

        Returns
        -------
        diff: numpy.uint8
            difference between the input pixels
        """
        x1, y1 = u_coords
        x2, y2 = v_coords
        
        # necessary check since pixels are uint8 (cannot use abs)
        if img[y1, x1] > img[y2, x2]:
            return img[y1, x1] - img[y2, x2]
        
        return img[y2, x2] - img[y1, x1]



    @staticmethod
    def __create_edge(
        img: np.ndarray, 
        u_coords: tuple, 
        v_coords: tuple
        ) -> (int, int, np.uint8):
        """ Creates the edge between two pixels of the input image.

        Parameters
-       ----------
        img: numpy.ndarray
            input grayscale image in array format (pixels values from 0 to 255)

        u_coords: tuple
            coordinates of first pixel (x1, y1)

        v_coords: tuple)
            coordinates of second pixel (x2, y2)

        Returns
        -------
        id1: int
            first pixel id

        id2: int
            second pixel id

        weight: numpy.uint8
            edge weight (difference between pixels values)
        """
        _, width = img.shape

        vertex_id = lambda coords: coords[1] * width + coords[0]
        id1 = vertex_id(u_coords)
        id2 = vertex_id(v_coords)

        weight = GraphBasedSegmentation.__get_diff(img, u_coords, v_coords)
        
        return (id1, id2, weight)

    

    @staticmethod
    def __threshold(
        k: int, 
        size: int
        ) -> int:
        """ Defines the threshold for a subset of cardinality equal to size.

        Parameters
        ----------
        k: int
            scale of observation (large k -> larger components)

        size: int
            cardinality of the component to take into consideration
        
        Returns
        -------
        threshold: int
            threshold for a given component of cardinality equal to size

        Notes
        -----
        In the computation of the minimum internal difference, a threshold function is used and it is defined as here.  
        See the [paper](http://cs.brown.edu/people/pfelzens/papers/seg-ijcv.pdf) for more informations.
        """
        return int(k/size)



    def __build_graph(self) -> None:
        """ Builds the graph, connecting the pixel and assigning the appropriate weights to the connections.
        
        The graph is a list of tuples of the form (u, v, weight) where u and v are two vertices and weight the connection weight.
        """
        
        self.graph = []

        for y in range(self.height):
            for x in range(self.width):

                # all columns except last one -> east connection 
                if x < self.width - 1:
                    u_coords = (x, y)
                    v_coords = (x + 1, y)
                    self.graph.append(GraphBasedSegmentation.__create_edge(self.preprocessed_arr, u_coords, v_coords))

                # all rows except last one -> south connection
                if y < self.height - 1:
                    u_coords = (x, y)
                    v_coords = (x, y + 1)
                    self.graph.append(GraphBasedSegmentation.__create_edge(self.preprocessed_arr, u_coords, v_coords))

                # all columns and rows except last ones -> south-east connection
                if x < self.width - 1 and y < self.height - 1:
                    u_coords = (x, y)
                    v_coords = (x + 1, y + 1)
                    self.graph.append(GraphBasedSegmentation.__create_edge(self.preprocessed_arr, u_coords, v_coords))

                # all columns except last one, all rows except first -> north-est connection
                if x < self.width - 1 and y > 0:
                    u_coords = (x, y)
                    v_coords = (x + 1, y - 1)
                    self.graph.append(GraphBasedSegmentation.__create_edge(self.preprocessed_arr, u_coords, v_coords))
    


    def __sort(self) -> None:
        """ Sorts the graph by non-decreasing order of edges' weight. """

        self.sorted_graph = sorted(self.graph, key=lambda edge: edge[2])

    

    def segment(
        self, 
        k: int=4000, 
        min_size: int=100, 
        preprocessing: bool=True, 
        **kwargs
        ) -> None:
        """ Segments the graph according to the graph-based segmentation algorithm proposed by Felzenszwalb et. al.

        Parameters
        ----------
        k: int (default=4000) 
            parameter for the threshold

        min_size: int (default=100) 
            subsets having size less than min_size are removed (if specified)
            removal not applied (if None)

        preprocessing: bool (default=True) 
            applies preprocessing operations to the image (calls `__preprocessing()`)

        References
        ----------
        [Paper](http://cs.brown.edu/people/pfelzens/papers/seg-ijcv.pdf)
        """

        #region - initialization and preparation
        if preprocessing:
            self.preprocessed_arr = GraphBasedSegmentation.__preprocessing(self.img, **kwargs)
        else:
            self.preprocessed_arr = self.img
            self.preprocessed_arr = self.preprocessed_arr.convert("L")
            self.preprocessed_arr = np.array(self.preprocessed_arr)

        self.height, self.width = self.preprocessed_arr.shape
        self.num_nodes = self.height * self.width
        
        self.components = DisjointSetForest(self.num_nodes)
        
        threshold = [GraphBasedSegmentation.__threshold(k, i) for i in self.components.size]
        
        self.__build_graph()
        self.__sort()
        #endregion

        #region - segmentation algorithm
        for edge in self.sorted_graph:
            u, v, w = edge

            u = self.components.find(u)
            v = self.components.find(v)

            if u != v:
                # boundary evidence check
                if w <= threshold[u] and w <= threshold[v]:
                    self.components.merge(u, v)
                    parent = self.components.find(u)
                    threshold[parent] = w + GraphBasedSegmentation.__threshold(k, self.components.size_of(u))

        # remove components having size less than min_size
        if min_size is not None:

            for edge in self.sorted_graph:
                u, v, _ = edge
                u = self.components.find(u)
                v = self.components.find(v)

                if u != v:
                    if self.components.size_of(u) < min_size or self.components.size_of(v) < min_size:
                        self.components.merge(u, v)
        #endregion
        

    
    def __create_segmented_arr(self) -> None:
        """ Creates the image array in which each pixel has a value equal to its parent node.

        The array will be composed of elements having some value according to the region they belong to.
        """
        parents = self.components.parents()

        self.segmented_arr = np.zeros((self.height, self.width), np.uint8)

        for y in range(self.height):
            for x in range(self.width):
                self.segmented_arr[y, x] = parents.index(self.components.parent[y * self.width + x])



    def generate_image(self) -> None:
        """ Generates the segmented colored image.

        The `segmented_arr` is converted into a PIL.Image (with random colors for each region).
        """
        
        #region - initialization
        random_color = lambda: (int(np.random.rand() * 255), int(np.random.rand() * 255), int(np.random.rand() * 255))
        color = [random_color() for i in range(self.components.num_components())]

        self.segmented_img = np.zeros((self.height, self.width, 3), np.uint8)

        if not hasattr(self, 'segmented_arr'):
            self.__create_segmented_arr()
        #endregion 

        #region - image generation
        for y in range(self.height):
            for x in range(self.width):
                self.segmented_img[y, x] = color[self.segmented_arr[y, x]]
        
        self.segmented_img = Image.fromarray(self.segmented_img)
        #endregion



    def __find_boundaries(self) -> None:
        """ Finds the boundaries of each region in the segmented image.

        By looping over the rows and columns we look for the extremes pixels of each region:
        * min_col: the column having smaller index which delimitates the region
        * min_row: the row having smaller index which delimitates the region
        * max_col: the column having greater index which delimitates the region
        * min_row: the row having greater index which delimitates the region

        The boundaries are a dictionary having as keys `min_col`, `min_row`, `max_col`, `max_row`.
        """

        #region - initialization
        self.boundaries = {}
        for i in range(self.components.num_components()):
            self.boundaries[i] = {
                              "min_row": self.height - 1
                            , "max_row": 0
                            , "min_col": self.width - 1
                            , "max_col": 0
                            }
        #endregion

        #region - boundaries calculation
        for row in range(self.height):
            for col in range(self.width):
            
                min_row = self.boundaries[self.segmented_arr[row, col]]['min_row']
                max_row = self.boundaries[self.segmented_arr[row, col]]['max_row']
                min_col = self.boundaries[self.segmented_arr[row, col]]['min_col']
                max_col = self.boundaries[self.segmented_arr[row, col]]['max_col']

                if (row < min_row):
                    self.boundaries[self.segmented_arr[row, col]]['min_row'] = row

                if (row > max_row):
                    self.boundaries[self.segmented_arr[row, col]]['max_row'] = row

                if (col < min_col):
                    self.boundaries[self.segmented_arr[row, col]]['min_col'] = col

                if (col > max_col):
                    self.boundaries[self.segmented_arr[row, col]]['max_col'] = col
        #endregion



    def digits_boxes_and_areas(self) -> None:
        """ Draws boxes around digits (regions) based on their boundaries and compute their area.

        The function builds:
        * an image (`boxed_img`) with green boxes around each digit.
        * a dictionary (`digits_regions`) which contains the coordinates and the area of each region.
        """

        #region - initialization
        if not hasattr(self, 'boundaries'):
            self.__find_boundaries()

        if not hasattr(self, 'segmented_img'):
            self.generate_image()

        self.boxed_img = self.segmented_img.copy()
        draw = ImageDraw.Draw(self.boxed_img)

        self.digits_regions = {}
        
        # counter for the digits_region dictionary keys
        counter = 0     
        
        # the area of the background
        max_area = (self.width-1) * (self.height-1)
        #endregion
        
        #region - boxes and areas calculation
        for _, extremes in self.boundaries.items():

            area = (extremes['max_row'] - extremes['min_row']) *\
                   (extremes['max_col'] - extremes['min_col'])

            # remove the background and the regions having area == 0
            if area == 0 or area == max_area: continue 

            A = (extremes['min_col'], extremes['min_row'])
            B = (extremes['max_col'], extremes['min_row'])
            C = (extremes['max_col'], extremes['max_row'])
            D = (extremes['min_col'], extremes['max_row'])
            
            self.digits_regions[counter] = {'extremes': [i for _, i in extremes.items()], 'area': area}

            counter += 1

            draw.line([A,B,C,D,A], fill='lightgreen', width=3)
        
        # sort the regions by min_col in order to obtain the ordered digits (as written on paper from left to right)
        self.sorted_keys = sorted(self.digits_regions.keys(), key=lambda x: self.digits_regions[x]['extremes'][2])
        #endregion


    
    def extract_digits(self) -> None:
        """ Extracts the single digits from the segmented image. 
        
        Once the regions' boundaries are found:
        * the regions are sliced out from the original image
        * the slices are resized according to the MNIST dataset samples dimensions (28x28)
        * the resized slices are modified in order to obtain an image which is as close as possible to the one that the network saw in training phase
        * the modified slices are converted into a `torch.tensor` which will be used as input to the network
        """
        
        digits = []
        for k in self.sorted_keys:
            
            # find digit extremes
            a = self.digits_regions[k]['extremes'][0]
            b = self.digits_regions[k]['extremes'][1] + 1
            c = self.digits_regions[k]['extremes'][2]
            d = self.digits_regions[k]['extremes'][3] + 1

            # slice original image array around digit
            digit = self.preprocessed_arr[a:b,c:d].copy()

            # apply threshold to move background to white
            threshold = lambda el, t: np.uint8(el) if el < t else np.uint8(255)
            threshold_func = np.vectorize(threshold)
            digit = threshold_func(digit, 70)

            # resize image
            height, width = digit.shape
            if height > width:
                diff = height - width
                left_cols_num = ceil((diff) / 2)
                right_cols_num = diff - left_cols_num
                digit = np.pad(digit, ((0,0),(right_cols_num, left_cols_num)), 'maximum')
            else:
                diff = width - height
                top_rows_num = ceil((diff) / 2)
                bottom_rows_num = diff - top_rows_num
                digit = np.pad(digit, ((bottom_rows_num, top_rows_num),(0,0)), 'maximum')
            
            digit = Image.fromarray(digit)  # convert to PIL
            digit = digit.resize((28,28))   # resize to 28x28 as MNIST input
            digit = np.array(digit)         # convert to np.array

            # make negative, since we want black background
            negative = lambda el: np.uint8(255 - el)
            negative_func = np.vectorize(negative)
            digit = negative_func(digit)

            digits.append(digit)


        digits = torch.FloatTensor(np.array(digits))
        self.digits = torch.unsqueeze(digits, 1)    # add one dimension (for Conv2d)
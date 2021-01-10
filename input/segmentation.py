'''
Copyright January 2021 - Filippo Guerranti <filippo.guerranti@student.unisi.it>

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw
import torch.tensor
from math import ceil


class DisjointSetForest:
    ''' 
    Disjoint-set forest data structure (https://en.wikipedia.org/wiki/Disjoint-set_data_structure).
    Stores a collection of disjoint non-overlapping sets (or a partition of a set into disjoint subsets).
    Provides operations for adding new sets, merging sets (replacing them by their union) and find a 
    representative member of a set. 
    '''

    def __init__(
          self
        , num_nodes: int
        ) -> None:
        ''' 
        Disjoint-set forest class constructor.

        Args:
            num_nodes (int): total number of elements to be partitioned.
        '''
        # list providing the parent of the indexed node (list)
        self.parent = [i for i in range(num_nodes)]

        # list providing the rank of the indexed node (list)
        self.rank = [0 for i in range(num_nodes)]

        # list providing the size of the child tree of the indexed node (list)
        self.size = [1 for i in range(num_nodes)]


    
    def size_of(  
          self
        , u: int
        ) -> int:
        ''' 
        Return the size of the child tree of the given component 
        
        Args:
            u (int): node of which we want to know the size.
        
        Returns:
            size (int): number of nodes in the subtree of u.    
        '''
        return self.size[u]



    def find(
        self
        , u: int
        ) -> int:
        ''' 
        Return the representative of a subset.

        Args:
            u (int): node of which we want to know the representative.

        Returns:
            representative of the subset/node.
        '''
        if self.parent[u] == u:
            return u

        self.parent[u] = self.find(self.parent[u])
        return self.parent[u]



    def merge(
          self
        , u: int
        , v: int
        ) -> None:
        ''' 
        Merge two nodes/subsets into one based on their rank.

        Args:
            u (int): first node/subset
            v (int): second node/subset
        '''
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
        ''' 
        Return the number of current components.
        
        Returns:
            num_components (int): number of current components
        '''
        return len(self.parents())



    def parents(self) -> list:
        ''' 
        Return the parent nodes.

        Returns:
            parents (list): list of parent nodes
        '''
        return list(set(self.parent))



    def sorted_parents(self) -> list:
        ''' 
        Return the parents nodes sorted by size in descreasing order.

        Returns:
            sorted_parents (list): list of sorted parent nodes
        '''
        return sorted(
                      self.parents()
                    , key=lambda parent: 
                            self.size_of(parent)
                    , reverse=True
                    )



class GraphBasedSegmentation:
    ''' 
    Class for the implementation of the graph-based segmentation algorithm.
    A graph is built starting from an input image.
    Each pixel is a vertex of G = (V, E) where G is the graph, V is the set of
    vertices and E is the set of edges. Each pixel is connected to its neighbors in
    an 8-grid sense.
    Then the algorithm is applied exploiting the disjoint-set forest structure.
    '''

    def __init__(
          self
        , img: str or np.ndarray
        ) -> None:
        ''' 
        GraphBasedSegmentation class constructor.

        Args:
            img (str or np.ndarray): path to the input image (if preprocessing == True)
                                     or np.ndarray of the input image (already preprocessed)
        '''
        # from path
        # ---------------------------------
        if type(img) == str:
            self.img = Image.open(img)
        # ---------------------------------

        # from array
        # ---------------------------------
        elif type(img) == np.ndarray:
            self.img = Image.fromarray(img)
        # ---------------------------------

        # wrong input
        # ---------------------------------
        else:
            raise ValueError("Wrong image type: must be either str or np.ndarray.")
        # ---------------------------------

        self.img_width, self.img_height = self.img.size
    


    @staticmethod
    def __preprocessing(
          img: Image
        , contrast: float=1.5
        , gaussian_blur: float=2.3
        , width: int=300
        , height: int=None
        ) -> np.ndarray:
        ''' 
        Convert an input RGB image to a grayscale Numpy array and apply some preprocessing operations.

        Args:
            img         (PIL.Image): image to be processed
            constrast       (float): (defualt=1.5) contrast filter
            gaussian_blur   (float): (default=1.5) Gaussian Blur filter
            width             (int): (default=300) new image width
            height            (int): (default=None) new image height
        
        Returns:
            img (np.ndarray): Numpy array represented the preprocessed image
        '''
        # grayscale conversion
        # ---------------------------------
        img = img.convert("L")
        # ---------------------------------
        
        # gaussian blur
        # ---------------------------------
        img = img.filter(ImageFilter.GaussianBlur(gaussian_blur))
        # ---------------------------------

        # contrast
        # ---------------------------------
        img = ImageEnhance.Contrast(img).enhance(contrast)
        # ---------------------------------

        # resize
        # ---------------------------------
        if height is None:
            percentage = float(width / img.size[0])
            height = int((float(img.size[1]) * float(percentage)))
        img = img.resize((width, height), Image.ANTIALIAS)
        # ---------------------------------
        
        return np.array(img)



    @staticmethod
    def __get_diff(
          img: np.ndarray
        , u_coords: tuple
        , v_coords: tuple
        ) -> np.uint8:
        ''' 
        Return the difference in terms of intensity between two pixels of an image.
        
        Args:
            img (numpy.ndarray): input grayscale image in array format (values from 0 to 255)
            u_coords    (tuple): coordinates of first pixel (x1, y1)
            v_coords    (tuple): coordinates of second pixel (x2, y2)

        Returns:
            diff (numpy.uint8): difference between the input pixels
        '''
        x1, y1 = u_coords
        x2, y2 = v_coords
        
        # necessary check since pixels are uint8 (problem with abs)
        if img[y1, x1] > img[y2, x2]:
            return img[y1, x1] - img[y2, x2]
        
        return img[y2, x2] - img[y1, x1]



    @staticmethod
    def __create_edge(
          img: np.ndarray
        , u_coords: tuple
        , v_coords: tuple
        ) -> tuple:
        ''' 
        Create the edge between two pixels of the input image.

        Args:
            img (numpy.ndarray): input grayscale image in array format (values from 0 to 255)
            u_coords (tuple): coordinates of first pixel (x1, y1)
            v_coords (tuple): coordinates of second pixel (x2, y2)

        Returns:
            id1 (int): first pixel id
            id2 (int): second pixel id
            weight (numpy.uint8): edge weight
        '''
        _, width = img.shape

        vertex_id = lambda coords: coords[1] * width + coords[0]
        id1 = vertex_id(u_coords)
        id2 = vertex_id(v_coords)

        weight = GraphBasedSegmentation.__get_diff(img, u_coords, v_coords)
        
        return (id1, id2, weight)

    

    @staticmethod
    def __threshold(
          k: int
        , size: int
        ) -> int:
        ''' 
        Define the threshold for a subset of cardinality given by size.

        Args:
            k       (int): scale of observation (large k -> larger components)
            size    (int): cardinality of the component into consideration
        
        Returns:
            threshold (int): threshold for a given component
        '''
        return int(k/size)



    def __build_graph(self) -> None:
        ''' 
        Build the graph.

        Returns:
            graph (list): list of tuples (u, v, weight) 
        '''
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
        ''' 
        Sort the graph by non-decreasing order of edges' weight.
        '''
        self.sorted_graph = sorted(
                                  self.graph
                                , key=lambda edge: edge[2]
                                )

    

    def segment(
          self
        , k: int=4000
        , min_size: int=100
        , preprocessing: bool=True
        , **kwargs
        ) -> None:
        ''' 
        Segment the graph according to the graph-based segmentation algorithm
        proposed by Felzenszwalb et. al.

        Args:
            k               (int): (default=4000) parameter for the threshold
            min_size        (int): (default=100) if specified, the components having size less than min_size are removed
                                          if None, the removal is not applied
            preprocessing  (bool): (default=True) to be applied if the image has not been preprocessed yet
        '''
        # preprocessing
        # ---------------------------------
        if preprocessing:
            self.preprocessed_arr = GraphBasedSegmentation.__preprocessing(self.img, **kwargs)
        # ---------------------------------
        
        # not preprocessing
        # ---------------------------------
        else:
            self.preprocessed_arr = self.img
            self.preprocessed_arr = self.preprocessed_arr.convert("L")
            self.preprocessed_arr = np.array(self.preprocessed_arr)
        # ---------------------------------

        self.height, self.width = self.preprocessed_arr.shape
        self.num_nodes = self.height * self.width

        # Disjoint-set forest initialization
        # ---------------------------------
        self.components = DisjointSetForest(self.num_nodes)
        # ---------------------------------

        # threshold list
        threshold = [GraphBasedSegmentation.__threshold(k, i) for i in self.components.size]

        # graph build and sorting by non-decreasing order of weights
        # ---------------------------------
        self.__build_graph()
        self.__sort()
        # ---------------------------------


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
        # ---------------------------------
        if min_size is not None:

            for edge in self.sorted_graph:
                u, v, _ = edge
                u = self.components.find(u)
                v = self.components.find(v)

                if u != v:
                    if self.components.size_of(u) < min_size or self.components.size_of(v) < min_size:
                        self.components.merge(u, v)
        # ---------------------------------
        

    
    def __create_segmented_arr(self) -> None:
        ''' 
        Create the image array in which each pixel has a value equal to its parent node.
        So, at the end the array will be composed of elements having some value according to
        the region they belong to.
        '''
        parents = self.components.parents()

        self.segmented_arr = np.zeros((self.height, self.width), np.uint8)

        for y in range(self.height):
            for x in range(self.width):
                self.segmented_arr[y, x] = parents.index(self.components.parent[y * self.width + x])



    def generate_image(self) -> None:
        ''' 
        Generate the segmented image as PIL Image from the array created with `__create_segmented_arr()`.
        '''
        
        # random color creation (3 levels of values between 0 and 255)
        # ---------------------------------
        random_color = lambda: (int(np.random.rand() * 255), int(np.random.rand() * 255), int(np.random.rand() * 255))
        color = [random_color() for i in range(self.components.num_components())]
        # ---------------------------------

        self.segmented_img = np.zeros((self.height, self.width, 3), np.uint8)

        if not hasattr(self, 'segmented_arr'):
            self.__create_segmented_arr()

        # generate the image with random colors, each pixel has the color of the corresponding parent
        # ---------------------------------
        for y in range(self.height):
            for x in range(self.width):
                self.segmented_img[y, x] = color[self.segmented_arr[y, x]]
        # ---------------------------------
        
        self.segmented_img = Image.fromarray(self.segmented_img)



    def __find_boundaries(self) -> None:
        ''' 
        Find the boundary of each region in the segmented image.
        By looping over the rows and columns we look for the extremes pixels of each region
        and set them as min_col, min_row, max_col, min_row.
        '''

        self.boundaries = {}

        # initialize the boundaries of each region
        # ---------------------------------
        for i in range(self.components.num_components()):
            self.boundaries[i] = {
                              "min_row": self.height - 1
                            , "max_row": 0
                            , "min_col": self.width - 1
                            , "max_col": 0
                            }
        # ---------------------------------

        # compute the boundaries by iterating over rows and cols of the image
        # ---------------------------------
        for row in range(self.height):
            for col in range(self.width):
            
                min_row = self.boundaries[self.segmented_arr[row, col]]['min_row']
                max_row = self.boundaries[self.segmented_arr[row, col]]['max_row']
                min_col = self.boundaries[self.segmented_arr[row, col]]['min_col']
                max_col = self.boundaries[self.segmented_arr[row, col]]['max_col']

                # update the boundaries of each region
                # ---------------------------------
                if (row < min_row):
                    self.boundaries[self.segmented_arr[row, col]]['min_row'] = row

                if (row > max_row):
                    self.boundaries[self.segmented_arr[row, col]]['max_row'] = row

                if (col < min_col):
                    self.boundaries[self.segmented_arr[row, col]]['min_col'] = col

                if (col > max_col):
                    self.boundaries[self.segmented_arr[row, col]]['max_col'] = col
                # ---------------------------------
        # ---------------------------------



    def draw_boxes(self) -> None:
        ''' 
        Draw boxes around digits based on boundaries (found using `__find_foundaries()`).
        At the same time build the dictionary `digits_regions` which contains the coordinates and the
        area of each digit.
        '''

        if not hasattr(self, 'boundaries'):
            self.__find_boundaries()

        if not hasattr(self, 'segmented_img'):
            self.generate_image()

        # image having boxes around the digits
        self.boxed_img = self.segmented_img.copy()

        # draw object to draw the boxes
        draw = ImageDraw.Draw(self.boxed_img)

        # dictionary containing info about the regions of the digits
        self.digits_regions = {}
        
        # counter for the digits_region dictionary keys
        counter = 0     
        
        # the area of the background
        max_area = (self.width-1) * (self.height-1)
        
        for _, extremes in self.boundaries.items():

            # compute the area of the region
            # ---------------------------------
            area = (extremes['max_row'] - extremes['min_row']) *\
                   (extremes['max_col'] - extremes['min_col'])
            # ---------------------------------

            # remove the background and the ones having area == 0
            # ---------------------------------
            if area == 0 or area == max_area: continue 
            # ---------------------------------

            A = (extremes['min_col'], extremes['min_row'])
            B = (extremes['max_col'], extremes['min_row'])
            C = (extremes['max_col'], extremes['max_row'])
            D = (extremes['min_col'], extremes['max_row'])
            
            # create the dictionary of the digits regions
            # ---------------------------------
            self.digits_regions[counter] = {'extremes': [i for _, i in extremes.items()], 'area': area}
            # ---------------------------------

            counter += 1

            draw.line([A,B,C,D,A], fill='lightgreen', width=3)
        
        # sort the regions by min_col in order to obtain the ordered digits
        self.sorted_keys = sorted(self.digits_regions.keys(), key=lambda x: self.digits_regions[x]['extremes'][2])


    
    def extract_digits(self) -> None:
        """ 
        Extract the single digits from the segmented image.
        """
        
        digits = []
        for k in self.sorted_keys:
            
            # find digit extremes
            # ---------------------------------
            a = self.digits_regions[k]['extremes'][0]
            b = self.digits_regions[k]['extremes'][1] + 1
            c = self.digits_regions[k]['extremes'][2]
            d = self.digits_regions[k]['extremes'][3] + 1
            # ---------------------------------

            # slice original image array around digit
            # ---------------------------------
            digit = self.preprocessed_arr[a:b,c:d].copy()
            # ---------------------------------

            # apply threshold to move background to white
            # ---------------------------------
            threshold = lambda el, t: np.uint8(el) if el < t else np.uint8(255)
            threshold_func = np.vectorize(threshold)
            digit = threshold_func(digit, 50)
            # ---------------------------------

            # resize image
            # ---------------------------------
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
            # ---------------------------------

            # make negative, since we want black background
            # ---------------------------------
            negative = lambda el: np.uint8(255 - el)
            negative_func = np.vectorize(negative)
            digit = negative_func(digit)
            # ---------------------------------

            digits.append(digit)


        # convert to torch tensor
        # ---------------------------------
        digits = torch.FloatTensor(np.array(digits))
        # ---------------------------------

        # convert


        self.digits = torch.unsqueeze(digits, 1)
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
from PIL import Image, ImageFilter, ImageEnhance
import DisjointSetForest as DSF
import time


class GraphBasedSegmentation:
    """ Class for the implementation of the graph-based segmentation algorithm.
    A graph is built starting from an input image.
    Each pixel is a vertex of G = (V, E) where G is the graph, V is the set of
    vertices and E is the set of edges. Each pixel is connected to its neighbors in
    an 8-grid sense.
    Then the algorithm is applied exploiting the disjoint-set forest structure.
    """

    # Class members
    filepath = None         # image filepath (str)
    img = None              # original image (PIL.Image)
    img_height = None       # original image height (int)
    img_width = None        # original image width (int)
    preprocessed_arr = None # preprocessed array (np.ndarray)
    segmented_img = None    # segmented image (PIL.Image)
    segmented_arr = None    # segmented array (np.ndarray)
    pre_height = None       # preprocessed array height (int)
    pre_width = None        # preprocessed array width (int)
    num_nodes = None        # number of nodes (a.k.a. number of pixels) (int)
    graph = None            # graph of the image (list)
    sorted_graph = None     # sorted graph by non-decreasing order of edges' weight
    components = None       # Disjoint-set forest containing the components of the segmentation
    threshold = None        # threshold for each component: k/|C| (int)


    def __init__(self, img: str or np.ndarray):
        """ GraphBasedSegmentation class constructor.

        Args:
            img (str or np.ndarray): path to the input image (if preprocessing == True)
                                     or np.ndarray of the input image (already preprocessed)
        """
        if type(img) == str:
            self.img = Image.open(img)
            self.filepath = img
        elif type(img) == np.ndarray:
            self.img = Image.fromarray(img)
        else:
            raise ValueError("Wrong image type: must be either str or np.ndarray.")

        self.img_width, self.img_height = self.img.size
    

    @staticmethod
    def _preprocessing(img: Image, contrast: float=1.5, gaussian_blur: float=2.3, width: int=300, height: int=None):
        """ Convert an input RGB image to a grayscale Numpy array and apply some preprocessing operations.

        Args:
            img (PIL.Image): image to be processed
            constrast (float): (defualt=1.5) contrast filter
            gaussian_blur (float): (default=1.5) Gaussian Blur filter
            width (int): (default=300) new image width
            height (int): (default=None) new image height
        
        Returns:
            img (np.ndarray): Numpy array represented the preprocessed image
        """
        img = img.convert("L")
        
        # gaussian blur
        img = img.filter(ImageFilter.GaussianBlur(gaussian_blur))

        # contrast
        img = ImageEnhance.Contrast(img).enhance(contrast)

        # resize
        if height == None:
            percentage = float(width / img.size[0])
            height = int((float(img.size[1]) * float(percentage)))
        img = img.resize((width, height), Image.ANTIALIAS)
        
        return np.array(img)



    @staticmethod
    def _get_diff(img: np.ndarray, u_coords: tuple, v_coords: tuple):
        """ Return the difference in terms of intensity between two pixels of an image.
        
        Args:
            img (numpy.ndarray): input grayscale image in array format (values from 0 to 255)
            u_coords (tuple): coordinates of first pixel (x1, y1)
            v_coords (tuple): coordinates of second pixel (x2, y2)

        Returns:
            diff (numpy.uint8): difference between the input pixels
        """
        x1, y1 = u_coords
        x2, y2 = v_coords
        
        # necessary check since pixels are uint8 (problem with abs)
        if img[y1, x1] > img[y2, x2]:
            return img[y1, x1] - img[y2, x2]
        
        return img[y2, x2] - img[y1, x1]



    @staticmethod
    def _create_edge(img: np.ndarray, u_coords: tuple, v_coords: tuple):
        """ Create the edge between two pixels of the input image.

        Args:
            img (numpy.ndarray): input grayscale image in array format (values from 0 to 255)
            u_coords (tuple): coordinates of first pixel (x1, y1)
            v_coords (tuple): coordinates of second pixel (x2, y2)

        Returns:
            id1 (int): first pixel id
            id2 (int): second pixel id
            weight (numpy.uint8): edge weight
        """
        _, width = img.shape

        vertex_id = lambda coords: coords[1] * width + coords[0]
        id1 = vertex_id(u_coords)
        id2 = vertex_id(v_coords)

        weight = GraphBasedSegmentation._get_diff(img, u_coords, v_coords)
        
        return (id1, id2, weight)

    

    @staticmethod
    def _threshold(k: int, size: int):
        """ Define the threshold for a subset of cardinality given by size.

        Args:
            k (int): scale of observation (large k -> larger components)
            size (int): cardinality of the component into consideration
        
        Returns:
            threshold (int): threshold for a given component
        """
        return int(k/size)



    def _build_graph(self):
        """ Build the graph.

        Returns:
            graph (list): list of tuples (u, v, weight) 
        """
        self.graph = []

        print("Building graph...")
        start = time.time()
        for y in range(self.pre_height):
            for x in range(self.pre_width):
                if x < self.pre_width - 1:
                    u_coords = (x, y)
                    v_coords = (x + 1, y)
                    self.graph.append(GraphBasedSegmentation._create_edge(self.preprocessed_arr, u_coords, v_coords))
                if y < self.pre_height - 1:
                    u_coords = (x, y)
                    v_coords = (x, y + 1)
                    self.graph.append(GraphBasedSegmentation._create_edge(self.preprocessed_arr, u_coords, v_coords))
                if x < self.pre_width - 1 and y < self.pre_height - 1:
                    u_coords = (x, y)
                    v_coords = (x + 1, y + 1)
                    self.graph.append(GraphBasedSegmentation._create_edge(self.preprocessed_arr, u_coords, v_coords))
                if x < self.pre_width - 1 and y > 0:
                    u_coords = (x, y)
                    v_coords = (x + 1, y - 1)
                    self.graph.append(GraphBasedSegmentation._create_edge(self.preprocessed_arr, u_coords, v_coords))
        end = time.time()
        print("Graph built in {:.3}s.\n".format(end-start))
    

    def _sort(self):
        """ Sort the graph by non-decreasing order of edges' weight.
        
        Returns:
            sorted_graph (list): sorted graph
        """
        self.sorted_graph = sorted(self.graph, key=lambda edge: edge[2])

    

    def segment(self, k: int=4000, min_size: int=100, preprocessing: bool=True, **kwargs):
        """ Segment the graph according to the graph-based segmentation algorithm
        proposed by Felzenszwalb et. al.

        Args:
            k (int): (default=4000) parameter for the threshold
            min_size (int): (default=100) if specified, the components having size less than min_size are removed
                                          if None, the removal is not applied
            preprocessing (bool): (default=True) to be applied if the image has not been preprocessed yet
        
        Returns:
            components (DisjointSetForest): Disjoint-set Forest containing the segmented components
        """
        if preprocessing:
            self.preprocessed_arr = GraphBasedSegmentation._preprocessing(self.img, **kwargs)
        else:
            self.preprocessed_arr = self.img
            self.preprocessed_arr = self.preprocessed_arr.convert("L")
            self.preprocessed_arr = np.array(self.preprocessed_arr)

        self.pre_height, self.pre_width = self.preprocessed_arr.shape
        self.num_nodes = self.pre_height * self.pre_width

        self.components = DSF.DisjointSetForest(self.num_nodes)
        threshold = [GraphBasedSegmentation._threshold(k, i) for i in self.components.size]

        self._build_graph()
        self._sort()

        print("Segmenting...")
        start = time.time()
        for edge in self.sorted_graph:
            u, v, w = edge

            u = self.components.find(u)
            v = self.components.find(v)

            if u != v:
                if w <= threshold[u] and w <= threshold[v]:
                    self.components.merge(u, v)
                    parent = self.components.find(u)
                    threshold[parent] = w + GraphBasedSegmentation._threshold(k, self.components.size_of(u))
        end = time.time()
        print("Segmentation done in {:.3}s.\n".format(end-start))

        # remove components having size less than min_size
        if min_size != None:
            print("Removing componentes having size less than {}...".format(min_size))
            start = time.time()
            for edge in self.sorted_graph:
                u, v, _ = edge
                u = self.components.find(u)
                v = self.components.find(v)

                if u != v:
                    if self.components.size_of(u) < min_size or self.components.size_of(v) < min_size:
                        self.components.merge(u, v)
            end = time.time()
            print("Removed components in {:.3}s.\n".format(end-start))
        

    
    def define_regions(self):
        """ Define the segmentation regions once the segmentation process is completed.

        Returns:
            segmented_arr (np.ndarray): segmented image array
        """
        parents = self.components.parents()

        self.segmented_arr = np.zeros((self.pre_height, self.pre_width), np.uint8)
        
        print("Defining regions...")
        start = time.time()
        for y in range(self.pre_height):
            for x in range(self.pre_width):
                self.segmented_arr[y, x] = parents.index(self.components.find(y * self.pre_width + x))
        
        end = time.time()
        print("Regions defined in {:.3}s.\n".format(end-start))



    def generate_image(self):
        """ Generate the segmented image as a numpy array.

        Returns:
            segmented_img (PIL.Image): segmented image

        """
        random_color = lambda: (int(np.random.rand() * 255), int(np.random.rand() * 255), int(np.random.rand() * 255))
        color = [random_color() for i in range(self.components.num_components())]

        self.segmented_img = np.zeros((self.pre_height, self.pre_width, 3), np.uint8)

        if self.segmented_arr == None:
            self.define_regions()

        print("Generating image...")
        start = time.time()
        for y in range(self.pre_height):
            for x in range(self.pre_pre_width):
                self.segmented_img[y, x] = color[self.segmented_arr[y, x]]
        
        self.segmented_img = Image.fromarray(self.segmented_img)
        end = time.time()
        print("Image generated in {:.3}s.\n".format(end-start))
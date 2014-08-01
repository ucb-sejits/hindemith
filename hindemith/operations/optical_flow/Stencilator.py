__author__ = 'chick'

import numpy as np


class Stencilator(object):
    def __init__(self, neighbor_hood):
        pass

    def create_destination(self, numpy_array):
        """
        override this method to create pyramidal reductions or expansions
        :param numpy_array:
        :return:
        """
        return np.empty_like(input)

    def stencil_function(self, input_arrays, ):
        pass

    def kernel(self, input_numpy_array, point):
        destination = self.create_destination(input_numpy_array)
        return input_numpy_array[ point ]

    @staticmethod
    def von_neuman_neighborhood(self, dim):
        """
        create a neighborhood of all adjacent points along
        coordinate axes, suitable for the dimension of this instance
        """
        neighborhood = []
        origin = [0 for _ in range(dim)]
        for dimension in range(dim):
            for offset in [-1, 1]:
                point = origin[:]
                point[dimension] = offset
                neighborhood.append(tuple(point))

        return neighborhood

    @staticmethod
    def moore_neighborhood(self, dim, include_origin=False):
        """
        create a neighborhood of all adjacent points along
        coordinate axes
        """

        neighborhood_list = []

        def dimension_iterator(dimension, point_accumulator):
            """
            accumulates into local neighborhood_list
            """
            if dimension >= dim:
                if include_origin or sum([abs(x) for x in point_accumulator]) != 0:
                    neighborhood_list.append(tuple(point_accumulator))
            else:
                for dimension_coordinate in [-1, 0, 1]:
                    new_point_accumulator = point_accumulator[:]
                    new_point_accumulator.append(dimension_coordinate)
                    dimension_iterator(
                        dimension+1,
                        new_point_accumulator
                    )

        dimension_iterator(0, [])

        return neighborhood_list



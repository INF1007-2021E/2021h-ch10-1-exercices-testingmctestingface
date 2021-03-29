#!/usr/bin/env python
# -*- coding: utf-8 -*-


# TODO: Importez vos modules ici
import numpy as np
import matplotlib.pyplot as plt

# TODO: Définissez vos fonctions ici (il en manque quelques unes)
def linear_values() -> np.ndarray:
    return np.array(np.linspace(-1.3,2.5,64))


def coordinate_conversion(cartesian_coordinates: np.ndarray) -> np.ndarray:
    polar_coordinates = []
    for cord in cartesian_coordinates:
        x,y = cord
        r = np.sqrt(x**2 + y**2)
        tet = np.arctan2(y,x)
        polar_coordinates.append((r,tet))
    return np.array(polar_coordinates)


def find_closest_index(values: np.ndarray, number: float) -> int:
    return np.abs(values - number).argmin()


def graphique():
    x = np.linspace(-1,1,250)
    y = x**2 * np.sin(x**-2) + x
    plt.scatter(x,y, label="y en fonction de x^2 * sin(1/x^2) + x")
    plt.xlim(-2,2)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Le titre")
    plt.legend()
    plt.show()
    return None

def monte_carlo(itt):
    x = np.random.rand(itt)
    y = np.random.rand(itt)
    r = np.sqrt((x ** 2) + (y ** 2))
    color = np.where(r<=1,"navy","darkorange")
    plt.scatter(x,y,c=color)
    plt.show()
    ratio = np.count_nonzero(r <= 1)/itt
    return ratio*4

if __name__ == '__main__':
    # TODO: Appelez vos fonctions ici
    """print(linear_values())
    print(coordinate_conversion(np.array([(3,4),(7,1),(29,4)])))
    print(find_closest_index(np.array([5,8,3]),4))
    graphique()"""
    print(monte_carlo(5000))
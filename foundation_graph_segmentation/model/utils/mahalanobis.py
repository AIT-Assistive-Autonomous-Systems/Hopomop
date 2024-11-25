import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
from matplotlib.patches import Ellipse

def mahalanobis_distance(points):
    # Calculate mean vector and covariance matrix
    mean_vector = np.mean(points, axis=0)
    covariance_matrix = np.cov(points, rowvar=False)

    # make sure the covariance matrix is invertible
    if np.linalg.matrix_rank(covariance_matrix) < covariance_matrix.shape[0]:
        return np.zeros(points.shape[0])

    # Compute inverse of the covariance matrix
    inv_covariance_matrix = np.linalg.inv(covariance_matrix)

    # Subtract mean vector from each data point
    centered_points = points - mean_vector

    # Calculate Mahalanobis distance for each point
    mahalanobis_distances = []

    for point in centered_points:
        mahalanobis_dist = np.sqrt(np.dot(point.T, np.dot(inv_covariance_matrix, point)))
        mahalanobis_distances.append(mahalanobis_dist)

    return np.array(mahalanobis_distances)




def plot_mahalanobis_ellipse(points, ax=None, confidence=0.95, **kwargs):
    if ax is None:
        ax = plt.gca()

    # Calculate mean vector and covariance matrix
    mean_vector = np.mean(points, axis=0)
    covariance_matrix = np.cov(points, rowvar=False)
    
    # Compute eigenvalues and eigenvectors of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    
    # Get the angle of rotation from the eigenvectors
    angle = np.degrees(np.arctan2(*eigenvectors[:,1]))
    
    # Calculate the Mahalanobis radius based on the chi-square distribution
    num_dimensions = points.shape[1]
    chi2_value = chi2.ppf(confidence, df=num_dimensions)
    mahalanobis_radius = np.sqrt(chi2_value)

    # Generate points on the unit circle
    t = np.linspace(0, 2 * np.pi, 100)
    circle = np.vstack((np.cos(t), np.sin(t))).T
    
    # Scale and rotate the circle to match the ellipse
    ellipse = mahalanobis_radius * np.dot(circle, eigenvectors.T) + mean_vector
    
    # Plot the ellipse
    ax.plot(ellipse[:, 0], ellipse[:, 1], **kwargs)

    return ax


def draw_ellipse_from_covariance(points, ax=None, **kwargs):

    if points.shape[0] < 2:
        return

    # Compute covariance matrix
    covariance_matrix = np.cov(points, rowvar=False)
    
    # Eigen decomposition of covariance matrix
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    
    # Find major and minor axes lengths and rotation angle
    major_axis_length = 2 * np.sqrt(2.0 * eigenvalues[0])
    minor_axis_length = 2 * np.sqrt(2.0 * eigenvalues[1])
    rotation_angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
    
    # Plot ellipse
    if ax is None:
        ax = plt.gca()
    ellipse = Ellipse(xy=np.mean(points, axis=0), width=major_axis_length, height=minor_axis_length, angle=rotation_angle, **kwargs)
    ax.add_patch(ellipse)


# Example usage
points = np.random.multivariate_normal(mean=[0, 0], cov=[[3, 1], [1, 1]], size=100)
draw_ellipse_from_covariance(points)
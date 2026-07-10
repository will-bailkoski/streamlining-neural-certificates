import numpy as np
from scipy.interpolate import LinearNDInterpolator


def mesh_generator(points, values):
    def build():
        if len(points) < 3:
            return None

        return LinearNDInterpolator(np.array(points), np.array(values))

    new_data = yield build()

    while True:
        if new_data:
            new_data = np.atleast_2d(new_data)
            for coord_array, value in new_data:
                points.append(coord_array.tolist())
                values.append(value)

        new_data = yield build()


def delaunay_mesh_approximation(f, domain, n_points=1000, random=False):
    domain = np.asarray(domain)
    dim = domain.shape[0]

    if random:
        points = np.random.uniform(
            low=domain[:, 0], high=domain[:, 1], size=(n_points, dim)
        )
    else:
        n_per_dim = int(np.ceil(n_points ** (1 / dim)))

        axes = [np.linspace(domain[d, 0], domain[d, 1], n_per_dim) for d in range(dim)]

        mesh = np.meshgrid(*axes, indexing="xy")
        points = np.stack(mesh, axis=-1).reshape(-1, dim)

    zs = f(points)

    gen = mesh_generator(points.tolist(), zs.tolist())

    return gen


def max_gradient(interp, ord=2):
    tri = interp.tri
    pts = tri.points
    vals = interp.values
    simplices = tri.simplices

    max_norm = 0.0

    for simplex in simplices:
        X = pts[simplex]  # (d+1, d)
        y = vals[simplex]  # (d+1,)

        A = (X[1:] - X[0]).T  # (d, d)
        b = y[1:] - y[0]  # (d,)

        # solve for gradient
        try:
            g = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            g = np.linalg.lstsq(A, b, rcond=None)[0]

        max_norm = max(max_norm, np.linalg.norm(g, ord=2))

    return max_norm

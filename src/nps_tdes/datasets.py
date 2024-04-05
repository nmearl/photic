from torch.utils.data import Dataset, DataLoader
from astropy.modeling.models import SmoothlyBrokenPowerLaw1D
import numpy as np
import torch


class MockData(Dataset):
    def __init__(
        self,
        *args, num_samples=1000, source='SBPL', **kwargs
    ):
        self.data = []
        self.num_samples = num_samples

        for i in range(num_samples):
            if source == 'SBPL':
                x, y = generate_mock_data(*args, **kwargs)
            else:
                x, y = generate_model_data(*args, **kwargs)

            self.data.append((x.squeeze().unsqueeze(1), y.squeeze().unsqueeze(1)))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.num_samples
    

def generate_model_data(
        amplitude_range=(0.75, 1.25),
        # t_peak_range=(75, 125),
        sigma_range=(10, 50),
        t_0_range=(0, 100),
        p_range=(-9/4, -5/12),
        batch_size=16,
        num_points=100,
        device='cpu',
        dtype=np.float32
    ):
        """
        A simple analytical model for generating TDE-like light curves. This 
        takes the approach of van Velzen (2021) and uses a Gaussian rise and
        power-law decline.
        """
        batch_size = batch_size
        num_points = num_points

        a_min, a_max = amplitude_range
        # b_min, b_max = t_peak_range
        c_min, c_max = sigma_range
        d_min, d_max = t_0_range
        e_min, e_max = p_range

        cxt_x = np.zeros(shape=(batch_size, 1, num_points))
        cxt_y = np.zeros(shape=(batch_size, 1, num_points))

        for i in range(batch_size):
            # Sample random amplitude
            a = (a_max - a_min) * np.random.rand() + a_min
            c = (c_max - c_min) * np.random.rand() + c_min
            b = 3 * c
            d = (d_max - d_min) * np.random.rand() + d_min
            e = (e_max - e_min) * np.random.rand() + e_min
            # Shape (num_points, x_dim)
            x = np.linspace(0, 500, num_points)

            # Shape (num_points, y_dim)
            y = np.zeros(x.shape)
            y[x <= b] = np.exp(-(x[x <= b] - b) ** 2 / (2 * c ** 2))
            y[x > b] = ((x[x > b] - b + d) / d) ** e

            # x = torch.from_numpy(x).float().unsqueeze(1)
            # y = torch.from_numpy(y).float().unsqueeze(1)
            
            cxt_x[i] = x
            cxt_y[i] = y * a

        mx, my = torch.from_numpy(cxt_x.astype(dtype)), torch.from_numpy(cxt_y.astype(dtype))

        return mx.to(device), my.to(device)



def generate_mock_data(
        amplitude_range=(0.9, 1.1),
        x_break_range=(75, 125),
        alpha_1_range=(-5, -1),
        alpha_2_range=(5/12, 9/4),
        delta_range=(0.01, 0.2),
        batch_size=16,
        num_points=100,
        device='cpu',
        dtype=np.float32
    ):
        """
        A more sophisticated model that handles broad peaks. Uses a
        smoothly broken power-law with a power-law rise and power-law
        decline.
        """
        batch_size = batch_size
        num_points = num_points

        # Generate data
        a_min, a_max = amplitude_range
        b_min, b_max = x_break_range
        c_min, c_max = alpha_1_range
        d_min, d_max = alpha_2_range
        e_min, e_max = delta_range

        cxt_x = np.zeros(shape=(batch_size, 1, num_points))
        cxt_y = np.zeros(shape=(batch_size, 1, num_points))

        for i in range(batch_size):
            a = (a_max - a_min) * np.random.rand() + a_min
            b = (b_max - b_min) * np.random.rand() + b_min
            c = (c_max - c_min) * np.random.rand() + c_min
            d = (d_max - d_min) * np.random.rand() + d_min
            e = (e_max - e_min) * np.random.rand() + e_min

            mod = SmoothlyBrokenPowerLaw1D(amplitude=a, x_break=b, alpha_1=c, alpha_2=d, delta=e)

            x = np.linspace(1, 501, num_points)
            y = mod(x)
            
            cxt_x[i] = x
            cxt_y[i] = y

        mx, my = torch.from_numpy(cxt_x.astype(dtype)), torch.from_numpy(cxt_y.astype(dtype))

        return mx.to(device), my.to(device)


def generate_sin_data(
    amplitude_range=(0.0, 2), shift_range=(0.0, 1.0), batch_size=16, num_points=10
):
    a_min, a_max = amplitude_range
    b_min, b_max = shift_range

    cxt_x = np.zeros(shape=(batch_size, 1, num_points))
    cxt_y = np.zeros(shape=(batch_size, 1, num_points))

    for i in range(batch_size):
        a = (a_max - a_min) * np.random.rand() + a_min
        b = (b_max - b_min) * np.random.rand() + b_min
        x = np.linspace(0, 6, num_points)
        cxt_x[i] = x
        cxt_y[i] = a * (np.sin(2 * np.pi / 2 * (x - b)) + 1)

    return torch.from_numpy(cxt_x).float(), torch.from_numpy(cxt_y).float()
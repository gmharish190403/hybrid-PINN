import torch
import torch.nn as nn
import numpy as np
import math

class AdaptiveActivation(nn.Module):
    """
    Layer-wise locally adaptive activation function as described in the PINN paper.
    """
    def __init__(self, n=10):
        super(AdaptiveActivation, self).__init__()
        self.n = n
        self.a = nn.Parameter(torch.ones(1))
        
    def forward(self, x):
        return torch.tanh(self.n * self.a * x)

class ResidualBlock(nn.Module):
    """Residual block with adaptive activations for better gradient flow"""
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.activation1 = AdaptiveActivation()
        self.linear2 = nn.Linear(dim, dim)
        self.activation2 = AdaptiveActivation()

    def forward(self, x):
        identity = x
        out = self.activation1(self.linear1(x))
        out = self.linear2(out)
        out += identity
        out = self.activation2(out)
        return out

class HybridPINN(nn.Module):
    """
    Hybrid Physics-informed neural network for Laser Metal Deposition temperature field prediction.
    Combines physics equations with FEM data for better predictions.
    """
    def __init__(self, domain_bounds, z_fixed=0.1, z_thickness=0.2, hidden_layers=6, neurons_per_layer=128, laser_params=None):
        super(HybridPINN, self).__init__()
        self.domain_bounds = domain_bounds
        self.z_fixed = z_fixed / 100.0  # Midpoint of FEM data in m (default 0.1 cm)
        self.z_thickness = z_thickness / 100.0  # Thickness of FEM data in m (2 mm = 0.2 cm)

        # Physical parameters
        self.T_ambient = 293.15  # Initial/ambient temperature in K
        self.T_liquidus = 1690.0  # Liquidus temperature
        self.T_solidus = 1730.0  # Solidus temperature

        # Input scaling factors for numerical stability
        self.scale_x = 1.0 / max(domain_bounds['x_max'] - domain_bounds['x_min'], 1e-6)
        self.scale_y = 1.0 / max(domain_bounds['y_max'] - domain_bounds['y_min'], 1e-6)
        self.scale_z = 1.0 / max(domain_bounds['z_max'] - domain_bounds['z_min'], 1e-6)
        self.scale_t = 1.0 / max(domain_bounds.get('t_max_deposition', 2.0), 1e-6)
        self.scale_T = 1.0 / 4000.0  # Temperature scaling factor

        # Laser parameters
        if laser_params is None:
            self.laser_params = {
                'power': 2000.0,     # Laser power (W)
                'velocity': 8.0,     # Scanning speed (mm/s)
                'absorption': 0.75,  # Absorption coefficient
                'Ra': 0.3 / 100.0,   # Semi-axis in x (cm to m)
                'Rb': 0.3 / 100.0,   # Semi-axis in y (cm to m)
                'Rc': 0.1 / 100.0,   # Semi-axis in z (cm to m)
                'start_pos': [domain_bounds['x_min'] / 100.0, domain_bounds['y_min'] / 100.0],  # Start pos (cm to m)
                'end_pos': [domain_bounds['x_max'] / 100.0, domain_bounds['y_min'] / 100.0]     # End pos (cm to m)
            }
        else:
            self.laser_params = laser_params

        # Build neural network with residual connections
        layers = []
        layers.append(nn.Linear(4, neurons_per_layer))  # 4 inputs: x, y, z, t
        layers.append(AdaptiveActivation())
        
        # Add residual blocks
        for i in range(hidden_layers):
            layers.append(ResidualBlock(neurons_per_layer))
        
        # Add final output layer
        layers.append(nn.Linear(neurons_per_layer, 1))
        self.net = nn.Sequential(*layers)

        # Initialize weights
        self._initialize_weights()
        self.stage = 'deposition'
        
        # Loss statistics for adaptive weighting
        self.register_buffer('loss_stats', torch.zeros(5))  # [pde, ic, bc, laser, data]

    def _initialize_weights(self):
        """Initialize weights using Xavier/Glorot initialization"""
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _thermal_conductivity(self, T):
        """
        Temperature-dependent thermal conductivity (W/m·K)
        Using improved piecewise linear interpolation for better stability
        """
        # Define temperature points and corresponding conductivity values
        temp_points = torch.tensor([0, 100, 200, 300, 400, 600, 800, 1200, 1300, 1500, 3000], 
                                  device=T.device, dtype=T.dtype)
        k_points = torch.tensor([14.6, 15.1, 16.1, 17.9, 18, 20.8, 23.9, 32.2, 33.7, 120, 140], 
                                device=T.device, dtype=T.dtype)
                                
        # Use a more stable and vectorized approach for interpolation
        k = torch.zeros_like(T)
        for i in range(len(temp_points) - 1):
            mask = (T >= temp_points[i]) & (T < temp_points[i + 1])
            if torch.any(mask):
                slope = (k_points[i + 1] - k_points[i]) / (temp_points[i + 1] - temp_points[i])
                k[mask] = k_points[i] + slope * (T[mask] - temp_points[i])
        
        # Handle values above the highest temperature point
        mask_upper = T >= temp_points[-1]
        if torch.any(mask_upper):
            k[mask_upper] = k_points[-1]
            
        return k
        
    def _density(self, T):
        """
        Temperature-dependent density (kg/m³)
        Using improved piecewise linear interpolation
        """
        # Define temperature points and corresponding density values
        temp_points = torch.tensor([0, 100, 200, 300, 400, 600, 800, 1200, 1300, 1500, 3000], 
                                  device=T.device, dtype=T.dtype)
        rho_points = torch.tensor([7900, 7880, 7830, 7790, 7750, 7660, 7560, 7370, 7320, 7300, 7250], 
                                 device=T.device, dtype=T.dtype)
        
        # Use the same interpolation approach as thermal conductivity
        rho = torch.zeros_like(T)
        for i in range(len(temp_points) - 1):
            mask = (T >= temp_points[i]) & (T < temp_points[i + 1])
            if torch.any(mask):
                slope = (rho_points[i + 1] - rho_points[i]) / (temp_points[i + 1] - temp_points[i])
                rho[mask] = rho_points[i] + slope * (T[mask] - temp_points[i])
        
        # Handle values above the highest temperature point
        mask_upper = T >= temp_points[-1]
        if torch.any(mask_upper):
            rho[mask_upper] = rho_points[-1]
            
        return rho
        
    def _specific_heat(self, T):
        """
        Temperature-dependent specific heat capacity (J/kg·K)
        Using improved piecewise linear interpolation
        """
        # Define temperature points and corresponding specific heat values
        temp_points = torch.tensor([0, 100, 200, 300, 400, 600, 800, 1200, 1300, 1500, 3000], 
                                  device=T.device, dtype=T.dtype)
        cp_points = torch.tensor([462, 496, 512, 525, 540, 577, 604, 676, 692, 700, 720], 
                                device=T.device, dtype=T.dtype)
        
        # Use the same interpolation approach as thermal conductivity
        cp = torch.zeros_like(T)
        for i in range(len(temp_points) - 1):
            mask = (T >= temp_points[i]) & (T < temp_points[i + 1])
            if torch.any(mask):
                slope = (cp_points[i + 1] - cp_points[i]) / (temp_points[i + 1] - temp_points[i])
                cp[mask] = cp_points[i] + slope * (T[mask] - temp_points[i])
        
        # Handle values above the highest temperature point
        mask_upper = T >= temp_points[-1]
        if torch.any(mask_upper):
            cp[mask_upper] = cp_points[-1]
            
        return cp
        
    def _scale_inputs(self, x, y, z, t):
        """Scale inputs to improve numerical stability"""
        x_s = x * self.scale_x
        y_s = y * self.scale_y
        z_s = z * self.scale_z
        t_s = t * self.scale_t
        return x_s, y_s, z_s, t_s
        
    def forward(self, x, y, z, t):
        """
        Forward pass of the PINN model
        with improved dimension handling and scaling
        """
        # Ensure all inputs have the right dimensions
        if x.dim() == 1:
            x = x.unsqueeze(1)
        if y.dim() == 1:
            y = y.unsqueeze(1)
        if z.dim() == 1:
            z = z.unsqueeze(1)
        if t.dim() == 1:
            t = t.unsqueeze(1)
        
        # Get batch size from max dimension
        batch_sizes = [x.shape[0], y.shape[0], z.shape[0], t.shape[0]]
        batch_size = max(batch_sizes)
        
        # Make sure all have the same batch dimension
        if x.shape[0] != batch_size:
            x = x.expand(batch_size, -1)
        if y.shape[0] != batch_size:
            y = y.expand(batch_size, -1)
        if z.shape[0] != batch_size:
            z = z.expand(batch_size, -1)
        if t.shape[0] != batch_size:
            t = t.expand(batch_size, -1)
        
        # Scale inputs
        x_s, y_s, z_s, t_s = self._scale_inputs(x, y, z, t)
        
        # Concatenate inputs
        inputs = torch.cat([x_s, y_s, z_s, t_s], dim=1)
        
        # Pass through network
        output = self.net(inputs)
        
        # Convert to temperature
        # Use tanh + shift to ensure positive temperatures above ambient
        T = self.T_ambient + 3000.0 * torch.tanh(output)
        
        # Ensure temperature is above ambient
        T = torch.clamp(T, min=self.T_ambient)
        
        return T
        
    def heat_source(self, x, y, z, t):
        """
        Enhanced Goldak's double-ellipsoidal heat source model
        with improved stability and formatting
        """
        # Ensure inputs have the right dimensions
        if x.dim() == 1:
            x = x.unsqueeze(1)
        if y.dim() == 1:
            y = y.unsqueeze(1)
        if z.dim() == 1:
            z = z.unsqueeze(1)
        if t.dim() == 1:
            t = t.unsqueeze(1)
            
        # Get batch size from max dimension
        batch_sizes = [x.shape[0], y.shape[0], z.shape[0], t.shape[0]]
        batch_size = max(batch_sizes)
        
        # Make sure all have the same batch dimension
        if x.shape[0] != batch_size:
            x = x.expand(batch_size, -1)
        if y.shape[0] != batch_size:
            y = y.expand(batch_size, -1)
        if z.shape[0] != batch_size:
            z = z.expand(batch_size, -1)
        if t.shape[0] != batch_size:
            t = t.expand(batch_size, -1)
            
        # Extract parameters
        power = self.laser_params['power']
        absorption = self.laser_params['absorption']
        Ra = self.laser_params['Ra']
        Rb = self.laser_params['Rb']
        Rc = self.laser_params['Rc']
        velocity = self.laser_params['velocity'] / 1000.0  # mm/s to m/s

        start_x, start_y = self.laser_params['start_pos']
        end_x, end_y = self.laser_params['end_pos']
        
        # Calculate path length and travel time
        path_length = ((end_x - start_x)**2 + (end_y - start_y)**2)**0.5
        travel_time = path_length / velocity if velocity > 0 else 1.0  # Avoid division by zero

        # Calculate current laser position
        t_clamped = torch.clamp(t, min=0.0, max=travel_time)
        fraction = t_clamped / travel_time
        current_x = start_x + (end_x - start_x) * fraction
        current_y = start_y + (end_y - start_y) * fraction
        current_z = torch.zeros_like(t)  # Laser is at z=0

        # Calculate distance from each point to laser center
        r_squared = ((x - current_x)**2 / Ra**2 + 
                     (y - current_y)**2 / Rb**2 + 
                     (z - current_z)**2 / Rc**2)

        # Calculate heat source based on Goldak's model
        prefactor = (6.0 * np.sqrt(3) * absorption * power) / (np.pi * np.sqrt(np.pi) * Ra * Rb * Rc)
        Q = prefactor * torch.exp(-3.0 * r_squared)

        # Zero out heat source after deposition or in cooling stage
        Q = torch.where((t > travel_time) | (self.stage == 'cooling'), 
                        torch.zeros_like(Q), Q)

        return Q
        
    def compute_derivatives(self, x, y, z, t):
        """
        Compute spatial and temporal derivatives
        with improved numerical stability
        """
        # Ensure inputs have the right dimensions
        if x.dim() == 1:
            x = x.unsqueeze(1)
        if y.dim() == 1:
            y = y.unsqueeze(1)
        if z.dim() == 1:
            z = z.unsqueeze(1)
        if t.dim() == 1:
            t = t.unsqueeze(1)
            
        # Clone inputs and set requires_grad
        x = x.clone().requires_grad_(True)
        y = y.clone().requires_grad_(True)
        z = z.clone().requires_grad_(True)
        t = t.clone().requires_grad_(True)

        # Get temperature prediction
        T = self.forward(x, y, z, t)

        # Create grad outputs (ones tensor for autograd)
        grad_T = torch.ones_like(T, requires_grad=True)

        # Compute gradients using autograd
        dT_dx = torch.autograd.grad(T, x, grad_outputs=grad_T, create_graph=True, retain_graph=True)[0]
        dT_dy = torch.autograd.grad(T, y, grad_outputs=grad_T, create_graph=True, retain_graph=True)[0]
        dT_dz = torch.autograd.grad(T, z, grad_outputs=grad_T, create_graph=True, retain_graph=True)[0]
        dT_dt = torch.autograd.grad(T, t, grad_outputs=grad_T, create_graph=True, retain_graph=True)[0]
        
        # Apply gradient clipping for better stability
        dT_dx = torch.clamp(dT_dx, min=-1e5, max=1e5)
        dT_dy = torch.clamp(dT_dy, min=-1e5, max=1e5)
        dT_dz = torch.clamp(dT_dz, min=-1e5, max=1e5)
        dT_dt = torch.clamp(dT_dt, min=-1e5, max=1e5)

        return {'T': T, 'dT_dx': dT_dx, 'dT_dy': dT_dy, 'dT_dz': dT_dz, 'dT_dt': dT_dt}
        
    def compute_pde_residual(self, x, y, z, t):
        """
        Compute PDE residual for heat conduction equation with
        improved numerical stability
        """
        # Ensure inputs have the right dimensions
        if x.dim() == 1:
            x = x.unsqueeze(1)
        if y.dim() == 1:
            y = y.unsqueeze(1)
        if z.dim() == 1:
            z = z.unsqueeze(1)
        if t.dim() == 1:
            t = t.unsqueeze(1)
            
        # Clone inputs and set requires_grad
        x = x.clone().requires_grad_(True)
        y = y.clone().requires_grad_(True)
        z = z.clone().requires_grad_(True)
        t = t.clone().requires_grad_(True)

        # Get temperature prediction
        T = self.forward(x, y, z, t)

        # Get material properties at current temperature
        k = self._thermal_conductivity(T)
        rho = self._density(T)
        cp = self._specific_heat(T)

        # Compute first derivatives
        grad_T = torch.ones_like(T, requires_grad=True)
        dT_dx = torch.autograd.grad(T, x, grad_outputs=grad_T, create_graph=True, retain_graph=True)[0]
        dT_dy = torch.autograd.grad(T, y, grad_outputs=grad_T, create_graph=True, retain_graph=True)[0]
        dT_dz = torch.autograd.grad(T, z, grad_outputs=grad_T, create_graph=True, retain_graph=True)[0]
        dT_dt = torch.autograd.grad(T, t, grad_outputs=grad_T, create_graph=True, retain_graph=True)[0]
        
        # Apply gradient clipping for better stability
        dT_dx = torch.clamp(dT_dx, min=-1e5, max=1e5)
        dT_dy = torch.clamp(dT_dy, min=-1e5, max=1e5)
        dT_dz = torch.clamp(dT_dz, min=-1e5, max=1e5)
        dT_dt = torch.clamp(dT_dt, min=-1e5, max=1e5)

        # Compute second derivatives
        ones_dx = torch.ones_like(dT_dx, requires_grad=True)
        ones_dy = torch.ones_like(dT_dy, requires_grad=True)
        ones_dz = torch.ones_like(dT_dz, requires_grad=True)
        
        d2T_dx2 = torch.autograd.grad(dT_dx, x, grad_outputs=ones_dx, 
                                      create_graph=True, retain_graph=True)[0]
        d2T_dy2 = torch.autograd.grad(dT_dy, y, grad_outputs=ones_dy, 
                                      create_graph=True, retain_graph=True)[0]
        d2T_dz2 = torch.autograd.grad(dT_dz, z, grad_outputs=ones_dz, 
                                      create_graph=True, retain_graph=True)[0]

        # Apply gradient clipping for better stability
        d2T_dx2 = torch.clamp(d2T_dx2, min=-1e3, max=1e3)
        d2T_dy2 = torch.clamp(d2T_dy2, min=-1e3, max=1e3)
        d2T_dz2 = torch.clamp(d2T_dz2, min=-1e3, max=1e3)

        # Compute Laplacian
        laplacian = d2T_dx2 + d2T_dy2 + d2T_dz2
        
        # Use a simpler approach for dkdT to avoid additional autograd
        dkdT = 0.05 * torch.ones_like(T)  # Approximate average derivative
        
        # Calculate additional term for temperature-dependent conductivity
        grad_T_squared = dT_dx.pow(2) + dT_dy.pow(2) + dT_dz.pow(2)
        
        # Calculate diffusion term
        diffusion_term = k * laplacian + dkdT * grad_T_squared
        
        # Calculate transient term
        transient_term = rho * cp * dT_dt
        
        # Calculate heat source term
        heat_source_term = self.heat_source(x, y, z, t) if self.stage == 'deposition' else torch.zeros_like(T)
        
        # The residual of the PDE is:
        # ρCp * ∂T/∂t - div(k∇T) - Q = 0
        residual = transient_term - diffusion_term - heat_source_term
        
        # Normalize residual for better training
        scale_factor = torch.mean(torch.abs(transient_term.detach())) + 1e-8
        normalized_residual = residual / scale_factor
        
        return normalized_residual, T
        
    def compute_data_residual(self, x_fem, y_fem, t_fem, T_fem):
        """
        Compute residual for FEM data, handling 2D to 3D mapping with thickness.
        This is key for the hybrid approach, integrating 2D FEM data with 3D physics.
        
        Args:
            x_fem: x-coordinates (cm)
            y_fem: y-coordinates (cm)
            t_fem: time points (s)
            T_fem: FEM temperatures (°C)
        
        Returns:
            normalized_residual: Scaled residual for loss function
            T_pred_avg: Predicted temperatures
        """
        # Ensure inputs have the right dimensions
        if x_fem.dim() == 1:
            x_fem = x_fem.unsqueeze(1)
        if y_fem.dim() == 1:
            y_fem = y_fem.unsqueeze(1)
        if t_fem.dim() == 1:
            t_fem = t_fem.unsqueeze(1)
        if T_fem.dim() == 1:
            T_fem = T_fem.unsqueeze(1)
            
        # Convert FEM coordinates from cm to m and temp from °C to K
        x = x_fem / 100.0
        y = y_fem / 100.0
        T_fem = T_fem + 273.15  # Convert °C to K

        # Get batch size
        batch_size = x.shape[0]
        
        # Create z values for averaging across the thickness
        z_min = self.z_fixed - self.z_thickness / 2.0
        z_max = self.z_fixed + self.z_thickness / 2.0
        
        # Use 5 z-values for better integration across thickness
        num_z_points = 5
        z_values = torch.linspace(z_min, z_max, num_z_points, device=x.device)
        
        # Compute Gaussian weights for integration (more weight in the middle)
        z_central = (z_values - self.z_fixed) / (self.z_thickness / 2.0)
        weights = torch.exp(-2.0 * z_central.pow(2))
        weights = weights / weights.sum()  # Normalize weights
        
        # Initialize weighted average temperature
        T_pred_avg = torch.zeros_like(T_fem)
        
        # Predict temperature at each z level and compute weighted average
        for i, z_val in enumerate(z_values):
            # Create z tensor of correct shape
            z = torch.ones(batch_size, 1, device=x.device) * z_val
            
            # Get prediction at this z-level
            T_pred = self.forward(x, y, z, t_fem)
            
            # Add weighted contribution
            T_pred_avg += weights[i] * T_pred
        
        # Compute normalized residual (relative error)
        residual = (T_pred_avg - T_fem) / torch.maximum(torch.abs(T_fem), 
                                                       torch.ones_like(T_fem) * 300.0)
        
        return residual, T_pred_avg
        
    def compute_ic_residual(self, x, y, z, t):
        """
        Compute initial condition residual.
        For deposition stage, this enforces ambient temperature at t=0.
        For cooling stage, this enforces the final temperature from deposition at t=0.
        
        Args:
            x, y, z, t: Coordinates and time
            
        Returns:
            normalized_residual: Scaled residual for loss function
            T: Temperature at the initial points
        """
        # Ensure inputs have the right dimensions
        if x.dim() == 1:
            x = x.unsqueeze(1)
        if y.dim() == 1:
            y = y.unsqueeze(1)
        if z.dim() == 1:
            z = z.unsqueeze(1)
        if t.dim() == 1:
            t = t.unsqueeze(1)
            
        # Get temperature prediction
        T = self.forward(x, y, z, t)

        # For deposition stage, initial condition is ambient temperature
        if self.stage == 'deposition':
            ic_residual = T - self.T_ambient
        else:
            # For cooling stage, we would use the final temperature from deposition
            # If we have a deposition model, we should use its predictions
            # But for now, we'll just set it to zero as a placeholder
            ic_residual = torch.zeros_like(T)

        # Normalize residual
        ic_residual = ic_residual / (self.T_liquidus - self.T_ambient + 1e-8)

        return ic_residual, T
        
    def compute_bc_residual(self, x, y, z, t, boundary_type):
        """
        Compute boundary condition residual with improved stability.
        Handles convection and radiation boundary conditions.
        
        Args:
            x, y, z, t: Coordinates and time
            boundary_type: Type of boundary ('top', 'bottom', 'left', 'right', 'front', 'back')
            
        Returns:
            normalized_residual: Scaled residual for loss function
            T: Temperature at the boundary points
        """
        # Ensure inputs have the right dimensions
        if x.dim() == 1:
            x = x.unsqueeze(1)
        if y.dim() == 1:
            y = y.unsqueeze(1)
        if z.dim() == 1:
            z = z.unsqueeze(1)
        if t.dim() == 1:
            t = t.unsqueeze(1)
            
        # Clone inputs and set requires_grad for gradient calculation
        x = x.clone().requires_grad_(True)
        y = y.clone().requires_grad_(True)
        z = z.clone().requires_grad_(True)

        # Get temperature prediction
        T = self.forward(x, y, z, t)
        grad_T = torch.ones_like(T, requires_grad=True)

        # Calculate gradient based on boundary type (normal derivative)
        if boundary_type == 'top':  # z = z_max
            dT_dn = torch.autograd.grad(T, z, grad_outputs=grad_T, create_graph=True)[0]
        elif boundary_type == 'bottom':  # z = z_min
            dT_dn = -torch.autograd.grad(T, z, grad_outputs=grad_T, create_graph=True)[0]
        elif boundary_type == 'left':  # x = x_min
            dT_dn = -torch.autograd.grad(T, x, grad_outputs=grad_T, create_graph=True)[0]
        elif boundary_type == 'right':  # x = x_max
            dT_dn = torch.autograd.grad(T, x, grad_outputs=grad_T, create_graph=True)[0]
        elif boundary_type == 'front':  # y = y_min
            dT_dn = -torch.autograd.grad(T, y, grad_outputs=grad_T, create_graph=True)[0]
        elif boundary_type == 'back':  # y = y_max
            dT_dn = torch.autograd.grad(T, y, grad_outputs=grad_T, create_graph=True)[0]
        else:
            raise ValueError(f"Unknown boundary type: {boundary_type}")

        # Get thermal conductivity at the current temperature
        k = self._thermal_conductivity(T)
        
        # Heat transfer coefficients
        h_c = 20.0  # Convection coefficient (W/m²K)
        epsilon = 0.85  # Emissivity
        sigma = 5.67e-8  # Stefan-Boltzmann constant

        # Clamp temperature for stability in radiation calculation
        T_clamped = torch.clamp(T, max=3000.0)
        
        # Calculate heat fluxes
        q_c = h_c * (T - self.T_ambient)  # Convection
        q_r = epsilon * sigma * (T_clamped**4 - self.T_ambient**4)  # Radiation

        # The boundary condition is:
        # -k * ∂T/∂n = h_c * (T - T_ambient) + ε * σ * (T^4 - T_ambient^4)
        bc_residual = k * dT_dn
        # The boundary condition is:
        # -k * ∂T/∂n = h_c * (T - T_ambient) + ε * σ * (T^4 - T_ambient^4)
        bc_residual = k * dT_dn + q_c + q_r
        
        # Normalize residual
        scale_factor = h_c * (self.T_liquidus - self.T_ambient) + epsilon * sigma * (self.T_liquidus**4 - self.T_ambient**4) + 1e-8
        normalized_residual = bc_residual / scale_factor

        return normalized_residual, T    
    
    def set_stage(self, stage):
        """
        Set the simulation stage (deposition or cooling)
        
        Args:
            stage: String, either 'deposition' or 'cooling'
        
        Returns:
            self: For method chaining
        """
        assert stage in ['deposition', 'cooling'], f"Invalid stage: {stage}"
        self.stage = stage
        return self

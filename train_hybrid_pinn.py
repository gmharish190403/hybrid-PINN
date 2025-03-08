import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import qmc

def load_preprocess_fem_data(file_path, plot_heatmap=True):
    """
    Load and preprocess 2D FEM data for PINN training.
    
    Args:
        file_path: Path to the FEM data file (CSV or Excel)
        plot_heatmap: Whether to plot a temperature heatmap
        
    Returns:
        processed_data: Numpy array with [t, x, y, z, T] columns
        domain_info: Dictionary with data statistics
    """
    print(f"Loading FEM data from {file_path}...")
    
    # Load data based on file extension
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")
    
    # Display basic information
    print(f"FEM data shape: {df.shape}")
    print("Columns:", df.columns.tolist())
    
    # Check expected columns (adjust column names as needed)
    required_cols = ['Time_s', 'Temperature_°C', 'x_cm', 'y_cm']
    actual_cols = list(df.columns)
    
    # Map actual column names to required names if needed
    col_mapping = {}
    for req_col in required_cols:
        for actual_col in actual_cols:
            if req_col.lower() in actual_col.lower():
                col_mapping[actual_col] = req_col
                break
    
    if col_mapping:
        print("Using column mapping:", col_mapping)
        df = df.rename(columns=col_mapping)
    
    # Check if we have all required columns
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in FEM data: {missing_cols}")
    
    # Extract data
    times = df['Time_s'].unique()
    x_values = df['x_cm'].unique()
    y_values = df['y_cm'].unique()
    
    print(f"Time steps: {len(times)}")
    print(f"X coordinates: {len(x_values)}")
    print(f"Y coordinates: {len(y_values)}")
    print(f"Temperature range: {df['Temperature_°C'].min():.1f}°C - {df['Temperature_°C'].max():.1f}°C")
    
    # Normalize temperature data for better training
    T_mean = df['Temperature_°C'].mean()
    T_std = df['Temperature_°C'].std() 
    
    # Store original temperature for denormalization
    df['Original_Temperature'] = df['Temperature_°C'].copy()
    
    # Create domain info dictionary
    domain_info = {
        'x_min': df['x_cm'].min(),
        'x_max': df['x_cm'].max(),
        'y_min': df['y_cm'].min(),
        'y_max': df['y_cm'].max(),
        'z_min': 0.0,  # Default z range
        'z_max': 0.2,  # Default z range (2 mm thickness)
        't_min': df['Time_s'].min(),
        't_max_deposition': df['Time_s'].max(),
        'T_min': df['Temperature_°C'].min(),
        'T_max': df['Temperature_°C'].max(),
        'T_mean': T_mean,
        'T_std': T_std
    }
    
    # Add z column at fixed z=0.1 cm (midpoint of thickness)
    z_fixed = 0.1  # cm
    processed_data = np.column_stack([
        df['Time_s'].values,          # t
        df['x_cm'].values,            # x
        df['y_cm'].values,            # y
        np.full(len(df), z_fixed),    # z
        df['Temperature_°C'].values   # T (still using original for now)
    ])
    
    # Plot heatmap of temperature at first time step
    if plot_heatmap:
        try:
            # Find data at the first time step
            t0 = times[0]
            df_t0 = df[df['Time_s'] == t0]
            
            plt.figure(figsize=(10, 8))
            
            # Check if we can create a 2D heatmap or need a line plot
            if len(y_values) <= 1:  # Single y value - create line plot
                plt.plot(df_t0['x_cm'], df_t0['Temperature_°C'], 'b-')
                plt.xlabel('X (cm)')
                plt.ylabel('Temperature (°C)')
                plt.title(f'Temperature at t={t0}s')
                plt.grid(True, alpha=0.3)
            else:  # Multiple y values - create heatmap
                # Create a pivot table for the heatmap
                pivot = df_t0.pivot_table(values='Temperature_°C', 
                                          index='y_cm', 
                                          columns='x_cm')
                plt.contourf(pivot.columns, pivot.index, pivot.values, 50, cmap='hot')
                plt.colorbar(label='Temperature (°C)')
                plt.xlabel('X (cm)')
                plt.ylabel('Y (cm)')
                plt.title(f'Temperature Field at t={t0}s')
            
            plt.show()
        except Exception as e:
            print(f"Warning: Could not plot heatmap: {e}")
    
    return processed_data, domain_info


def generate_training_points(domain_bounds, n_interior=10000, n_boundary=2000, n_initial=2000, stage='deposition'):
    """
    Generate training points for physics-informed constraints.
    
    Args:
        domain_bounds: Dictionary with domain bounds
        n_interior: Number of interior points for PDE
        n_boundary: Number of boundary points
        n_initial: Number of initial points
        stage: 'deposition' or 'cooling'
        
    Returns:
        points: Dictionary with interior, boundary, and initial points
    """
    print(f"Generating sampling points for {stage} stage...")
    
    # Extract domain bounds
    x_min, x_max = domain_bounds['x_min'], domain_bounds['x_max']
    y_min, y_max = domain_bounds['y_min'], domain_bounds['y_max']
    z_min, z_max = domain_bounds['z_min'], domain_bounds['z_max']
    
    if stage == 'deposition':
        t_max = domain_bounds['t_max_deposition']
    else:
        t_max = domain_bounds.get('t_max_cooling', 100.0)
    
    # Generate interior points using Latin Hypercube Sampling for better coverage
    sampler = qmc.LatinHypercube(d=4)
    sample = sampler.random(n=n_interior)
    
    x_interior = torch.from_numpy((sample[:, 0] * (x_max - x_min) + x_min).astype(np.float32)).reshape(-1, 1)
    y_interior = torch.from_numpy((sample[:, 1] * (y_max - y_min) + y_min).astype(np.float32)).reshape(-1, 1)
    z_interior = torch.from_numpy((sample[:, 2] * (z_max - z_min) + z_min).astype(np.float32)).reshape(-1, 1)
    t_interior = torch.from_numpy((sample[:, 3] * t_max).astype(np.float32)).reshape(-1, 1)
    
    interior_points = torch.cat([x_interior, y_interior, z_interior, t_interior], dim=1)
    
    # Generate boundary points - use more points near the laser path
    boundary_points = {}
    
    # Bottom boundary (z = z_min)
    sampler = qmc.LatinHypercube(d=3)
    sample = sampler.random(n=n_boundary // 6)
    
    x_bottom = torch.from_numpy((sample[:, 0] * (x_max - x_min) + x_min).astype(np.float32)).reshape(-1, 1)
    y_bottom = torch.from_numpy((sample[:, 1] * (y_max - y_min) + y_min).astype(np.float32)).reshape(-1, 1)
    z_bottom = torch.ones_like(x_bottom) * z_min
    t_bottom = torch.from_numpy((sample[:, 2] * t_max).astype(np.float32)).reshape(-1, 1)
    
    boundary_points['bottom'] = torch.cat([x_bottom, y_bottom, z_bottom, t_bottom], dim=1)
    
    # Top boundary (z = z_max) - add extra points where the laser hits
    sampler = qmc.LatinHypercube(d=3)
    sample = sampler.random(n=n_boundary // 6)
    
    x_top = torch.from_numpy((sample[:, 0] * (x_max - x_min) + x_min).astype(np.float32)).reshape(-1, 1)
    y_top = torch.from_numpy((sample[:, 1] * (y_max - y_min) + y_min).astype(np.float32)).reshape(-1, 1)
    z_top = torch.ones_like(x_top) * z_max
    t_top = torch.from_numpy((sample[:, 2] * t_max).astype(np.float32)).reshape(-1, 1)
    
    # Add extra points at top where laser is applied
    n_extra = n_boundary // 12
    x_extra = torch.linspace(x_min, x_max, n_extra).reshape(-1, 1)
    y_extra = torch.ones(n_extra, 1) * (y_min + y_max) / 2
    z_extra = torch.ones(n_extra, 1) * z_max
    t_extra = torch.linspace(0, t_max, n_extra).reshape(-1, 1)
    
    top_points = torch.cat([x_top, y_top, z_top, t_top], dim=1)
    extra_points = torch.cat([x_extra, y_extra, z_extra, t_extra], dim=1)
    boundary_points['top'] = torch.cat([top_points, extra_points], dim=0)
    
    # Generate points for other boundaries similarly
    # Left boundary (x = x_min)
    sampler = qmc.LatinHypercube(d=3)
    sample = sampler.random(n=n_boundary // 6)
    
    x_left = torch.ones((n_boundary // 6, 1)) * x_min
    y_left = torch.from_numpy((sample[:, 0] * (y_max - y_min) + y_min).astype(np.float32)).reshape(-1, 1)
    z_left = torch.from_numpy((sample[:, 1] * (z_max - z_min) + z_min).astype(np.float32)).reshape(-1, 1)
    t_left = torch.from_numpy((sample[:, 2] * t_max).astype(np.float32)).reshape(-1, 1)
    
    boundary_points['left'] = torch.cat([x_left, y_left, z_left, t_left], dim=1)
    
    # Right boundary (x = x_max)
    x_right = torch.ones((n_boundary // 6, 1)) * x_max
    y_right = torch.from_numpy((sample[:, 0] * (y_max - y_min) + y_min).astype(np.float32)).reshape(-1, 1)
    z_right = torch.from_numpy((sample[:, 1] * (z_max - z_min) + z_min).astype(np.float32)).reshape(-1, 1)
    t_right = torch.from_numpy((sample[:, 2] * t_max).astype(np.float32)).reshape(-1, 1)
    
    boundary_points['right'] = torch.cat([x_right, y_right, z_right, t_right], dim=1)
    
    # Front boundary (y = y_min)
    x_front = torch.from_numpy((sample[:, 0] * (x_max - x_min) + x_min).astype(np.float32)).reshape(-1, 1)
    y_front = torch.ones((n_boundary // 6, 1)) * y_min
    z_front = torch.from_numpy((sample[:, 1] * (z_max - z_min) + z_min).astype(np.float32)).reshape(-1, 1)
    t_front = torch.from_numpy((sample[:, 2] * t_max).astype(np.float32)).reshape(-1, 1)
    
    boundary_points['front'] = torch.cat([x_front, y_front, z_front, t_front], dim=1)
    
    # Back boundary (y = y_max)
    x_back = torch.from_numpy((sample[:, 0] * (x_max - x_min) + x_min).astype(np.float32)).reshape(-1, 1)
    y_back = torch.ones((n_boundary // 6, 1)) * y_max
    z_back = torch.from_numpy((sample[:, 1] * (z_max - z_min) + z_min).astype(np.float32)).reshape(-1, 1)
    t_back = torch.from_numpy((sample[:, 2] * t_max).astype(np.float32)).reshape(-1, 1)
    
    boundary_points['back'] = torch.cat([x_back, y_back, z_back, t_back], dim=1)
    
    # Generate initial condition points (t = 0)
    sampler = qmc.LatinHypercube(d=3)
    sample = sampler.random(n=n_initial)
    
    x_initial = torch.from_numpy((sample[:, 0] * (x_max - x_min) + x_min).astype(np.float32)).reshape(-1, 1)
    y_initial = torch.from_numpy((sample[:, 1] * (y_max - y_min) + y_min).astype(np.float32)).reshape(-1, 1)
    z_initial = torch.from_numpy((sample[:, 2] * (z_max - z_min) + z_min).astype(np.float32)).reshape(-1, 1)
    t_initial = torch.zeros((n_initial, 1))
    
    initial_points = torch.cat([x_initial, y_initial, z_initial, t_initial], dim=1)
    
    print(f"Generated {len(interior_points)} interior points, {sum(len(v) for v in boundary_points.values())} boundary points, {len(initial_points)} initial points")
    
    return {
        'interior': interior_points,
        'boundary': boundary_points,
        'initial': initial_points
    }


def train_hybrid_pinn(
    model, 
    domain_bounds, 
    fem_data=None,
    epochs=200, 
    batch_size=512, 
    learning_rate=1e-3,
    lambda_pde=1.0,
    lambda_ic=0.1,
    lambda_bc=0.1,
    lambda_data=10.0,
    save_path="./results",
    use_physics=True,
    use_scheduler=True,
    patience=20
):
    """
    Train the hybrid PINN model using both physics constraints and FEM data.
    
    Args:
        model: HybridPINN model
        domain_bounds: Dictionary with domain bounds
        fem_data: Numpy array with [t, x, y, z, T] columns
        epochs: Number of training epochs
        batch_size: Batch size for FEM data
        learning_rate: Learning rate
        lambda_pde: Weight for PDE residual
        lambda_ic: Weight for initial condition residual
        lambda_bc: Weight for boundary condition residual
        lambda_data: Weight for data residual
        save_path: Path to save results
        use_physics: Whether to use physics constraints
        use_scheduler: Whether to use learning rate scheduler
        patience: Number of epochs for early stopping
        
    Returns:
        model: Trained model
        history: Training history
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create save directory
    os.makedirs(save_path, exist_ok=True)
    
    # Move model to device
    model = model.to(device)
    
    # Save the original lambda values
    lambda_pde_save = lambda_pde
    lambda_ic_save = lambda_ic
    lambda_bc_save = lambda_bc
    lambda_data_save = lambda_data
    
    # Generate physics-informed sampling points if using physics
    interior_points = None
    boundary_points = None
    initial_points = None
    
    if use_physics:
        points = generate_training_points(
            domain_bounds, 
            n_interior=20000, 
            n_boundary=4000, 
            n_initial=2000, 
            stage=model.stage
        )
        
        # Move points to device
        interior_points = points['interior'].to(device)
        boundary_points = {k: v.to(device) for k, v in points['boundary'].items()}
        initial_points = points['initial'].to(device)
    
    # Prepare FEM data if provided
    fem_dataset = None
    if fem_data is not None:
        print(f"Preparing FEM data with shape: {fem_data.shape}")
        # Convert to tensors
        fem_t = torch.tensor(fem_data[:, 0], dtype=torch.float32, requires_grad=True).reshape(-1, 1)  # t
        fem_x = torch.tensor(fem_data[:, 1], dtype=torch.float32, requires_grad=True).reshape(-1, 1)  # x
        fem_y = torch.tensor(fem_data[:, 2], dtype=torch.float32, requires_grad=True).reshape(-1, 1)  # y
        fem_z = torch.tensor(fem_data[:, 3], dtype=torch.float32, requires_grad=True).reshape(-1, 1)  # z
        fem_T = torch.tensor(fem_data[:, 4], dtype=torch.float32, requires_grad=True).reshape(-1, 1)  # T
        
        # Create dataset
        fem_dataset = torch.cat([fem_t, fem_x, fem_y, fem_z, fem_T], dim=1).to(device)
        print(f"FEM dataset prepared with {len(fem_dataset)} points")
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Create scheduler
    scheduler = None
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=patience//2, verbose=True,
            threshold=1e-4, threshold_mode='rel', cooldown=0, min_lr=1e-6
        )
    
    # Initialize history
    history = {
        'epoch': [],
        'loss': [],
        'pde_loss': [],
        'ic_loss': [],
        'bc_loss': [],
        'data_loss': [],
        'lr': []
    }
    
    best_loss = float('inf')
    no_improve_count = 0
    
    # Progressive training
    print("Starting progressive training strategy...")
    
    # Phase 1: Data-only training (a few epochs)
    if use_physics:
        print("Phase 1: Data-only training...")
        lambda_pde = 0.0
        lambda_ic = 0.0
        lambda_bc = 0.0
    
        for epoch in range(min(20, epochs // 5)):
            model.train()
            
            # Initialize losses
            pde_loss = torch.tensor(0.0, device=device)
            ic_loss = torch.tensor(0.0, device=device)
            bc_loss = torch.tensor(0.0, device=device)
            data_loss = torch.tensor(0.0, device=device)
            
            # Compute data loss if FEM data is provided
            if fem_dataset is not None:
                optimizer.zero_grad()
                
                # Shuffle data
                indices = torch.randperm(fem_dataset.size(0))
                fem_dataset_shuffled = fem_dataset[indices]
                
                # Process in batches
                all_data_residuals = []
                
                for i in range(0, fem_dataset.size(0), batch_size):
                    end = min(i + batch_size, fem_dataset.size(0))
                    batch = fem_dataset_shuffled[i:end]
                    
                    fem_t_batch = batch[:, 0:1]
                    fem_x_batch = batch[:, 1:2]
                    fem_y_batch = batch[:, 2:3]
                    fem_z_batch = batch[:, 3:4]
                    fem_T_batch = batch[:, 4:5]
                    
                    # Compute data residual
                    data_residual, _ = model.compute_data_residual(
                        fem_x_batch, fem_y_batch, fem_t_batch, fem_T_batch
                    )
                    all_data_residuals.append(data_residual)
                
                # Combine all residuals
                if all_data_residuals:
                    all_residuals = torch.cat(all_data_residuals, dim=0)
                    data_loss = torch.mean(all_residuals**2)
                else:
                    data_loss = torch.tensor(0.0, device=device)
            
            # Compute total loss
            total_loss = lambda_data * data_loss
            
            # Backpropagation
            optimizer.zero_grad()
            total_loss.backward()
            
            # Clip gradients for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # Update weights
            optimizer.step()
            
            # Update learning rate
            if scheduler is not None:
                scheduler.step(total_loss)
            
            # Track current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            
            # Record history
            history['epoch'].append(epoch)
            history['loss'].append(total_loss.item())
            history['pde_loss'].append(pde_loss.item())
            history['ic_loss'].append(ic_loss.item())
            history['bc_loss'].append(bc_loss.item())
            history['data_loss'].append(data_loss.item())
            history['lr'].append(current_lr)
            
            # Print progress
            if epoch % 5 == 0 or epoch == min(20, epochs // 5) - 1:
                print(f"Phase 1 - Epoch {epoch}/{min(20, epochs // 5)} - "
                      f"Loss: {total_loss.item():.6f}, "
                      f"Data: {data_loss.item():.6f}, "
                      f"LR: {current_lr:.6f}")
    
        # Phase 2: Gradually introduce physics constraints
        print("Phase 2: Introducing physics constraints...")
        lambda_pde = lambda_pde_save * 0.1
        lambda_ic = lambda_ic_save * 0.1
        lambda_bc = lambda_bc_save * 0.1

        # We need to ensure the model parameters require grad before proceeding
        for param in model.parameters():
            param.requires_grad = True

        for epoch in range(min(20, epochs // 5)):
            model.train()
            
            # Initialize losses
            pde_loss = torch.tensor(0.0, device=device, requires_grad=True)
            ic_loss = torch.tensor(0.0, device=device, requires_grad=True)
            bc_loss = torch.tensor(0.0, device=device, requires_grad=True)
            data_loss = torch.tensor(0.0, device=device, requires_grad=True)
            
            # Compute physics losses
            try:
                # PDE residual
                x_pde = interior_points[:, 0:1].clone().requires_grad_(True)
                y_pde = interior_points[:, 1:2].clone().requires_grad_(True)
                z_pde = interior_points[:, 2:3].clone().requires_grad_(True)
                t_pde = interior_points[:, 3:4].clone().requires_grad_(True)
                
                pde_residual, _ = model.compute_pde_residual(x_pde, y_pde, z_pde, t_pde)
                pde_loss = torch.mean(pde_residual**2)
                
                # Initial condition residual
                x_ic = initial_points[:, 0:1].clone().requires_grad_(True)
                y_ic = initial_points[:, 1:2].clone().requires_grad_(True)
                z_ic = initial_points[:, 2:3].clone().requires_grad_(True)
                t_ic = initial_points[:, 3:4].clone().requires_grad_(True)
                
                ic_residual, _ = model.compute_ic_residual(x_ic, y_ic, z_ic, t_ic)
                ic_loss = torch.mean(ic_residual**2)
                
                # Boundary condition residual
                bc_loss_sum = torch.tensor(0.0, device=device, requires_grad=True)
                bc_count = 0
                
                for boundary_type, boundary_points_tensor in boundary_points.items():
                    x_bc = boundary_points_tensor[:, 0:1].clone().requires_grad_(True)
                    y_bc = boundary_points_tensor[:, 1:2].clone().requires_grad_(True)
                    z_bc = boundary_points_tensor[:, 2:3].clone().requires_grad_(True)
                    t_bc = boundary_points_tensor[:, 3:4].clone().requires_grad_(True)
                    
                    try:
                        bc_residual, _ = model.compute_bc_residual(x_bc, y_bc, z_bc, t_bc, boundary_type)
                        bc_loss_sum = bc_loss_sum + torch.mean(bc_residual**2)
                        bc_count += 1
                    except Exception as e:
                        # Skip if BC computation fails for this boundary
                        print(f"Warning: BC computation failed for {boundary_type}: {e}")
                
                if bc_count > 0:
                    bc_loss = bc_loss_sum / bc_count
            except Exception as e:
                print(f"Warning in computing physics losses: {e}")
                # Continue without physics losses if they fail
                pde_loss = torch.tensor(0.0, device=device, requires_grad=True)
                ic_loss = torch.tensor(0.0, device=device, requires_grad=True)
                bc_loss = torch.tensor(0.0, device=device, requires_grad=True)
                    
            # Compute data loss if FEM data is provided
            if fem_dataset is not None:
                # Shuffle data
                indices = torch.randperm(fem_dataset.size(0))
                fem_dataset_shuffled = fem_dataset[indices]
                
                # Process in batches
                all_data_residuals = []
                
                for i in range(0, fem_dataset.size(0), batch_size):
                    end = min(i + batch_size, fem_dataset.size(0))
                    batch = fem_dataset_shuffled[i:end]
                    
                    fem_t_batch = batch[:, 0:1]
                    fem_x_batch = batch[:, 1:2]
                    fem_y_batch = batch[:, 2:3]
                    fem_z_batch = batch[:, 3:4]
                    fem_T_batch = batch[:, 4:5]
                    
                    # Compute data residual
                    data_residual, _ = model.compute_data_residual(
                        fem_x_batch, fem_y_batch, fem_t_batch, fem_T_batch
                    )
                    all_data_residuals.append(data_residual)
                
                # Combine all residuals
                if all_data_residuals:
                    all_residuals = torch.cat(all_data_residuals, dim=0)
                    data_loss = torch.mean(all_residuals**2)
                else:
                    data_loss = torch.tensor(0.0, device=device, requires_grad=True)
            
            # Compute total loss
            total_loss = (
                lambda_pde * pde_loss +
                lambda_ic * ic_loss +
                lambda_bc * bc_loss +
                lambda_data * data_loss
            )
            
            # Backpropagation
            optimizer.zero_grad()
            total_loss.backward()
            
            # Clip gradients for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # Update weights
            # Update weights
            optimizer.step()
            
            # Update learning rate
            if scheduler is not None:
                scheduler.step(total_loss)
            
            # Track current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            
            # Record history
            history['epoch'].append(epoch + min(20, epochs // 5))
            history['loss'].append(total_loss.item())
            history['pde_loss'].append(pde_loss.item())
            history['ic_loss'].append(ic_loss.item())
            history['bc_loss'].append(bc_loss.item())
            history['data_loss'].append(data_loss.item())
            history['lr'].append(current_lr)
            
            # Print progress
            if epoch % 5 == 0 or epoch == min(20, epochs // 5) - 1:
                print(f"Phase 2 - Epoch {epoch}/{min(20, epochs // 5)} - "
                      f"Loss: {total_loss.item():.6f}, "
                      f"PDE: {pde_loss.item():.6f}, "
                      f"IC: {ic_loss.item():.6f}, "
                      f"BC: {bc_loss.item():.6f}, "
                      f"Data: {data_loss.item():.6f}, "
                      f"LR: {current_lr:.6f}")
        
        # Phase 3: Full hybrid training
        print("Phase 3: Full hybrid training...")
        lambda_pde = lambda_pde_save
        lambda_ic = lambda_ic_save
        lambda_bc = lambda_bc_save
        lambda_data = lambda_data_save
    
    # Start or continue with main training loop
    start_epoch = len(history['epoch'])
    remaining_epochs = epochs - start_epoch
    
    print(f"Starting main training with {remaining_epochs} epochs...")
    
    for epoch in range(remaining_epochs):
        model.train()
        
        # Initialize losses
        pde_loss = torch.tensor(0.0, device=device, requires_grad=True)
        ic_loss = torch.tensor(0.0, device=device, requires_grad=True)
        bc_loss = torch.tensor(0.0, device=device, requires_grad=True)
        data_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        # Compute physics losses if using physics
        if use_physics:
            try:
                # PDE residual
                x_pde = interior_points[:, 0:1].clone().requires_grad_(True)
                y_pde = interior_points[:, 1:2].clone().requires_grad_(True)
                z_pde = interior_points[:, 2:3].clone().requires_grad_(True)
                t_pde = interior_points[:, 3:4].clone().requires_grad_(True)
                
                pde_residual, _ = model.compute_pde_residual(x_pde, y_pde, z_pde, t_pde)
                pde_loss = torch.mean(pde_residual**2)
                
                # Initial condition residual
                x_ic = initial_points[:, 0:1].clone().requires_grad_(True)
                y_ic = initial_points[:, 1:2].clone().requires_grad_(True)
                z_ic = initial_points[:, 2:3].clone().requires_grad_(True)
                t_ic = initial_points[:, 3:4].clone().requires_grad_(True)
                
                ic_residual, _ = model.compute_ic_residual(x_ic, y_ic, z_ic, t_ic)
                ic_loss = torch.mean(ic_residual**2)
                
                # Boundary condition residual
                bc_loss_sum = torch.tensor(0.0, device=device, requires_grad=True)
                bc_count = 0
                
                for boundary_type, boundary_points_tensor in boundary_points.items():
                    x_bc = boundary_points_tensor[:, 0:1].clone().requires_grad_(True)
                    y_bc = boundary_points_tensor[:, 1:2].clone().requires_grad_(True)
                    z_bc = boundary_points_tensor[:, 2:3].clone().requires_grad_(True)
                    t_bc = boundary_points_tensor[:, 3:4].clone().requires_grad_(True)
                    
                    try:
                        bc_residual, _ = model.compute_bc_residual(x_bc, y_bc, z_bc, t_bc, boundary_type)
                        bc_loss_sum = bc_loss_sum + torch.mean(bc_residual**2)
                        bc_count += 1
                    except Exception as e:
                        # Skip if BC computation fails for this boundary
                        print(f"Warning: BC computation failed for {boundary_type}: {e}")
                
                if bc_count > 0:
                    bc_loss = bc_loss_sum / bc_count
            except Exception as e:
                print(f"Warning in computing physics losses: {e}")
                # Continue without physics losses if they fail
                pde_loss = torch.tensor(0.0, device=device, requires_grad=True)
                ic_loss = torch.tensor(0.0, device=device, requires_grad=True)
                bc_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        # Compute data loss if FEM data is provided
        if fem_dataset is not None:
            # Shuffle data
            indices = torch.randperm(fem_dataset.size(0))
            fem_dataset_shuffled = fem_dataset[indices]
            
            # Process in batches
            all_data_residuals = []
            
            for i in range(0, fem_dataset.size(0), batch_size):
                end = min(i + batch_size, fem_dataset.size(0))
                batch = fem_dataset_shuffled[i:end]
                
                fem_t_batch = batch[:, 0:1]
                fem_x_batch = batch[:, 1:2]
                fem_y_batch = batch[:, 2:3]
                fem_z_batch = batch[:, 3:4]
                fem_T_batch = batch[:, 4:5]
                
                # Compute data residual
                data_residual, _ = model.compute_data_residual(
                    fem_x_batch, fem_y_batch, fem_t_batch, fem_T_batch
                )
                all_data_residuals.append(data_residual)
            
            # Combine all residuals
            if all_data_residuals:
                all_residuals = torch.cat(all_data_residuals, dim=0)
                data_loss = torch.mean(all_residuals**2)
            else:
                data_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        # Compute total loss
        total_loss = (
            lambda_pde * pde_loss +
            lambda_ic * ic_loss +
            lambda_bc * bc_loss +
            lambda_data * data_loss
        )
        
        # Backpropagation
        optimizer.zero_grad()
        total_loss.backward()
        
        # Clip gradients for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # Update weights
        optimizer.step()
        
        # Update learning rate
        if scheduler is not None:
            scheduler.step(total_loss)
        
        # Track current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Record history
        history['epoch'].append(start_epoch + epoch)
        history['loss'].append(total_loss.item())
        history['pde_loss'].append(pde_loss.item())
        history['ic_loss'].append(ic_loss.item())
        history['bc_loss'].append(bc_loss.item())
        history['data_loss'].append(data_loss.item())
        history['lr'].append(current_lr)
        
        # Print progress
        if epoch % 10 == 0 or epoch == remaining_epochs - 1:
            print(f"Main Training - Epoch {start_epoch + epoch}/{epochs} - "
                  f"Loss: {total_loss.item():.6f}, "
                  f"PDE: {pde_loss.item():.6f}, "
                  f"IC: {ic_loss.item():.6f}, "
                  f"BC: {bc_loss.item():.6f}, "
                  f"Data: {data_loss.item():.6f}, "
                  f"LR: {current_lr:.6f}")
        
        # Check for early stopping
        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            no_improve_count = 0
            # Save best model
            torch.save(model.state_dict(), os.path.join(save_path, "best_model.pth"))
        else:
            no_improve_count += 1
            if no_improve_count >= patience:
                print(f"Early stopping after {start_epoch + epoch + 1} epochs with no improvement.")
                break
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(save_path, "final_model.pth"))
    
    # Plot loss history
    plt.figure(figsize=(12, 8))
    plt.semilogy(history['epoch'], history['loss'], 'b-', label='Total Loss')
    plt.semilogy(history['epoch'], history['pde_loss'], 'r-', label='PDE Loss')
    plt.semilogy(history['epoch'], history['ic_loss'], 'g-', label='IC Loss')
    plt.semilogy(history['epoch'], history['bc_loss'], 'y-', label='BC Loss')
    plt.semilogy(history['epoch'], history['data_loss'], 'm-', label='Data Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log scale)')
    plt.title('Training Loss History')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.savefig(os.path.join(save_path, "loss_history.png"), dpi=300)
    plt.close()
    
    print(f"Training completed. Model saved to {save_path}")
    
    return model, history


def visualize_temperature_field(model, domain_bounds, time_point=1.0, resolution=100, save_path=None):
    """
    Improved visualization with higher resolution and proper temperature range.
    
    Args:
        model: Trained HybridPINN model
        domain_bounds: Dictionary with domain bounds
        time_point: Time point to visualize
        resolution: Resolution of the visualization grid
        save_path: Path to save the visualization
    """
    device = next(model.parameters()).device
    model.eval()
    
    # Extract domain bounds
    x_min, x_max = domain_bounds['x_min'], domain_bounds['x_max']
    y_min, y_max = domain_bounds['y_min'], domain_bounds['y_max']
    z_min, z_max = domain_bounds['z_min'], domain_bounds['z_max']
    
    # Create fixed y value (assuming 2D data in xy plane)
    fixed_y = y_min
    
    # Create grid with higher resolution
    x = np.linspace(x_min, x_max, resolution)
    z = np.linspace(z_min, z_max, resolution)
    X, Z = np.meshgrid(x, z)
    
    # Create tensors for prediction
    x_tensor = torch.tensor(X.flatten(), dtype=torch.float32).reshape(-1, 1).to(device)
    y_tensor = torch.ones_like(x_tensor) * fixed_y
    z_tensor = torch.tensor(Z.flatten(), dtype=torch.float32).reshape(-1, 1).to(device)
    t_tensor = torch.ones_like(x_tensor) * time_point
    
    # Convert from cm to m for input to the model
    x_tensor = x_tensor / 100.0
    y_tensor = y_tensor / 100.0
    z_tensor = z_tensor / 100.0
    
    # Predict temperatures in batches to avoid OOM
    batch_size = 1000
    n_points = x_tensor.shape[0]
    temperatures = []
    
    with torch.no_grad():
        for i in range(0, n_points, batch_size):
            end_idx = min(i + batch_size, n_points)
            temp_batch = model(
                x_tensor[i:end_idx], 
                y_tensor[i:end_idx], 
                z_tensor[i:end_idx], 
                t_tensor[i:end_idx]
            )
            temperatures.append(temp_batch.cpu())
    
    # Combine and reshape
    temperatures = torch.cat(temperatures, dim=0).numpy().reshape(resolution, resolution)
    
    # Convert to Celsius
    temperatures_c = temperatures - 273.15
    
    # Auto-adjust colormap range based on data
    temp_min = max(np.min(temperatures_c), 25.0)  # Min temp or 25°C
    temp_max = min(np.max(temperatures_c), 3000.0)  # Max temp or 3000°C
    
    # Create figure with better colormap
    plt.figure(figsize=(12, 10))
    contour = plt.contourf(X, Z, temperatures_c, 50, cmap='inferno', 
                          vmin=temp_min, vmax=temp_max)
    plt.colorbar(label='Temperature (°C)')
    plt.xlabel('X (cm)')
    plt.ylabel('Z (cm)')
    plt.title(f'Temperature Field at t={time_point}s')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return temperatures_c


def plot_temperature_history(model, domain_bounds, points, save_path=None):
    """
    Plot temperature history at specific points.
    
    Args:
        model: Trained HybridPINN model
        domain_bounds: Dictionary with domain bounds
        points: List of dictionaries with x, y, z coordinates and names
        save_path: Path to save the plot
    """
    device = next(model.parameters()).device
    model.eval()
    
    # Extract domain bounds
    t_max = domain_bounds.get('t_max_deposition', 100.0)
    
    # Create time points
    time_points = np.linspace(0, t_max, 100)
    
    plt.figure(figsize=(10, 6))
    
    for point in points:
        x = torch.tensor([[point['x'] / 100.0]], dtype=torch.float32).to(device)  # Convert cm to m
        y = torch.tensor([[point['y'] / 100.0]], dtype=torch.float32).to(device)
        z = torch.tensor([[point['z'] / 100.0]], dtype=torch.float32).to(device)
        
        temperatures = []
        
        # Predict temperatures at each time point
        with torch.no_grad():
            for t in time_points:
                t_tensor = torch.tensor([[t]], dtype=torch.float32).to(device)
                temp = model(x, y, z, t_tensor).item()
                temperatures.append(temp)
        
        # Convert to Celsius
        temperatures_c = [temp - 273.15 for temp in temperatures]
        
        # Plot temperature history
        plt.plot(time_points, temperatures_c, label=f"{point['name']} ({point['x']}, {point['y']}, {point['z']} cm)")
    
    plt.xlabel('Time (s)')
    plt.ylabel('Temperature (°C)')
    plt.title('Temperature History at Different Points')
    plt.grid(True)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

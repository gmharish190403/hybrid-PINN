import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import traceback

from hybrid_pinn_model import HybridPINN
from train_hybrid_pinn import (
    load_preprocess_fem_data, 
    train_hybrid_pinn,
    visualize_temperature_field,
    plot_temperature_history
)

def main():
    # For testing without command line args
    if not os.path.exists("./test_flag"):
        
        # Create args object with hardcoded settings for testing
        class Args:
            pass
        
        args = Args()
        args.train = True
        args.visualize = True
        args.predict = False
        args.data = "2D Temp_Data.xlsx"
        args.model = None
        args.output_dir = "./results"
        args.epochs = 100
        args.lr = 1e-3
        args.batch_size = 512
        args.lambda_data = 10.0
        args.lambda_pde = 1.0
        args.use_physics = True
        args.time_points = "0,75,150"
        
        # Create a test flag file so we know we've run with hardcoded args
        with open("./test_flag", "w") as f:
            f.write("Test run completed")
    else:
        # Use command line arguments
        parser = argparse.ArgumentParser(description='Hybrid PINN for LMD Temperature Prediction')
        
        # Action arguments
        parser.add_argument('--train', action='store_true', help='Train a new model')
        parser.add_argument('--visualize', action='store_true', help='Visualize results from a trained model')
        parser.add_argument('--predict', action='store_true', help='Make predictions using a trained model')
        
        # Data and model arguments
        parser.add_argument('--data', type=str, default='2D Temp_Data.xlsx', help='Path to FEM data file')
        parser.add_argument('--model', type=str, default=None, help='Path to trained model file')
        parser.add_argument('--output_dir', type=str, default='./results', help='Directory to save outputs')
        
        # Training parameters
        parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
        parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
        parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
        parser.add_argument('--lambda_data', type=float, default=10.0, help='Weight for data loss')
        parser.add_argument('--lambda_pde', type=float, default=1.0, help='Weight for PDE loss')
        parser.add_argument('--use_physics', action='store_true', help='Use physics constraints in training')
        
        # Visualization parameters
        parser.add_argument('--time_points', type=str, default='0,75,150', 
                          help='Comma-separated list of time points to visualize')
        
        # Parse args
        args = parser.parse_args()
    
    print("\n=== DEBUG: CHECKING ARGS ===")
    print(f"args.train = {args.train}")
    print(f"Type of args: {type(args)}")
    print(f"Attributes in args: {[attr for attr in dir(args) if not attr.startswith('__')]}\n")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load FEM data if specified
    fem_data = None
    domain_bounds = None
    
    if args.data and os.path.exists(args.data):
        try:
            print(f"Loading FEM data from {args.data}...")
            fem_data, domain_info = load_preprocess_fem_data(
                args.data, 
                plot_heatmap=True
            )
            
            domain_bounds = {
                'x_min': domain_info['x_min'],
                'x_max': domain_info['x_max'],
                'y_min': domain_info['y_min'],
                'y_max': domain_info['y_max'],
                'z_min': 0.0,
                'z_max': 0.2,
                't_max_deposition': domain_info['t_max_deposition']
            }
            
            print(f"FEM data loaded with shape: {fem_data.shape}")
            print(f"Domain bounds: {domain_bounds}")
        except Exception as e:
            print(f"Error loading FEM data: {e}")
            traceback.print_exc()
            if args.train:
                print("Cannot train without FEM data. Exiting.")
                return
    else:
        if args.train and not args.use_physics:
            print("No FEM data found. Cannot train without data or physics. Exiting.")
            return
        
        # Define default domain bounds if no FEM data is available
        domain_bounds = {
            'x_min': 1.0,
            'x_max': 8.0,
            'y_min': 3.0,
            'y_max': 3.0,
            'z_min': 0.0,
            'z_max': 0.2,
            't_max_deposition': 150.0
        }
    
    print("Debug: Before training check")
    # TRAIN a new model
    if args.train:
        print("\n=== DEBUG: STARTING TRAINING ===")
        print(f"Training parameters: epochs={args.epochs}, batch_size={args.batch_size}")
        print(f"Loss weights: lambda_data={args.lambda_data}, lambda_pde={args.lambda_pde}")
        print(f"Use physics: {args.use_physics}")
        
        try:
            # Initialize model
            print("Initializing model...")
            model = HybridPINN(
                domain_bounds=domain_bounds,
                z_fixed=0.1,
                z_thickness=0.2,
                hidden_layers=6,
                neurons_per_layer=128,
                laser_params={
                    'power': 2000.0,     # Laser power (W)
                    'velocity': 8.0,     # Scanning speed (mm/s)
                    'absorption': 0.75,  # Absorption coefficient
                    'Ra': 0.3 / 100.0,   # Semi-axis in x (cm to m)
                    'Rb': 0.3 / 100.0,   # Semi-axis in y (cm to m)
                    'Rc': 0.1 / 100.0,   # Semi-axis in z (cm to m)
                    'start_pos': [domain_bounds['x_min'] / 100.0, domain_bounds['y_min'] / 100.0],  # Start pos (cm to m)
                    'end_pos': [domain_bounds['x_max'] / 100.0, domain_bounds['y_min'] / 100.0]     # End pos (cm to m)
                }
            )
            
            # Set up training parameters
            train_params = {
                'epochs': args.epochs,
                'batch_size': args.batch_size,
                'learning_rate': args.lr,
                'lambda_data': args.lambda_data,
                'lambda_pde': args.lambda_pde if args.use_physics else 0.0,
                'lambda_ic': 0.1 if args.use_physics else 0.0,
                'lambda_bc': 0.1 if args.use_physics else 0.0,
                'use_physics': args.use_physics,
                'save_path': args.output_dir,
                'patience': 20  # Increased patience for early stopping
            }
            
            print(f"Starting training with parameters: {train_params}")
            
            # Train model
            trained_model, history = train_hybrid_pinn(
                model=model,
                domain_bounds=domain_bounds,
                fem_data=fem_data,
                **train_params
            )
            
            print(f"Training completed. Model saved to {args.output_dir}")
            
            # Visualize results after training
            print("Visualizing training results...")
            try:
                # Parse time points 
                time_points = [float(t) for t in args.time_points.split(',')]
            except:
                time_points = [0.0, domain_bounds['t_max_deposition'] / 2, domain_bounds['t_max_deposition']]
            
            for t in time_points:
                print(f"Visualizing temperature field at t={t}s...")
                visualize_temperature_field(
                    model=trained_model,
                    domain_bounds=domain_bounds,
                    time_point=t,
                    save_path=os.path.join(args.output_dir, f"temperature_field_t{t:.1f}.png")
                )
        except Exception as e:
            print(f"ERROR in training: {e}")
            traceback.print_exc()
    else:
        print("Debug: args.train is False, skipping training")
    
    # VISUALIZE results from a trained model
    if args.visualize:
        if args.model is None and not args.train:
            # Try to load the best model from the output directory
            model_path = os.path.join(args.output_dir, "best_model.pth")
            if not os.path.exists(model_path):
                print(f"No model found at {model_path}. Please specify a model path with --model.")
                return
        else:
            model_path = args.model if args.model else os.path.join(args.output_dir, "best_model.pth")
        
        try:
            # Load model
            model = HybridPINN(
                domain_bounds=domain_bounds,
                z_fixed=0.1,
                z_thickness=0.2,
                hidden_layers=6,
                neurons_per_layer=128
            )
            
            # Using weights_only=True to avoid security warning
            model.load_state_dict(torch.load(model_path, weights_only=True))
            print(f"Model loaded from {model_path}")
            
            # Parse time points to visualize
            try:
                time_points = [float(t) for t in args.time_points.split(',')]
            except:
                time_points = [0.0, domain_bounds['t_max_deposition'] / 2, domain_bounds['t_max_deposition']]
            
            # Visualize temperature fields at specified time points
            for t in time_points:
                visualize_temperature_field(
                    model=model,
                    domain_bounds=domain_bounds,
                    time_point=t,
                    save_path=os.path.join(args.output_dir, f"temperature_field_t{t:.1f}.png")
                )
            
            # Define points of interest for temperature history plot
            points = [
                {'name': 'Start', 'x': domain_bounds['x_min'], 'y': domain_bounds['y_min'], 'z': 0.0},
                {'name': 'Middle', 'x': (domain_bounds['x_min'] + domain_bounds['x_max']) / 2, 
                'y': domain_bounds['y_min'], 'z': 0.0},
                {'name': 'End', 'x': domain_bounds['x_max'], 'y': domain_bounds['y_min'], 'z': 0.0},
            ]
            
            # Plot temperature history
            plot_temperature_history(
                model=model,
                domain_bounds=domain_bounds,
                points=points,
                save_path=os.path.join(args.output_dir, "temperature_history.png")
            )
        except Exception as e:
            print(f"ERROR in visualization: {e}")
            traceback.print_exc()
    
    # PREDICT specific temperatures
    if args.predict:
        if args.model is None and not args.train:
            # Try to load the best model from the output directory
            model_path = os.path.join(args.output_dir, "best_model.pth")
            if not os.path.exists(model_path):
                print(f"No model found at {model_path}. Please specify a model path with --model.")
                return
        else:
            model_path = args.model if args.model else os.path.join(args.output_dir, "best_model.pth")
        
        try:
            # Load model
            model = HybridPINN(
                domain_bounds=domain_bounds,
                z_fixed=0.1,
                z_thickness=0.2,
                hidden_layers=6,
                neurons_per_layer=128
            )
            
            # Using weights_only=True to avoid security warning
            model.load_state_dict(torch.load(model_path, weights_only=True))
            print(f"Model loaded from {model_path}")
            
            # Simple interactive prediction mode
            print("\nPrediction Mode - Enter coordinates and time to predict temperature")
            print("Enter 'q' to quit")
            
            while True:
                try:
                    user_input = input("\nEnter x,y,z,t (cm,cm,cm,s): ")
                    if user_input.lower() == 'q':
                        break
                    
                    # Parse input
                    x, y, z, t = map(float, user_input.split(','))
                    
                    # Convert to tensors and from cm to m
                    x_tensor = torch.tensor([[x / 100.0]], dtype=torch.float32)
                    y_tensor = torch.tensor([[y / 100.0]], dtype=torch.float32)
                    z_tensor = torch.tensor([[z / 100.0]], dtype=torch.float32)
                    t_tensor = torch.tensor([[t]], dtype=torch.float32)
                    
                    # Predict temperature
                    with torch.no_grad():
                        temp = model(x_tensor, y_tensor, z_tensor, t_tensor).item()
                    
                    print(f"Temperature at ({x}, {y}, {z}) cm, t={t}s: {temp:.2f} K ({temp-273.15:.2f} °C)")
                    
                except ValueError:
                    print("Invalid input format. Please use format: x,y,z,t")
                except Exception as e:
                    print(f"Error: {e}")
                    
        except Exception as e:
            print(f"ERROR in prediction: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    print("Debug: Before calling main()")
    try:
        main()
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        traceback.print_exc()

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def visualize_logs(log_dir='tensor_logs'):
    """
    Visualizes training logs saved by TensorMonitor. For each layer (recorded in metadata)
    it displays, for each epoch, histograms for weights, biases, weight gradients (dweights)
    and bias gradients (dbiases). The figures are embedded in a scrollable, resizable Tkinter window.
    
    Args:
        log_dir (str): Directory containing the log files.
    """
    log_files = [f for f in os.listdir(log_dir) if f.endswith('.json')]
    if not log_files:
        print(f"No JSON log files found in '{log_dir}'.")
        return

    # Create main Tkinter window for scrolling and resizing
    root = tk.Tk()
    root.title("Tensor Logs Visualization")
    
    # Create a canvas and a vertical scrollbar
    canvas = tk.Canvas(root, borderwidth=0)
    vscrollbar = ttk.Scrollbar(root, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=vscrollbar.set)
    vscrollbar.pack(side="right", fill="y")
    canvas.pack(side="left", fill="both", expand=True)
    
    # Create a frame inside the canvas to hold all figures
    container = ttk.Frame(canvas)
    canvas.create_window((0, 0), window=container, anchor="nw")
    
    def on_configure(event):
        canvas.configure(scrollregion=canvas.bbox("all"))
    container.bind("<Configure>", on_configure)
    
    # Process each log file
    for log_file in log_files:
        log_path = os.path.join(log_dir, log_file)
        print(f"Visualizing logs from: {log_path}")
        with open(log_path, 'r') as f:
            log_data = json.load(f)
        
        epochs_data = log_data.get('epochs', [])
        if not epochs_data:
            print("No epoch data found in logs.")
            continue
        
        layer_metadata = log_data.get('metadata', {}).get('layers', [])
        # For each layer logged in metadata, create a separate figure.
        for layer_info in layer_metadata:
            layer_name = layer_info.get('name', 'Unknown Layer')
            n_epochs = len(epochs_data)
            # We will use 4 columns: weights, biases, dweights, dbiases.
            fig_cols = 4
            # Determine dynamic figure height; e.g., 2 inches per epoch row
            fig_height_per_epoch = 2.5
            total_fig_height = max(6, n_epochs * fig_height_per_epoch)
            
            fig, axes = plt.subplots(n_epochs, fig_cols, figsize=(16, total_fig_height),
                                     gridspec_kw={'hspace': 0.5, 'wspace': 0.3})
            fig.suptitle(f'Layer: {layer_name}', fontsize=16)
            
            # If there is only one epoch, make axes 2D for uniform indexing.
            if n_epochs == 1:
                axes = np.array([axes])
            
            # For each epoch, extract the histograms and plot them.
            for epoch_index, epoch_data in enumerate(epochs_data):
                histograms = epoch_data.get('parameters', {}).get('histograms', {})
                
                # List of parameter tags to plot for this layer
                tags = [f'{layer_name}/weights', f'{layer_name}/biases', 
                        f'{layer_name}/dweights', f'{layer_name}/dbiases']
                titles = ['Weights', 'Biases', 'dWeights', 'dBiases']
                
                for col, (tag, title) in enumerate(zip(tags, titles)):
                    ax = axes[epoch_index, col]
                    if tag in histograms:
                        param_data = histograms[tag]
                        hist = np.array(param_data.get('histogram', []))
                        bin_edges = np.array(param_data.get('bin_edges', []))
                        if len(hist) > 0 and len(bin_edges) > 1:
                            ax.bar(bin_edges[:-1], hist, width=np.diff(bin_edges), align='edge')
                            ax.set_title(f'Epoch {epoch_data.get("epoch", epoch_index+1)} {title}', fontsize=10)
                            ax.set_xlabel(title + ' Value', fontsize=8)
                            ax.set_ylabel('Frequency', fontsize=8)
                            ax.tick_params(axis='both', labelsize=8)
                        else:
                            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', color='red', fontsize=10)
                            ax.set_title(f'Epoch {epoch_data.get("epoch", epoch_index+1)} {title} - No Data', fontsize=10)
                            ax.axis('off')
                    else:
                        ax.text(0.5, 0.5, 'No Data', ha='center', va='center', color='red', fontsize=10)
                        ax.set_title(f'Epoch {epoch_data.get("epoch", epoch_index+1)} {title} - Not Logged', fontsize=10)
                        ax.axis('off')
            
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            
            # Embed the figure in the Tkinter window
            canvas_fig = FigureCanvasTkAgg(fig, master=container)
            canvas_fig.draw()
            widget = canvas_fig.get_tk_widget()
            widget.pack(padx=10, pady=10)
            
            # Optional: add a separator between figures
            separator = ttk.Separator(container, orient='horizontal')
            separator.pack(fill='x', padx=5, pady=5)
    
    # Start Tkinter's event loop
    root.mainloop()

# Example usage:
if __name__ == "__main__":
    visualize_logs()  # Adjust log_dir as needed

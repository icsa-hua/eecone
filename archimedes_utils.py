import os
import cv2
import numpy as np
from scipy import stats
from plotly.subplots import make_subplots
import plotly.graph_objects as go


def load_images(folder):
        """Load images from a given folder and return as a list of numpy arrays."""
        images = []
        for filename in sorted(os.listdir(folder)):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                img = cv2.imread(os.path.join(folder, filename))
                if img is not None:
                    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    images.append(img)
        return images


def perform_ks_test(ref_sample, test_sample):
    """Perform the Kolmogorov-Smirnov test to detect significant change."""
    ks_stat, p_value = stats.ks_2samp(ref_sample, test_sample)
    return ks_stat, p_value


def compute_color_relative_intensity(image, color='red'):
    """Compute the Red Relative Intensity (RRI) of an image."""
    r, g, b = cv2.split(image)
    total_intensity = (r + g + b).astype(np.float32) + 1e-6  # Avoid division by zero
    if color == 'red':
        r_relative = r.astype(np.float32) / total_intensity
    elif color == 'green':
        r_relative = g.astype(np.float32) / total_intensity
    elif color == 'blue':
        r_relative = b.astype(np.float32) / total_intensity
    else:
        raise ValueError("Color must be 'red', 'green', or 'blue'")

    return np.mean(r_relative)


def plot_mrri_over_time(mrri_values, ref_start, ref_end, test_start, test_end):
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=list(range(len(mrri_values))),y=mrri_values, name='MRRI', mode='lines+markers', marker=dict(size=4), line=dict(width=2)))
    fig.add_vrect(x0=ref_start, x1=ref_end, fillcolor="lightblue", opacity=0.5, layer="below", line_width=0, name="Reference Images")
    fig.add_vrect(x0=test_start, x1=test_end, fillcolor="tomato", opacity=0.5, layer="below", line_width=0, name="Test Images")
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(color='lightblue', opacity=0.5), name='Reference Images', showlegend=True))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(color='tomato', opacity=0.5), name='Test Images',showlegend=True))

    fig.update_layout(
        height=400,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        margin=dict(t=85, b=0, l=0, r=0),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False, visible=False),
        xaxis=dict(range=[0, len(mrri_values)-1],autorange=False)
    )

    return fig
    
    
def plot_avg_comparison(ref_images, test_images):
    # Compute average images
    ref_stack = np.stack(ref_images, axis=0)
    ref_avg = np.mean(ref_stack, axis=0).astype(np.uint8)
    test_stack = np.stack(test_images, axis=0)
    test_avg = np.mean(test_stack, axis=0).astype(np.uint8)
    
    # Create subplot with 1 row and 2 columns
    fig = make_subplots(rows=1, cols=2, subplot_titles=(
        '<span style="color: lightblue">Averaged RGB Reference Image</span>',
        '<span style="color: tomato">Averaged RGB Test Image</span>'
    ), horizontal_spacing=0)
    
    # Add images to subplots
    fig.add_trace(go.Image(z=ref_avg), row=1, col=1)
    fig.add_trace(go.Image(z=test_avg), row=1, col=2)
    
    # Update layout
    fig.update_layout(
        margin=dict(t=40, b=10, l=0, r=0),
        showlegend=False,
        width=200,
    )
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, row=1, col=1)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, row=1, col=1)
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, row=1, col=2)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, row=1, col=2)
    
    return fig


def plot_avg_rgb(images):
    stack = np.stack(images, axis=0)
    avg = np.mean(stack, axis=0).astype(np.uint8)

    fig = go.Figure()
    fig.add_trace(go.Image(z=avg))
    
    fig.update_layout(
        margin=dict(t=0, b=0, l=0, r=0),
        showlegend=False,
    )
    
    return fig


def apply_status_style(val):
    if val == 'No Drift Detected':
        return 'color: green'
    elif val == 'Drift Detected!':
        return 'color: red'
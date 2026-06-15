import logging
from pathlib import Path
from typing import Dict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

# Set style for modern visuals
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'axes.edgecolor': '#e0e6ed',
    'axes.linewidth': 0.8,
    'figure.facecolor': '#ffffff',
    'axes.facecolor': '#ffffff'
})

logger = logging.getLogger(__name__)

# Primary clinical theme color palette
CLINICAL_COLORS = {
    'primary': '#0f62fe',      # Corporate Blue
    'secondary': '#00d8f6',    # Cyan
    'accent_dark': '#111b24',  # Dark Navy
    'melanoma': '#c1272d',     # Crimson
    'benign': '#24a148',       # Green
    'warning': '#f5a623'       # Amber
}

def plot_class_distribution(df: pd.DataFrame, class_map: Dict[str, str], output_path: Path):
    """
    Generates class distribution charts: a frequency bar chart and a percentage pie chart.
    """
    logger.info("Plotting class distribution...")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Compute counts and maps
    counts = df['dx'].value_counts()
    labels = [f"{class_map.get(k, k)} ({k})" for k in counts.index]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # 1. Bar Chart
    colors = [CLINICAL_COLORS['melanoma'] if k == 'mel' else CLINICAL_COLORS['primary'] for k in counts.index]
    sns.barplot(x=counts.values, y=labels, ax=axes[0], palette=colors, hue=labels, legend=False)
    axes[0].set_title("Number of Images Per Lesion Category", fontsize=14, fontweight='bold', pad=15)
    axes[0].set_xlabel("Image Count", fontweight='semibold')
    axes[0].set_ylabel("Lesion Type", fontweight='semibold')
    for index, value in enumerate(counts.values):
        axes[0].text(value + (max(counts.values) * 0.01), index, f" {value}", va='center', fontweight='semibold')
        
    # 2. Pie Chart
    pie_colors = sns.color_palette("Blues_r", len(counts))
    # Highlight Melanoma
    explode = [0.1 if k == 'mel' else 0 for k in counts.index]
    axes[1].pie(counts.values, labels=labels, autopct='%1.1f%%', startangle=140, 
                colors=pie_colors, explode=explode, shadow=False,
                textprops={'fontsize': 10, 'weight': 'semibold'})
    axes[1].set_title("Lesion Proportion Breakdown (%)", fontsize=14, fontweight='bold', pad=15)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Class distribution chart saved to: {output_path.resolve()}")

def plot_metadata_analysis(df: pd.DataFrame, output_path: Path):
    """
    Generates metadata distribution charts: Age distribution, Gender count, and Localization.
    """
    logger.info("Plotting metadata analysis...")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Age Distribution Histogram
    # Remove null values for distribution plot
    age_data = df['age'].dropna()
    if not age_data.empty:
        sns.histplot(age_data, kde=True, bins=20, ax=axes[0, 0], color=CLINICAL_COLORS['primary'])
        axes[0, 0].set_title("Patient Age Distribution", fontsize=13, fontweight='bold', pad=12)
        axes[0, 0].set_xlabel("Age (years)", fontweight='semibold')
        axes[0, 0].set_ylabel("Count", fontweight='semibold')
    else:
        axes[0, 0].text(0.5, 0.5, "No Age Data Available", ha='center', va='center')
        
    # 2. Gender Distribution Bar
    gender_counts = df['sex'].value_counts()
    sns.barplot(x=gender_counts.index, y=gender_counts.values, ax=axes[0, 1], 
                palette="Blues_r", hue=gender_counts.index, legend=False)
    axes[0, 1].set_title("Patient Gender Breakdown", fontsize=13, fontweight='bold', pad=12)
    axes[0, 1].set_xlabel("Gender", fontweight='semibold')
    axes[0, 1].set_ylabel("Count", fontweight='semibold')
    for index, value in enumerate(gender_counts.values):
        axes[0, 1].text(index, value + (max(gender_counts.values) * 0.01), f"{value}", ha='center', fontweight='semibold')

    # 3. Anatomical Localization
    loc_counts = df['localization'].value_counts()
    sns.barplot(x=loc_counts.values, y=loc_counts.index, ax=axes[1, 0], 
                palette="viridis", hue=loc_counts.index, legend=False)
    axes[1, 0].set_title("Lesion Anatomical Site Localization", fontsize=13, fontweight='bold', pad=12)
    axes[1, 0].set_xlabel("Count", fontweight='semibold')
    axes[1, 0].set_ylabel("Anatomical Site", fontweight='semibold')
    
    # 4. Age vs Lesion Type Box Plot
    sns.boxplot(data=df, x='age', y='dx', ax=axes[1, 1], palette="Set2", hue='dx', legend=False)
    axes[1, 1].set_title("Age Distribution per Lesion Type", fontsize=13, fontweight='bold', pad=12)
    axes[1, 1].set_xlabel("Age (years)", fontweight='semibold')
    axes[1, 1].set_ylabel("Lesion Code", fontweight='semibold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Metadata demographic analysis chart saved to: {output_path.resolve()}")

def plot_sample_images(df: pd.DataFrame, class_map: Dict[str, str], output_path: Path):
    """
    Renders a grid displaying one sample image per disease category.
    Includes details about width/height dimensions.
    """
    logger.info("Plotting sample images...")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    classes = list(class_map.keys())
    fig, axes = plt.subplots(2, 4, figsize=(16, 9))
    axes = axes.ravel()
    
    for idx, dx in enumerate(classes):
        ax = axes[idx]
        
        # Filter for rows in this category with a valid path mapping
        valid_rows = df[(df['dx'] == dx) & (df['image_path'].notna())]
        
        if not valid_rows.empty:
            sample_row = valid_rows.iloc[0]
            filepath = sample_row['image_path']
            image_id = sample_row['image_id']
            
            # Read image
            img = cv2.imread(filepath)
            if img is not None:
                h, w, c = img.shape
                # OpenCV uses BGR, Matplotlib uses RGB
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                ax.imshow(img_rgb)
                ax.axis('off')
                
                # Check color scheme status
                colorspace = "RGB" if c == 3 else "Gray"
                
                title_text = f"{class_map[dx]}\n({dx})\nID: {image_id}\n{w}x{h} | {colorspace}"
                
                # Frame Melanoma and other skin cancers in warning colors
                box_color = CLINICAL_COLORS['melanoma'] if dx in ['mel', 'bcc', 'akiec'] else '#dddddd'
                ax.text(0.5, -0.22, title_text, transform=ax.transAxes, 
                        ha='center', va='top', fontsize=9, fontweight='semibold',
                        bbox=dict(boxstyle="round,pad=0.3", fc="#ffffff", ec=box_color, lw=1.5))
            else:
                _draw_placeholder(ax, f"Corrupted:\n{image_id}", dx, class_map)
        else:
            _draw_placeholder(ax, "No Valid Image Found", dx, class_map)
            
    # Hide the final empty grid cell since we have 7 classes
    if len(classes) < len(axes):
        for idx in range(len(classes), len(axes)):
            axes[idx].axis('off')
            
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Sample images grid saved to: {output_path.resolve()}")

def _draw_placeholder(ax, msg: str, dx: str, class_map: Dict[str, str]):
    """
    Draws a visual placeholder if an image is missing or corrupted.
    """
    ax.axis('off')
    placeholder = np.ones((100, 100, 3), dtype=np.uint8) * 230 # Gray box
    ax.imshow(placeholder)
    ax.text(50, 50, msg, ha='center', va='center', fontsize=8, color='#888888', fontweight='bold')
    
    title_text = f"{class_map[dx]} ({dx})\n[MISSING]"
    ax.text(0.5, -0.22, title_text, transform=ax.transAxes, 
            ha='center', va='top', fontsize=9, fontweight='semibold',
            bbox=dict(boxstyle="round,pad=0.3", fc="#fcf8e3", ec="#fbeed5", lw=1))

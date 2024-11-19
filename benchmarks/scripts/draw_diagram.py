import pandas as pd
import matplotlib.pyplot as plt

# Function to create and save the plot
def create_plot(dataframe, filename, title):
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(20, 6))
    
    # Plot rectangles and text
    box_size = 1.2  # Set size of the square box
    spacing = 0.2   # Set spacing between the boxes

    for idx, row in dataframe.iterrows():
        stage = row['stage']
        step = row['step']
        micro_batch_id = row['micro_batch_id']
        forward = row['forward']
        
        # Choose color based on 'forward'
        color = '#375390' if forward else '#B0CE94'
        
        # Shade box if micro_batch_id is not between 0 and 5
        if not (0 <= micro_batch_id <= 5):
            color = 'gray'  # Highlighted color for invalid micro_batch_id
        
        # Draw rectangle at (step, stage)
        rect = plt.Rectangle(
            (step * (box_size + spacing), stage * (box_size + spacing)),  # Position with spacing
            box_size,  # Width
            box_size,  # Height
            facecolor=color,
            edgecolor='black'
        )
        ax.add_patch(rect)
        
        # Add text of 'micro_batch_id' inside rectangle
        ax.text(
            step * (box_size + spacing) + box_size / 2,
            stage * (box_size + spacing) + box_size / 2,
            str(micro_batch_id),
            ha='center',
            va='center',
            fontsize=16,
            fontweight='bold',
            color='black'
        )

    # Set axis limits
    ax.set_xlim(-spacing, (dataframe['step'].max() + 1) * (box_size + spacing))
    ax.set_ylim(-spacing, (dataframe['stage'].max() + 1) * (box_size + spacing))

    # Customize ticks and labels
    ax.set_xticks([(i + 0.5) * (box_size + spacing) for i in range(dataframe['step'].max() + 1)])
    ax.set_xticklabels(range(dataframe['step'].max() + 1), size=16)
    ax.set_yticks([(i + 0.5) * (box_size + spacing) for i in sorted(dataframe['stage'].unique())])
    ax.set_yticklabels(sorted(dataframe['stage'].unique()), va='center', size=16)

    # Label axes
    ax.set_xlabel('Micro Step within a Step', size=16)
    ax.set_ylabel('Stage', size=16)
    # ax.set_title(title, size=18)

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# Read datasets
df_ours = pd.read_csv('grid_ours.csv')
df = pd.read_csv('grid.csv')

# Create and save plots
create_plot(df_ours, 'grid_ours.png', 'Grid Ours')
create_plot(df, 'grid.png', 'Grid')

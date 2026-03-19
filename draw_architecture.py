import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def draw_block(ax, x, y, width, height, text, color='#E6F3FF', edgecolor='#3366CC', fontsize=10, weight='bold'):
    """Draw a basic rectangular block with text."""
    rect = patches.FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.1,rounding_size=0.2", 
                                  linewidth=1.5, edgecolor=edgecolor, facecolor=color, zorder=2, alpha=0.9)
    ax.add_patch(rect)
    ax.text(x + width/2, y + height/2, text, ha='center', va='center', fontsize=fontsize, fontweight=weight, color='#333333', zorder=3, wrap=True)
    return rect

def draw_arrow(ax, x1, y1, x2, y2, text='', arrow_style='-|>', rad=0.0):
    """Draw an arrow from (x1, y1) to (x2, y2)."""
    arrow = patches.FancyArrowPatch((x1, y1), (x2, y2), connectionstyle=f"arc3,rad={rad}", 
                                    color='#666666', arrowstyle=f'{arrow_style},head_length=6,head_width=4', 
                                    linewidth=1.5, zorder=1)
    ax.add_patch(arrow)
    if text:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        # Offset slightly for clarity
        ax.text(mx, my + 0.1, text, ha='center', va='bottom', fontsize=9, color='#555555', zorder=3, style='italic')

def draw_dashed_box(ax, x, y, width, height, title=''):
    """Draw a grouping box with dashed lines."""
    rect = patches.Rectangle((x, y), width, height, linewidth=1.5, edgecolor='#999999', facecolor='none', linestyle='--', zorder=1, alpha=0.8)
    ax.add_patch(rect)
    if title:
        ax.text(x + 0.1, y + height - 0.1, title, ha='left', va='top', fontsize=11, fontweight='bold', color='#666666', zorder=3)
    return rect

def generate_architecture_diagram():
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Colors
    color_input = '#F5F5F5'  # Light gray
    color_stat = '#FFF0F0'   # Light red/pink
    color_sem = '#F0FFF0'    # Light green
    color_struct = '#F0F5FF' # Light blue
    color_gnn = '#FFF5E6'    # Light orange
    color_fusion = '#F0E6FF' # Light purple
    color_out = '#FFFFE6'    # Light yellow
    
    edge_color = '#4D4D4D'
    
    # 1. Input Layer
    draw_dashed_box(ax, 0.5, 3.5, 2.5, 5, title="Input Data")
    draw_block(ax, 0.8, 6.5, 1.9, 1.2, "Raw Network\nTraffic (PCAP)", color=color_input, edgecolor=edge_color)
    draw_block(ax, 0.8, 4.5, 1.9, 1.2, "Flow Records\n(NetFlow)", color=color_input, edgecolor=edge_color)
    
    draw_arrow(ax, 1.75, 6.5, 1.75, 5.7)
    
    # 2. Feature Extraction Layer
    draw_dashed_box(ax, 4, 1.5, 3.5, 8, title="Tri-Modal Feature Extraction")
    
    # Structural
    b_struct = draw_block(ax, 4.5, 7.5, 2.5, 1.2, "Structural Feature\nExtractor\n(Topology & Graph)", color=color_struct, edgecolor='#4A86E8')
    # Statistical
    b_stat = draw_block(ax, 4.5, 5.0, 2.5, 1.2, "Statistical Feature\nExtractor\n(High-Dim & IAT)", color=color_stat, edgecolor='#CC0000')
    # Semantic
    b_sem = draw_block(ax, 4.5, 2.5, 2.5, 1.2, "Semantic Feature\nExtractor\n(Payload & Intent)", color=color_sem, edgecolor='#38761D')
    
    # Arrows from input to extractors
    draw_arrow(ax, 2.7, 5.1, 4.5, 8.1, text="Connections", rad=0.2)
    draw_arrow(ax, 2.7, 5.1, 4.5, 5.6, text="Stats")
    draw_arrow(ax, 2.7, 5.1, 4.5, 3.1, text="Payloads", rad=-0.2)
    
    # 3. Graph Neural Network Layer
    draw_dashed_box(ax, 8.5, 1.5, 3.5, 8, title="Heterogeneous GNN Modules")
    
    # GCN (Struct)
    draw_block(ax, 9.0, 7.5, 2.5, 1.2, "Structure GCN\n(Message Passing)", color=color_gnn, edgecolor='#E69138')
    # GAT (Stat)
    draw_block(ax, 9.0, 5.0, 2.5, 1.2, "Statistical GAT\n(Attention Mech)", color=color_gnn, edgecolor='#E69138')
    # GCN (Sem)
    draw_block(ax, 9.0, 2.5, 2.5, 1.2, "Semantic GCN\n(Content Passing)", color=color_gnn, edgecolor='#E69138')
    
    # Arrows from Extractors to GNN
    draw_arrow(ax, 7.0, 8.1, 9.0, 8.1, "16-Dim")
    draw_arrow(ax, 7.0, 5.6, 9.0, 5.6, "64-Dim")
    draw_arrow(ax, 7.0, 3.1, 9.0, 3.1, "32-Dim")
    
    # Inter-GNN connections (Heterogeneous aspect)
    # To show that they are a joint heterogeneous graph
    ax.text(10.25, 4.1, "Heterogeneous\nEdges", ha='center', va='center', fontsize=9, color='#777777', style='italic')
    draw_arrow(ax, 10.25, 5.0, 10.25, 3.7, arrow_style='<|-|>', rad=0.2)
    draw_arrow(ax, 10.25, 7.5, 10.25, 6.2, arrow_style='<|-|>', rad=-0.2)
    
    # 4. Cross-Modal Fusion
    draw_dashed_box(ax, 13, 4, 2.5, 4, title="Fusion & Output")
    
    # Attention Fusion
    draw_block(ax, 13.25, 6.0, 2.0, 1.5, "Cross-Modal\nAttention\nFusion", color=color_fusion, edgecolor='#674EA7')
    
    # Arrows to Fusion
    draw_arrow(ax, 11.5, 8.1, 13.25, 7.0, rad=-0.2)
    draw_arrow(ax, 11.5, 5.6, 13.25, 6.75)
    draw_arrow(ax, 11.5, 3.1, 13.25, 6.5, rad=0.2)
    
    # 5. Output / Detection
    draw_block(ax, 13.25, 4.5, 2.0, 1.0, "Intelligent\nThreshold\nOptimizer", color=color_out, edgecolor='#B45F06')
    draw_arrow(ax, 14.25, 6.0, 14.25, 5.5)
    
    # Final Result
    ax.text(14.25, 3.5, "Botnet\nNode Detection", ha='center', va='center', fontsize=11, fontweight='bold', color='#CC0000', bbox=dict(boxstyle="round,pad=0.3", edgecolor='#CC0000', facecolor='#FFF0F0'))
    draw_arrow(ax, 14.25, 4.5, 14.25, 4.0)

    # Title
    plt.suptitle("V3 Final: Tri-Modal Heterogeneous Graph Neural Network for Botnet Detection", fontsize=16, fontweight='bold', y=0.95, color='#333333')
    
    # Save the figure
    plt.tight_layout()
    plt.savefig('v3_final_architecture.png', dpi=300, bbox_inches='tight')
    plt.savefig('v3_final_architecture.pdf', bbox_inches='tight')  # PDF for academic papers
    print("Architecture diagram saved as v3_final_architecture.png and .pdf")

if __name__ == '__main__':
    generate_architecture_diagram()

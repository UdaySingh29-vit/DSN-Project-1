import matplotlib.pyplot as plt
import numpy as np
import os

# Create a directory to save the graphs if it doesn't exist
output_dir = 'report_graphs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Helper function to add labels on top of bars
def autolabel(rects, ax, is_percent=False):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        label = f'{height}%' if is_percent else f'{height}'
        ax.annotate(label,
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

# ---------------------------------------------------------
# 1. FPS Graph
# ---------------------------------------------------------
scenarios_fps = ['Corridor', 'Classroom', 'Office', 'Multiple Objects', 'Moving Pedestrian']
# Mock data (you can change these values based on your actual results)
average_fps = [28, 25, 29, 18, 22] 

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(scenarios_fps, average_fps, color='#4C72B0')

ax.set_title('Average Frames Per Second Across Test Scenarios', fontsize=14)
ax.set_xlabel('Test Scenario', fontsize=12)
ax.set_ylabel('Average FPS', fontsize=12)
ax.set_ylim(0, max(average_fps) + 10)
autolabel(bars, ax)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, '1_fps_graph.png'))
plt.close()

# ---------------------------------------------------------
# 2. Latency Graph
# ---------------------------------------------------------
# Mock data (you can change these values)
latency_ms = [35, 40, 34, 55, 45] 

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(scenarios_fps, latency_ms, color='#DD8452')

ax.set_title('Average Processing Latency Across Test Scenarios', fontsize=14)
ax.set_xlabel('Test Scenario', fontsize=12)
ax.set_ylabel('Latency (ms)', fontsize=12)
ax.set_ylim(0, max(latency_ms) + 15)
autolabel(bars, ax)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, '2_latency_graph.png'))
plt.close()

# ---------------------------------------------------------
# 3. Detection Success Graph
# ---------------------------------------------------------
categories = ['Person', 'Chair', 'Bag', 'Vehicle', 'Other']
# Mock data (you can change these values)
success_rate = [96, 88, 85, 92, 78] 

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(categories, success_rate, color='#55A868')

ax.set_title('Object Detection Success Rate', fontsize=14)
ax.set_xlabel('Object Category', fontsize=12)
ax.set_ylabel('Detection Success (%)', fontsize=12)
ax.set_ylim(0, 105)
autolabel(bars, ax, is_percent=True)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, '3_detection_success_graph.png'))
plt.close()

# ---------------------------------------------------------
# 4. Environmental Performance Graph
# ---------------------------------------------------------
conditions = ['Good Lighting', 'Moderate Lighting', 'Low Lighting']
# Mock data (you can change these values)
env_success_rate = [95, 82, 60]
env_fps = [28, 24, 15]

x = np.arange(len(conditions))  # the label locations
width = 0.35  # the width of the bars

fig, ax1 = plt.subplots(figsize=(10, 6))

color1 = '#4C72B0'
ax1.set_xlabel('Lighting Condition', fontsize=12)
ax1.set_ylabel('Detection Success (%)', color=color1, fontsize=12)
rects1 = ax1.bar(x - width/2, env_success_rate, width, label='Success Rate (%)', color=color1)
ax1.tick_params(axis='y', labelcolor=color1)
ax1.set_ylim(0, 110)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color2 = '#DD8452'
ax2.set_ylabel('FPS', color=color2, fontsize=12)  # we already handled the x-label with ax1
rects2 = ax2.bar(x + width/2, env_fps, width, label='FPS', color=color2)
ax2.tick_params(axis='y', labelcolor=color2)
ax2.set_ylim(0, max(env_fps) + 10)

ax1.set_xticks(x)
ax1.set_xticklabels(conditions)
ax1.set_title('System Performance Under Different Lighting Conditions', fontsize=14)

# Adding legends
fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)

# Autolabels
def autolabel_env(rects, ax, is_percent=False):
    for rect in rects:
        height = rect.get_height()
        label = f'{height}%' if is_percent else f'{height}'
        ax.annotate(label,
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

autolabel_env(rects1, ax1, is_percent=True)
autolabel_env(rects2, ax2)

fig.tight_layout()
plt.savefig(os.path.join(output_dir, '4_environmental_performance_graph.png'))
plt.close()

print(f"Successfully generated all graphs in the '{output_dir}' directory.")

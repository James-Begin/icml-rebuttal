import matplotlib.pyplot as plt
import numpy as np


x = np.arange(1, 10)


initial_malicious_share = x / 10.0 


final_malicious_share = [0.01, 0.05, 0.03, 0.11, 0.25, 0.34, 0.51, 0.68, 0.92]

market_accuracy = [0.73, 0.65, 0.59, 0.54, 0.51, 0.47, 0.41, 0.33, 0.28]

fig, ax1 = plt.subplots(figsize=(10, 6))

color_wealth = 'tab:green'
color_baseline = 'gray'

ax1.set_xlabel('Number of Malicious Agents (out of 10)', fontsize=12)
ax1.set_ylabel('Share of Total Market Wealth', fontsize=12)
ax1.set_ylim(-0.05, 1.05)

ax1.plot(x, initial_malicious_share, color=color_baseline, linestyle='--', label='Initial Malicious Share', alpha=0.8)

ax1.plot(x, final_malicious_share, color=color_wealth, marker='o', linewidth=2, label='Final Malicious Share')


ax1.fill_between(x, initial_malicious_share, final_malicious_share,
                 color=color_wealth, alpha=0.5, label='Wealth Transfer')

ax1.tick_params(axis='y')

ax2 = ax1.twinx()  # Instantiate a second axes that shares the same x-axis
color_acc = 'tab:blue'

ax2.set_ylabel('Market Accuracy', color=color_acc, fontsize=12)
ax2.set_ylim(0, 1.0)

ax2.plot(x, market_accuracy, color=color_acc, linestyle='--', marker='s', linewidth=2, label='Market Accuracy')
ax2.tick_params(axis='y', labelcolor=color_acc)

plt.title('Resilience to Malicious Agents', fontsize=14)
ax1.grid(True, linestyle=':', alpha=0.6)

lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='lower left')

plt.tight_layout()

plt.show()
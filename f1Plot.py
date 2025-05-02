from statistics import mean
import matplotlib.pyplot as plt
import seaborn as sns

f1Scores = [
    0.99, 0.98, 0.93, 0.99, 0.95, 0.92, 0.96, 0.92, 0.98, 0.93,
    0.77, 0.96, 0.98, 0.97, 0.99, 0.86, 0.91, 0.93, 0.91, 0.90,
    0.93, 0.86, 1.00, 0.58, 0.97, 0.98, 0.98, 0.94, 0.95, 0.31,
    0.97, 0.95, 0.98, 0.98, 0.99, 0.98, 1.00, 0.99, 0.97, 0.95,
    0.98, 0.99, 0.95, 0.69, 0.97, 1.00, 0.97, 0.81, 0.96, 0.99,
    0.36, 0.92, 0.96, 0.99, 0.95, 0.99, 0.87, 0.95, 0.65, 0.93,
    0.98, 0.97, 0.92, 0.98, 0.98, 0.96, 0.83, 0.99, 0.98, 0.98,
    0.98, 0.97, 0.98, 0.56, 1.00, 0.85, 0.93, 0.59, 0.96, 0.99,
    0.99, 0.92, 0.86, 0.90, 0.96, 0.94, 0.68, 0.99, 0.84, 0.98,
    0.88, 0.91, 0.90, 0.97, 0.94, 0.97, 0.97, 0.98, 0.95, 0.93,
    0.70, 0.81, 0.96, 0.95, 0.97, 0.97, 1.00, 0.99, 0.97, 0.99,
    0.97, 0.76, 0.98, 1.00, 0.77, 0.96, 0.99, 0.97, 1.00, 0.84,
    0.97, 0.95, 0.83, 0.99, 0.93, 0.91, 0.90, 0.97, 0.94, 0.86,
    0.99, 0.98, 0.98, 1.00, 0.87, 0.59, 0.99, 0.92, 0.73, 0.84,
    0.90, 0.98, 0.97, 0.92, 0.97, 0.95, 0.97, 0.97, 0.98, 0.95,
    0.93, 0.93, 0.70, 0.99, 0.99, 0.98, 0.68, 0.99, 0.97, 0.98,
    0.96, 0.93, 0.95, 0.99, 0.97, 0.63, 0.99, 0.97, 0.95, 0.93,
    0.98, 0.95, 0.98, 0.99, 0.99, 0.99, 0.98, 0.98, 0.97, 0.93,
    0.98, 0.98, 0.98, 0.97, 0.98, 0.98, 0.97, 0.98, 0.99, 0.99,
    0.97, 0.94, 0.97, 0.99, 0.99, 0.96, 0.15, 0.96, 0.98, 0.99,
    0.92, 0.94, 0.27, 0.49
]

meanf1 = mean(f1Scores)
print("Mean F1 Score:", meanf1)

plt.figure(figsize=(8, 6))
sns.histplot(f1Scores, bins=25, kde=True, color='steelblue', edgecolor='black')

# Titles and labels
plt.title('Distribution of F1 Scores Across Classes', fontsize=14, weight='bold')
plt.xlabel('F1 Score', fontsize=12)
plt.ylabel('Number of Classes', fontsize=12)

# Customize tick sizes
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Add vertical line for mean F1 score
mean_f1 = sum(f1Scores) / len(f1Scores)
plt.axvline(mean_f1, color='red', linestyle='--', linewidth=2, label=f'Mean = {mean_f1:.2f}')
plt.legend()

plt.tight_layout()
plt.savefig('f1score_distribution.png', dpi=300)
plt.show()
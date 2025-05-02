import matplotlib.pyplot as plt

datasets = ['WiLI', 'Europarl', 'Tatoeba']
train_samples = [117_500, 160_000, 400_016]

colors = ['#66c2a5', '#fc8d62', '#8da0cb']

plt.figure(figsize=(8, 6))
wedges, texts, autotexts = plt.pie(
    train_samples,
    labels=datasets,
    autopct='%1.1f%%',
    startangle=140,
    colors=colors,
    textprops={'fontsize': 12}
)

for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')

for text in texts:
    text.set_fontweight('bold')

plt.title('Training Data Distribution by Dataset', fontsize=16, fontweight='bold')
plt.axis('equal')
plt.tight_layout()
plt.savefig('training_data_distribution_pie_chart.png', dpi=300)
plt.show()

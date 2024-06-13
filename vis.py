import json
import matplotlib.pyplot as plt


with open('results.json', 'r') as file:
    results = json.load(file)

caching_intervals = [1, 2, 3, 4, 5]

for style in results:
    fig, ax = plt.subplots()
    for category in results[style]:
        ax.plot(caching_intervals, results[style][category], label=f'{style}: {category}')
    ax.set_title('Effect of caching interval on CLIP score of LoRAs')
    ax.set_xlabel('Caching interval')
    ax.set_ylabel('CLIP score')
    ax.legend()
    ax.grid()
    fig.savefig(f'{style}_plot.png', dpi=300)
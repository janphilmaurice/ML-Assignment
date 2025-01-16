import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from cross_validation import initializeMLP_with_bestHyper

def plt_err_with_bestHyper(best_params, X, y):
    mlp = initializeMLP_with_bestHyper(best_params, X, y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    mlp.fit(X_train, y_train, epoch=200, batch_size=32)
    mlp.show_error_plot()

# draw the combination of hyper
def plt_hyper_comb(data):
    fig, axs = plt.subplots(1, 1, figsize=(8, 6))
    labels = range(1, len(data)+1, 1)

    acc = [round(x, 3) for x in data]

    bars = axs.bar(labels, acc, alpha=0.6)
    axs.plot(labels, acc, label="Mean Reliability", marker="s")
    axs.set_title("Combination of hyperparameter", fontsize=14, fontweight='bold')
    axs.spines['top'].set_linewidth(2)
    axs.spines['right'].set_linewidth(2)
    axs.spines['bottom'].set_linewidth(2)
    axs.spines['left'].set_linewidth(2)
    axs.set_xlabel("Set of combination")
    axs.set_ylabel("Accuracy")
    axs.set_xticks(labels, fontsize=20, color='#333')
    axs.set_ylim(0.5, 0.85)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height}',
                 ha='center', va='bottom')

    # Gets the current X-axis scale label
    ax = plt.gca()
    xticklabels = ax.get_xticklabels()

    colors = ['black'] * len(data)

    colors[data.index(max(data))] = 'red'

    # Apply color to the scale label
    for tick, color in zip(xticklabels, colors):
        tick.set_color(color)
    plt.savefig(
        "figure/comb_hyper.svg",
        format="svg")
    plt.show(block=False)

def plt_acc_imbalance(data):
    labels= ['-4','-3','-2','-1','0','1','2','3','4']

    plt.figure(figsize=(8, 5))

    plt.plot(data[0], label='Oversampling', marker='o', linewidth=2, color='blue')
    plt.plot(data[1], label='Undersampling', marker='o', linewidth=2, color='green')
    plt.plot(data[2], label='SMOTE', marker='o', linewidth=2, color='red')

    # 3. Add title and axis labels
    plt.title('Strategy Accuracy with different imbalance degree')
    plt.xlabel('Imbalance degree')
    plt.ylabel('Accuracy')
    #plt.xticks([i for i in range(9)], label)

    # 4. Add grids, legends, and appropriate axis ranges
    plt.grid(True)
    plt.legend(loc='best')
    plt.xticks(np.arange(9), labels)
    plt.savefig(
        "figure/acc_with_imbalance.svg",
        format="svg")
    plt.show(block=False)

def plt_s_avg(datas):
    data = [np.array(datas[0]).mean(),np.array(datas[1]).mean(),np.array(datas[2]).mean()]
    fig, axs = plt.subplots(1, 1, figsize=(6, 4))
    labels=['Oversampling','Undersampling','SMOTE']

    value=[round(x, 2) for x in data]


    # subplot1
    bars = axs.bar(labels, value, alpha=0.6, color="green")
    #axs.plot(labels, s_avg_acc, label="Mean Reliability", color="green", marker="s")
    axs.set_title("Average accuracy of strategy",fontsize=14, fontweight='bold')
    axs.spines['top'].set_linewidth(1)
    axs.spines['right'].set_linewidth(1)
    axs.spines['bottom'].set_linewidth(2)
    axs.spines['left'].set_linewidth(2)
    axs.set_ylabel("Accuracy")
    axs.set_xticks(labels, fontsize=20, color='#333')
    axs.set_ylim(0.8,1)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, height, f'{height}',
                 ha='center', va='bottom')
    plt.savefig(
        "figure/average_acc_s.svg",
        format="svg")
    plt.show(block=False)

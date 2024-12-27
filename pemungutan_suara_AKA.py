import time
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate

sizes = [10, 20, 100, 1000, 10000]
execution_times_iterative = []
execution_times_recursive = []
num_repeats = 10  # Menambahkan jumlah pengulangan untuk stabilitas pengukuran

for size in sizes:
    X, y = make_classification(n_samples=size, n_features=5, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    total_iterative_time = 0
    total_recursive_time = 0
    
    # Mengulangi untuk mendapatkan rata-rata waktu
    for _ in range(num_repeats):
        start = time.time()
        clf_iterative = DecisionTreeClassifier(criterion="gini", splitter="best")
        clf_iterative.fit(X_train, y_train)
        clf_iterative.predict(X_test)
        total_iterative_time += (time.time() - start) * 1000  # Waktu dalam ms
        
        start = time.time()
        clf_recursive = DecisionTreeClassifier(criterion="gini", splitter="random")
        clf_recursive.fit(X_train, y_train)
        clf_recursive.predict(X_test)
        total_recursive_time += (time.time() - start) * 1000  # Waktu dalam ms

    # Menyimpan rata-rata waktu untuk setiap ukuran sampel
    execution_times_iterative.append(total_iterative_time / num_repeats)
    execution_times_recursive.append(total_recursive_time / num_repeats)

data = {
    'n': sizes,
    'Waktu Rekursif (ms)': execution_times_recursive,
    'Waktu Iteratif (ms)': execution_times_iterative
}
df = pd.DataFrame(data)

table = tabulate(df, headers='keys', tablefmt='grid', showindex=False)
print(table)

plt.figure(figsize=(10, 6))

plt.plot(sizes, execution_times_iterative, label="Iterative", marker='o', color='blue', linestyle='-', linewidth=2, markersize=8)
plt.plot(sizes, execution_times_recursive, label="Recursive", marker='s', color='green', linestyle='--', linewidth=2, markersize=8)

plt.grid(True, axis='both', linestyle='--', linewidth=0.7, alpha=0.6)
plt.xticks(sizes, fontsize=12)
plt.yticks(np.round(np.linspace(0, max(max(execution_times_iterative), max(execution_times_recursive)), num=10), 4), fontsize=12)

plt.xlabel('Data Size', fontsize=14, labelpad=10)
plt.ylabel('Execution Time (ms)', fontsize=14, labelpad=10)
plt.title('Perbandingan Waktu Eksekusi Decision Tree: Iteratif vs Rekursif', fontsize=16, pad=15)

plt.legend(title='Pendekatan', loc='upper left', fontsize=12)

for i, txt in enumerate(execution_times_iterative):
    plt.text(sizes[i], execution_times_iterative[i], f'{execution_times_iterative[i]:.2f}', color='blue', ha='center', va='bottom', fontsize=10)
    
for i, txt in enumerate(execution_times_recursive):
    plt.text(sizes[i], execution_times_recursive[i], f'{execution_times_recursive[i]:.2f}', color='green', ha='center', va='top', fontsize=10)

plt.tight_layout()
plt.show()

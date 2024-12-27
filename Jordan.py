import numpy as np
import time
import matplotlib.pyplot as plt

# Untuk menyimpan data hasil
n_sizes = []
times_recursive = []
times_iterative = []

# Visualisasi data
def visualize_performance():
    plt.figure(figsize=(10, 6))
    plt.plot(n_sizes, times_recursive, marker='o', label='Recursive')
    plt.plot(n_sizes, times_iterative, marker='s', label='Iterative')
    plt.title('Execution Time Comparison: Iterative vs Recursive')
    plt.xlabel('Matrix Size (n)')
    plt.ylabel('Execution Time (milliseconds)')
    plt.grid(True)
    plt.legend()
    plt.show()


# Iterative Gauss-Jordan
def gauss_jordan_iterative(matrix):
    rows, cols = matrix.shape
    for i in range(rows):
        # Normalisasi pivot
        if matrix[i, i] == 0:  # Jika diagonal nol, coba tukar baris
            for k in range(i + 1, rows):
                if matrix[k, i] != 0:
                    matrix[[i, k]] = matrix[[k, i]]
                    break
        matrix[i] = matrix[i] / matrix[i, i]
        
        # Eliminasi elemen lain di kolom
        for j in range(rows):
            if j != i:
                factor = matrix[j, i]
                matrix[j] -= factor * matrix[i]
    return matrix

# Recursive Gauss-Jordan
def recursive_gauss_jordan(matrix, current=0):
    rows, cols = matrix.shape
    if current >= rows:  # Basis rekursi
        return matrix
    
    # Normalisasi pivot
    if matrix[current, current] == 0:
        for k in range(current + 1, rows):
            if matrix[k, current] != 0:
                matrix[[current, k]] = matrix[[k, current]]
                break
    
    matrix[current] = matrix[current] / matrix[current, current]
    
    # Eliminasi elemen lain di kolom secara rekursif
    for i in range(rows):
        if i != current:
            factor = matrix[i, current]
            matrix[i] -= factor * matrix[current]
    
    return recursive_gauss_jordan(matrix, current + 1)

# Main program untuk evaluasi
def main():
    print("=== Perbandingan Iteratif dan Rekursif pada Eliminasi Gauss-Jordan ===")
    print("Ukuran Matriks | Waktu Rekursif (ms) | Waktu Iteratif (ms)")
    print("-----------------------------------------------------------")

    for n in range(10, 101, 20):  # Ukuran matriks: 10x11, 30x31, dst.
        n_sizes.append(n)
        matrix = np.random.randint(1, 10, size=(n, n + 1)).astype(float)

        # Rekursif
        matrix_rec = matrix.copy()
        start = time.time()
        recursive_gauss_jordan(matrix_rec)
        end = time.time()
        recursive_time = (end - start) * 1000  # Waktu dalam milidetik
        times_recursive.append(recursive_time)

        # Iteratif
        matrix_iter = matrix.copy()
        start = time.time()
        gauss_jordan_iterative(matrix_iter)
        end = time.time()
        iterative_time = (end - start) * 1000  # Waktu dalam milidetik
        times_iterative.append(iterative_time)

        # Cetak hasil
        print(f"{n:<14} | {recursive_time:>17.4f} | {iterative_time:>17.4f}")

    print("\n=== Grafik Perbandingan Waktu ===")
    visualize_performance()



# Eksekusi program utama
main()

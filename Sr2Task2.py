import numpy as np
import matplotlib.pyplot as plt

# Матриця переходів для поглинаючого ланцюга
transition_matrix_absorbing = np.array([
    [1, 0, 0, 0, 0, 0, 0],
    [0.3, 0.7, 0, 0, 0, 0, 0],
    [0.4, 0.2, 0.4, 0, 0, 0, 0],
    [0.2, 0.3, 0.5, 0, 0, 0, 0],
    [0, 0.2, 0.3, 0.5, 0, 0, 0],
    [0, 0, 0.4, 0.3, 0.3, 0, 0],
    [0, 0, 0, 0, 0.3, 0.7, 0]
])

# Початковий вектор станів
initial_state = np.array([0, 1, 0, 0, 0, 0, 0])

# Функція для моделювання поглинаючого ланцюга Маркова
def simulate_absorbing_chain(transition_matrix, initial_state, max_steps=100):
    current_state = np.random.choice(len(initial_state), p=initial_state)
    steps = 0
    for _ in range(max_steps):
        if transition_matrix[current_state, current_state] == 1:  # Поглинаючий стан
            break
        current_state = np.random.choice(
            len(transition_matrix),
            p=transition_matrix[current_state]
        )
        steps += 1
    return steps

# Змоделюємо більше 100 реалізацій
num_realizations = 100
steps_to_absorption = [simulate_absorbing_chain(transition_matrix_absorbing, initial_state) for _ in range(num_realizations)]

# Обчислимо середнє значення та дисперсію
mean_steps = np.mean(steps_to_absorption)
variance_steps = np.var(steps_to_absorption)

print(f"Середнє значення часу до поглинання: {mean_steps}")
print(f"Дисперсія часу до поглинання: {variance_steps}")

# Побудуємо гістограму для розподілу часу до поглинання
plt.hist(steps_to_absorption, bins=10, edgecolor='black', alpha=0.7)
plt.title('Гістограма часу до поглинання')
plt.xlabel('Кількість кроків')
plt.ylabel('Частота')
plt.grid(axis='y', alpha=0.75)
plt.show()

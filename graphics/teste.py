valores_Gender = [
    [0.021396633, 0.000489569, 0.000443358, 0.000753232, 0.000747642, 0.000372802, 0.000397471],
    [0.021150737, 0.000513757, 0.000007853, 0.000491322, 0.000520241, 0.000013031, 0.000012787],
    [0.021924542, 0.000037142, 0.000038531, 0.000080662, 0.000077088, 0.000501350, 0.000260919]
]

valores_Age = [
    [0.021396633, 0.000489569, 0.000443358, 0.000753232, 0.000747642, 0.000372802, 0.000397471],
    [0.021150737, 0.000513757, 0.000007853, 0.000491322, 0.000520241, 0.000013031, 0.000012787],
    [0.021924542, 0.000037142, 0.000038531, 0.000080662, 0.000077088, 0.000501350, 0.000260919]
]

# Invertendo os valores em cada sublista em valores_Gender
valores_Gender = [
    [sublista[len(sublista) - i - 1] for i, _ in enumerate(sublista)] for sublista in valores_Gender
]

# Invertendo os valores em cada sublista em valores_Age
valores_Age = [
    [sublista[len(sublista) - i - 1] for i, _ in enumerate(sublista)] for sublista in valores_Age
]

print(valores_Gender)
print(valores_Age)
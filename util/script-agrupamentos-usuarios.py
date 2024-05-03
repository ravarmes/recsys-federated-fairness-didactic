def read_user_ids(file_path):
    user_ids = []
    with open(file_path, 'r') as file:
        for line in file:
            user_ids.append(int(line.strip()))
    return user_ids

def read_user_data(file_path):
    user_data = {}
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split('::')
            user_id = int(parts[0])
            gender = int(parts[1])
            age = int(parts[2])
            user_data[user_id] = (gender, age)
    return user_data

def create_dictionaries(user_ids, user_data):
    G_Gender = {}
    G_Age = {}
    G_Gender_Index = {}
    G_Age_Index = {}
    
    # Initialize dictionaries with possible Gender and Age as empty lists
    for uid in user_ids:
        if uid in user_data:
            gender, age = user_data[uid]
            if gender not in G_Gender:
                G_Gender[gender] = []
                G_Gender_Index[gender] = []
            if age not in G_Age:
                G_Age[age] = []
                G_Age_Index[age] = []
    
    # Populate the dictionaries
    for index, uid in enumerate(user_ids):
        if uid in user_data:
            gender, age = user_data[uid]
            G_Gender[gender].append(uid)
            G_Gender_Index[gender].append(index)
            G_Age[age].append(uid)
            G_Age_Index[age].append(index)
    
    return G_Gender, G_Gender_Index, G_Age, G_Age_Index

def main():
    user_ids = read_user_ids('MovieLens-UserID-Activity.txt')
    user_data = read_user_data('users.dat')
    
    G_Gender, G_Gender_Index, G_Age, G_Age_Index = create_dictionaries(user_ids, user_data)
    
    print("G_Gender:", G_Gender)
    print("G_Gender_Index:", G_Gender_Index)
    print("G_Age:", G_Age)
    print("G_Age_Index:", G_Age_Index)

if __name__ == '__main__':
    main()

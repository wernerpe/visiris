import os
import pickle 

def create_experiment_directory(world_name, n, seed):
    dirname = world_name[:-5]
    dirname = "./experiment_logs/"+dirname
    if not os.path.exists(dirname):
        os.makedirs(dirname)
        print("Directory created successfully!")
    else:
        print("Directory already exists.")
    return

def experiment_string(world_name, n, seed, approach):
    return world_name[:-5]+f"_experiment_{n}_{seed}_{approach}"

def dump_experiment_results(world_name, experiment_name, chosen_verts, regions, tind, treg):
    with open("./experiment_logs/"+world_name[:-5] + "/" + experiment_name+".log", 'wb') as f:
        A = []
        b = []
        for r in regions:
            A.append(r.A())
            b.append(r.b())
        pickle.dump({'verts': chosen_verts, 'regions': [A,b], 'tind': tind, 'treg': treg}, f)
import pickle
with open("assets/motions/walk1_subject1_portable.pkl", "rb") as f:
    m = pickle.load(f)
print(type(m["fps"]), len(m["root_pos"]), len(m["root_pos"][0]),
      len(m["root_rot"][0]), len(m["dof_pos"][0]))
# expect: int/float fps, [T,3], [T,4], [T,23]
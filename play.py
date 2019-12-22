# from utils.draw_bb import draw_bb

# print(draw_bb)
import os

root = 'data/yolo_cad_dataset_v2/labels'
s = set()
n_f = 0
for f in os.listdir(root):
    n_f += 1
    print(n_f)
    with open(os.path.join(root, f)) as _f:
        rows = _f.readlines()
        for row in rows:
            cls = row[0]
            s.add(cls)

print(s)
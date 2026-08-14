import numpy as np
from scipy.spatial import distance

class CentroidTracker:
    def __init__(self, max_disappeared=10):
        self.next_id = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared

    def update(self, rects):
        if len(rects) == 0:
            for uid in list(self.disappeared.keys()):
                self.disappeared[uid] += 1
                if self.disappeared[uid] > self.max_disappeared:
                    del self.objects[uid]
                    del self.disappeared[uid]
            return self.objects

        centroids = np.array([[(startX + endX) / 2.0, (startY + endY) / 2.0] for startX, startY, endX, endY in rects])

        if len(self.objects) == 0:
            for c in centroids:
                self.objects[self.next_id] = c
                self.disappeared[self.next_id] = 0
                self.next_id += 1
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())
            D = distance.cdist(np.array(object_centroids), centroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            used_rows, used_cols = set(), set()

            for row, col in zip(rows, cols):
                if row in used_rows or col in used_cols: continue
                uid = object_ids[row]
                self.objects[uid] = centroids[col]
                self.disappeared[uid] = 0
                used_rows.add(row)
                used_cols.add(col)

            for row in set(range(0, D.shape[0])).difference(used_rows):
                uid = object_ids[row]
                self.disappeared[uid] += 1
                if self.disappeared[uid] > self.max_disappeared:
                    del self.objects[uid]
                    del self.disappeared[uid]

            for col in set(range(0, D.shape[1])).difference(used_cols):
                self.objects[self.next_id] = centroids[col]
                self.disappeared[self.next_id] = 0
                self.next_id += 1

        return self.objects

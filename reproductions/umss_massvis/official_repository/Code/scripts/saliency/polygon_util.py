# Bucket 1-D index for semantic regions:
# 0: Annotation
# 1: Axis
# 2: Graphical Element
# 3: Legend
# 4: Object [Photograph]
# 5: Object [Pictogram]
# 6: Text [Title]
# 7: Text [Header Row]
# 8: Text [Label]
# 9: Text [Paragraph]
# 10: Text [Source]
# 11: Data [Type] 
# 12: Data (Region)
# 13: Other 

BUCKETCOLORS=[
    ( 44/255., 105/255., 154/255., 0.5),
    ( 13/255., 179/255., 158/255., 0.5),
    (239/255., 234/255.,  90/255., 0.5),
    (241/255., 196/255.,  83/255., 0.5),
    ( 22/255., 219/255., 147/255., 0.5),
    ( 22/255., 219/255., 147/255., 0.5),
    (242/255., 158/255.,  76/255., 0.5),
    ( 44/255., 105/255., 154/255., 0.5),
    (  4/255., 139/255., 168/255., 0.5),
    (131/255., 227/255., 119/255., 0.5),
    ( 84/255.,  71/255., 140/255., 0.5),
    (241/255., 196/255.,  83/255., 0.5),
    (185/255., 231/255., 105/255., 0.5),
    (185/255., 231/255., 105/255., 0.5),
    (128/255., 128/255., 128/255., 0.5)
]

def bucketColors(idx):
    return BUCKETCOLORS[idx]


BUCKETCOLORSNEW=[
    (128/255.,  64/255., 128/255., 0.5),
    (  0/255.,   0/255., 142/255., 0.5),
    (244/255.,  35/255., 232/255., 0.5),
    (102/255., 102/255., 156/255., 0.5),
    (150/255., 120/255.,  90/255., 0.5),
    (250/255., 170/255.,  30/255., 0.5),
    (220/255., 220/255.,        0, 0.5),
    (107/255., 142/255.,  35/255., 0.5),
    ( 70/255.,  70/255.,  70/255., 0.5),
]

def bucketColorsNew(idx):
    return BUCKETCOLORSNEW[idx]

def polygon_area(polygon):
    """
    compute polygon area
    polygon: list with shape [n, 2], n is the number of polygon points
    """
    area = 0
    q = polygon[-1]
    for p in polygon:
        area += p[0] * q[1] - p[1] * q[0]
        q = p
    return abs(area / 2.0)

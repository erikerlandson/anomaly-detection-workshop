import pandas as pd

def altairDF(xv, fv, labels=None, xcol="x", ycol="y", ccol="color"):
    n = len(xv)
    nf = len(fv)
    if labels is None:
        labels = ["y%d"%(j+1) for j in range(nf)]
    x = []
    y = []
    c = []
    for j in range(nf):
        x += xv
        y += [fv[j](x) for x in xv]
        c += [labels[j]]*n
    df = pd.DataFrame()
    df[xcol] = x
    df[ycol] = y
    df[ccol] = c
    return df
